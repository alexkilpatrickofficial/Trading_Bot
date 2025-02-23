import logging
import os
import pickle
import random
import glob
from filelock import FileLock, Timeout
from itertools import count
from joblib import Parallel, delayed  # For potential future parallel processing
import numpy as np
import torch
import torch.nn as nn
# Do not create DEAP creator classes at the module level!
# Instead, define an initialization function below.
from deap import base  # Creator will be imported inside the init function when needed.
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from callbacks import TensorBoardCallback, CheckpointCallback
from config import (
    LOG_DIR, GA_STATE_FILE, FEATURE_COLUMNS, ACCOUNT_METRICS, 
    TENSORBOARD_LOG_DIR, CONTINUOUS_FEATURES, BINARY_FEATURES
)
from envs.multi_stock_env import MultiStockTradingEnv
from model_saver import save_model_atomically
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# DEAP Creator Initialization
# -------------------------------------------------------------------------
def init_deap_creators():
    """
    Initialize the DEAP creator classes. Call this function once in your main script
    (inside if __name__ == '__main__':) so that these classes are defined only in the main process.
    """
    from deap import creator, base
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

torch.set_num_threads(os.cpu_count())

def get_tensorboard_gen_log_dir(current_gen, log_dir=TENSORBOARD_LOG_DIR):
    return os.path.join(log_dir, f"generation_{current_gen}")

_unique_id = count(start=1)

def init_individual(individual):
    """
    Assign a unique ID and prepare the model path for each new Individual.
    """
    individual.id = f"ind_{next(_unique_id):08x}"
    individual.evaluated = False
    individual.model_path = os.path.join('models', f'{individual.id}.zip')
    logger.debug(f"Initialized Individual ID: {individual.id}")
    return individual

# ---------------------------------------------------------------------
# shape_is_valid: checks CNN/LSTM shape, attention heads, etc.
# ---------------------------------------------------------------------
def shape_is_valid(chromosome, window_size=16):
    """
    Return True if the chromosome’s parameters yield a valid model architecture.
    Checks include kernel size relative to window size and proper divisibility for attention heads.
    """
    # Check CNN kernel size does not exceed window size.
    cnn_kernel_size = int(chromosome[5])
    if cnn_kernel_size > window_size:
        return False

    # If attention is enabled (index 15 == 1), check LSTM hidden size and num_heads.
    use_attention_flag = int(chromosome[15])
    if use_attention_flag == 1:
        lstm_hidden_size = int(chromosome[7])
        num_heads = int(chromosome[12])
        if num_heads < 1 or num_heads > lstm_hidden_size:
            return False
        if (lstm_hidden_size % num_heads) != 0:
            return False

    return True

# ---------------------------------------------------------------------
# custom_mutate_enhanced: advanced mutation that reverts invalid shape
# ---------------------------------------------------------------------
def custom_mutate_enhanced(individual, mu=0, sigma=0.125, indpb=0.2,
                           use_feature_extractor=True, window_size=16):
    """
    Shuffle and mutate each gene with a given probability.
    Genes 16 and 17 (post-fusion LSTM parameters) are not mutated here.
    If the mutation causes an invalid architecture (as determined by shape_is_valid),
    the change is reverted.
    """
    attention_flag_index = 15
    num_heads_index = 12

    # Define indices for floating-point genes and integer genes.
    float_indices = [0, 1, 2, 13, 14]  # learning_rate, gamma, ent_coef, reward_multiplier, dropout_prob
    int_indices = [3, 4, 5, 6, 7, 8, 9, attention_flag_index, num_heads_index]
    gene_indices = float_indices + int_indices
    random.shuffle(gene_indices)

    for i in gene_indices:
        if random.random() < indpb:
            old_value = individual[i]
            if i in float_indices:
                individual[i] += random.gauss(mu, sigma)
                if i == 0:  # learning_rate
                    individual[i] = max(1e-8, min(1e-2, individual[i]))
                elif i == 1:  # gamma
                    individual[i] = max(0.3, min(0.9999, individual[i]))
                elif i == 2:  # ent_coef
                    individual[i] = max(1e-7, min(0.1, individual[i]))
                elif i == 13:  # reward_multiplier
                    individual[i] = max(0.1, min(2.0, individual[i]))
                elif i == 14:  # dropout_prob
                    individual[i] = max(0.1, min(0.5, individual[i]))
            else:
                # Integer genes.
                mutation = random.choice([-1, 1])
                individual[i] += mutation
                if i == 3:  # cnn_num_layers
                    individual[i] = max(3, min(128, individual[i]))
                elif i == 4:  # cnn_filters
                    individual[i] = max(8, min(512, individual[i]))
                elif i == 5:  # cnn_kernel_size
                    if individual[i] < 4:
                        individual[i] = 4
                    if individual[i] % 4 != 0:
                        individual[i] += (4 - individual[i] % 4)
                elif i == 6:  # cnn_stride
                    individual[i] = max(1, min(5, individual[i]))
                elif i == 7:  # lstm_hidden_size
                    individual[i] = max(12, min(256, individual[i]))
                    if individual[i] % 4 != 0:
                        individual[i] += (4 - individual[i] % 4)
                elif i == 8:  # lstm_num_layers
                    individual[i] = max(4, min(128, individual[i]))
                elif i == 9:  # fc_hidden_size
                    individual[i] = max(12, min(256, individual[i]))
                elif i == attention_flag_index:  # use_attention_flag
                    if individual[i] not in [0, 1]:
                        individual[i] = 1 if old_value == 0 else 0
                elif i == num_heads_index:  # num_heads
                    if int(individual[attention_flag_index]) == 1:
                        max_heads = len(CONTINUOUS_FEATURES)
                        individual[i] = max(3, min(max_heads, individual[i]))
                    else:
                        individual[i] = 0
            if not shape_is_valid(individual, window_size=window_size):
                individual[i] = old_value

    return (individual,)

def setup_toolbox():
    from deap import tools, base
    toolbox = base.Toolbox()

    # Register genes. The chromosome now includes additional genes for the post-fusion LSTM.
    toolbox.register("learning_rate", lambda: random.uniform(1e-8, 1e-2))
    toolbox.register("gamma", lambda: random.uniform(0.3, 0.9999))
    toolbox.register("ent_coef", lambda: random.uniform(1e-8, 0.1))
    toolbox.register("cnn_num_layers", lambda: random.randint(3, 128))
    toolbox.register("cnn_filters", lambda: random.randint(8, 512))
    toolbox.register("cnn_kernel_size", lambda: random.randint(4, 26))
    toolbox.register("cnn_stride", lambda: 1)
    toolbox.register("lstm_hidden_size", lambda: random.randint(12, 256))
    toolbox.register("lstm_num_layers", lambda: random.randint(4, 48))
    toolbox.register("fc_hidden_size", lambda: random.randint(12, 256))
    toolbox.register("total_timesteps", lambda: random.randint(80000, 120000))
    toolbox.register("eval_freq", lambda: random.randint(1000, 2000))
    toolbox.register("num_heads", lambda: random.randint(3, len(CONTINUOUS_FEATURES)))
    toolbox.register("reward_multiplier", lambda: random.uniform(0.1, 2.0))
    toolbox.register("dropout_prob", lambda: random.uniform(0.1, 0.5))
    toolbox.register("use_attention_flag", lambda: random.randint(0, 1))
    # Register additional genes for the post-fusion LSTM with separate ranges.
    toolbox.register("post_lstm_hidden_size", lambda: random.randint(4, 256))
    toolbox.register("post_lstm_num_layers", lambda: random.randint(4, 128))

    toolbox.register(
        "individual",
        tools.initCycle,
        __import__("deap").creator.Individual,
        (
            toolbox.learning_rate,           # 0
            toolbox.gamma,                   # 1
            toolbox.ent_coef,                # 2
            toolbox.cnn_num_layers,          # 3
            toolbox.cnn_filters,             # 4
            toolbox.cnn_kernel_size,         # 5
            toolbox.cnn_stride,              # 6
            toolbox.lstm_hidden_size,        # 7
            toolbox.lstm_num_layers,         # 8
            toolbox.fc_hidden_size,          # 9
            toolbox.total_timesteps,         # 10
            toolbox.eval_freq,               # 11
            toolbox.num_heads,               # 12
            toolbox.reward_multiplier,       # 13
            toolbox.dropout_prob,            # 14
            toolbox.use_attention_flag,      # 15
            toolbox.post_lstm_hidden_size,   # 16 - post-fusion LSTM hidden size
            toolbox.post_lstm_num_layers       # 17 - post-fusion LSTM number of layers
        ),
        n=1,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate_enhanced, mu=0, sigma=0.125, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)
    return toolbox

def calculate_sharpe_ratio(rewards, risk_free_rate=0.0):
    rewards = np.array(rewards)
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    if std_reward == 0:
        return 0.0
    return (mean_reward - risk_free_rate) / std_reward

def save_ga_state(population, generation, filepath):
    try:
        lock_path = f"{filepath}.lock"
        with FileLock(lock_path):
            with open(filepath, 'wb') as f:
                pickle.dump({'population': population, 'generation': generation}, f)
        logger.info(f"GA state successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save GA state to {filepath}: {e}")

def load_latest_ga_state(base_filepath=GA_STATE_FILE, lock_timeout=10):
    checkpoint_dir = os.path.dirname(base_filepath) or LOG_DIR
    base, ext = os.path.splitext(os.path.basename(base_filepath))
    pattern = f"{base}_gen_*{ext}"
    checkpoint_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, pattern)),
        key=lambda x: int(x.split('_gen_')[1].split(ext)[0])
    )
    if not checkpoint_files:
        logger.info("No existing GA state found. Starting fresh.")
        return None, 1
    latest_checkpoint = checkpoint_files[-1]
    lock_path = f"{latest_checkpoint}.lock"
    try:
        with FileLock(lock_path, timeout=lock_timeout):
            with open(latest_checkpoint, 'rb') as f:
                state = pickle.load(f)
        population = state['population']
        generation = state['generation']
        logger.info(f"GA state loaded from {latest_checkpoint} | Generation: {generation} | Population Size: {len(population)}")
        return population, generation
    except Exception as e:
        logger.error(f"Failed to load GA state from {latest_checkpoint}: {e}. Starting fresh.")
        return None, 1

# -----------------------------------------------------------------
# SelfAttention & Updated Feature Extractor with Post-Fusion LSTM
# -----------------------------------------------------------------
class SelfAttention(nn.Module):
    """
    Self-Attention module used in the feature extractor.
    """
    def __init__(self, embed_size, heads=4):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        if self.head_dim * heads != embed_size:
            raise ValueError(
                f"Embedding size {embed_size} not divisible by heads {heads}. Please clamp heads or hidden size so that (hidden_size % heads == 0)."
            )
        self.values  = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys    = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out  = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # Reshape for multi-head.
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

class DynamicCNNLSTMAttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    A combined CNN + LSTM + optional Self-Attention feature extractor with a post-fusion LSTM.

    The architecture:
      - Processes continuous features with CNN layers.
      - Passes CNN outputs through a primary LSTM (with optional attention).
      - Aggregates the continuous features (via mean over time).
      - Processes binary features with a small fully connected network.
      - Fuses the continuous and binary streams.
      - Processes the fused vector with a post-fusion LSTM.
        (The post-fusion LSTM hyperparameters are evolved using separate ranges.)
      - Passes the post-LSTM output through final fully connected layers.
    """
    def __init__(self, observation_space, chromosome, features_dim=256, device="cuda"):
        super(DynamicCNNLSTMAttentionFeatureExtractor, self).__init__(observation_space, features_dim)
        self.device = torch.device(device)
        self.window_size = observation_space.shape[0]
        # Support for 2D or 3D observations.
        if len(observation_space.shape) == 2:
            self.original_features_dim = observation_space.shape[1]
        elif len(observation_space.shape) == 3:
            self.threeD_shape = observation_space.shape
        else:
            raise ValueError(f"Unsupported observation shape: {observation_space.shape}")

        # Parse primary architecture hyperparameters from chromosome.
        cnn_num_layers = int(chromosome[3])
        cnn_filters = int(chromosome[4])
        cnn_kernel_size = int(chromosome[5])
        cnn_stride = int(chromosome[6])
        lstm_hidden_size = int(chromosome[7])
        lstm_num_layers = int(chromosome[8])
        fc_hidden_size = int(chromosome[9])
        dropout_prob = float(chromosome[14])
        use_attention = bool(int(chromosome[15]))
        num_heads = int(chromosome[12])
        # Validate post-fusion LSTM hyperparameters using a separate range.
        chromosome[16] = max(4, min(256, int(chromosome[16])))   # post_fusion lstm_hidden
        chromosome[17] = max(4, min(128, int(chromosome[17])))    # post_fusion lstm_layers
        post_lstm_hidden_size = int(chromosome[16])
        post_lstm_num_layers = int(chromosome[17])

        logger.debug(
            f"Chromosome hyperparams: cnn_num_layers={cnn_num_layers}, cnn_filters={cnn_filters}, "
            f"cnn_kernel={cnn_kernel_size}, stride={cnn_stride}, lstm_hidden_size={lstm_hidden_size}, "
            f"lstm_layers={lstm_num_layers}, fc_hidden_size={fc_hidden_size}, dropout={dropout_prob}, "
            f"num_heads={num_heads}, use_attention={use_attention}, "
            f"post_lstm_hidden_size={post_lstm_hidden_size}, post_lstm_num_layers={post_lstm_num_layers}"
        )

        # Build CNN layers.
        cnn_layers = []
        in_channels = 16  # Assumes the first 16 columns are continuous.
        for _ in range(cnn_num_layers):
            cnn_layers.append(nn.Conv1d(in_channels, cnn_filters, kernel_size=cnn_kernel_size,
                                        stride=cnn_stride, padding=cnn_kernel_size // 2))
            cnn_layers.append(nn.BatchNorm1d(cnn_filters))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout_prob))
            in_channels = cnn_filters
        self.cnn = nn.Sequential(*cnn_layers).to(self.device)

        # Build the primary LSTM.
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=(dropout_prob if lstm_num_layers > 1 else 0.0)
        ).to(self.device)

        # Optionally add self-attention.
        self.use_attention = use_attention
        if self.use_attention and num_heads > 0:
            self.attention = SelfAttention(embed_size=lstm_hidden_size, heads=num_heads).to(self.device)
        else:
            self.attention = None

        # Process binary features with a small FC network.
        self.binary_fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        ).to(self.device)

        # Fuse continuous and binary features.
        # Continuous features are aggregated (mean over time) to a vector of size lstm_hidden_size.
        # Binary features (after processing) are of size 16.
        # Fused vector has dimension (lstm_hidden_size + 16).
        self.post_lstm = nn.LSTM(
            input_size=lstm_hidden_size + 16,
            hidden_size=post_lstm_hidden_size,
            num_layers=post_lstm_num_layers,
            batch_first=True,
            dropout=(dropout_prob if post_lstm_num_layers > 1 else 0.0)
        ).to(self.device)
        # Final fully connected layers: input dimension is post_lstm_hidden_size.
        self.fusion_fc = nn.Sequential(
            nn.Linear(post_lstm_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_size, features_dim),
            nn.ReLU()
        ).to(self.device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Incoming observation shape: {observations.shape}")
        batch_size = 2 # observations.shape[0]

        # Unflatten if necessary.
        if len(observations.shape) == 2:
            flattened_dim = observations.shape[1]
            expected_flat_dim = self.window_size * self.original_features_dim
            if flattened_dim == expected_flat_dim:
                obs_unflattened = observations.view(batch_size, self.window_size, self.original_features_dim)
            else:
                raise ValueError(
                    f"Mismatch in flattened shape {observations.shape}, "
                    f"expected {self.window_size}*{self.original_features_dim}={expected_flat_dim}"
                )
        elif len(observations.shape) == 3:
            obs_unflattened = observations
        else:
            raise ValueError("Unsupported observation rank (not 2D or 3D).")

        # Separate continuous and binary features.
        cont_dim = 16
        bin_dim = 8
        cont_obs = obs_unflattened[:, :, :cont_dim].to(self.device)
        bin_obs = obs_unflattened[:, :, cont_dim:cont_dim+bin_dim].to(self.device)

        # Process continuous features through the CNN.
        x = cont_obs.permute(0, 2, 1)  # (batch, cont_dim, window_size)
        x = self.cnn(x)              # (batch, cnn_filters, window_size)
        x = x.permute(0, 2, 1)         # (batch, window_size, cnn_filters)

        # Process through the primary LSTM.
        lstm_out, _ = self.lstm(x)     # (batch, window_size, lstm_hidden_size)

        # Optionally apply self-attention.
        if self.use_attention and self.attention is not None:
            attn_out = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = lstm_out + attn_out

        # Aggregate continuous features (mean over time).
        cont_features = lstm_out.mean(dim=1)  # (batch, lstm_hidden_size)

        # Process binary features (mean over time).
        bin_avg = bin_obs.mean(dim=1)          # (batch, bin_dim)
        bin_features = self.binary_fc(bin_avg)   # (batch, 16)

        # Fuse continuous and binary features.
        fused = torch.cat([cont_features, bin_features], dim=1)  # (batch, lstm_hidden_size + 16)

        # Process fused features through the post-fusion LSTM.
        fused_seq = fused.unsqueeze(1)   # (batch, 1, lstm_hidden_size + 16)
        post_lstm_out, _ = self.post_lstm(fused_seq)  # (batch, 1, post_lstm_hidden_size)
        post_features = post_lstm_out.squeeze(1)      # (batch, post_lstm_hidden_size)

        # Final fully connected layers.
        final_features = self.fusion_fc(post_features)  # (batch, features_dim)
        return final_features

# ---------------------------------------------------------------------
# Evaluation, Chromosome Validation, and GA State Functions
# ---------------------------------------------------------------------
def evaluate(individual, **kwargs):
    """
    Evaluate an Individual by training (or evaluating) a PPO model and returning (reward, sharpe, -volatility).

    New parameters:
      - do_train (bool): If True (default), training continues; if False, only evaluation is performed.
      - min_train_timesteps (int): If the loaded model already has at least this many timesteps, skip training.
    """
    from callbacks import EarlyStoppingCallback

    do_train = kwargs.get("do_train", True)
    min_train_timesteps = kwargs.get("min_train_timesteps", 50000)

    stock_data = kwargs.get('stock_data', None)
    window_size = kwargs.get('window_size', 16)
    log_dir = kwargs.get('LOG_DIR', LOG_DIR)
    current_gen = kwargs.get('current_gen', 1)

    if stock_data is None:
        logger.error("No 'stock_data' provided to the evaluate function.")
        return (-1e6, -1e6, 1e6)
    logger.info(f"Evaluating Individual ID: {individual.id}")

    try:
        validated_chromosome = validate_chromosome(
            chromosome=individual,
            input_dim=len(stock_data["BTCUSD"].columns),
            window_size=window_size
        )
    except Exception as e:
        logger.error(f"Error validating chromosome for Individual {individual.id}: {e}")
        return (-1e6, -1e6, 1e6)

    total_timesteps = int(validated_chromosome[10])
    eval_freq = int(validated_chromosome[11])
    forced_gamma = 0.99
    forced_ent_coef = 0.02

    policy_kwargs = {
        "features_extractor_class": DynamicCNNLSTMAttentionFeatureExtractor,
        "features_extractor_kwargs": {
            "chromosome": validated_chromosome,
            "features_dim": 256,
            "device": 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    }

    try:
        env_kwargs = {
            "stock_data": stock_data,
            "chromosome": validated_chromosome,
            "window_size": window_size,
            "initial_balance": 10000.0,
            "scaler_path": os.path.join(log_dir, "scaler.pkl"),
        }
        vec_env = SubprocVecEnv([lambda: MultiStockTradingEnv(**env_kwargs) for _ in range(4)])
    except Exception as e:
        logger.error(f"Error initializing environment for Individual {individual.id}: {e}")
        return (-1e6, -1e6, 1e6)

    model_path = individual.model_path
    partial_checkpoints = sorted(
        glob.glob(model_path.replace(".zip", "_step*.zip")),
        key=lambda x: int(x.split("_step")[1].split(".zip")[0])
    )

    from stable_baselines3 import PPO
    model = None
    if partial_checkpoints:
        latest_partial = partial_checkpoints[-1]
        logger.info(f"Resuming training for {individual.id} from partial checkpoint: {latest_partial}")
        try:
            model = PPO.load(latest_partial, env=vec_env, reset_num_timesteps=False)
            logger.info(f"Resumed model has {model.num_timesteps} timesteps.")
        except Exception as e:
            logger.error(f"Error loading partial checkpoint for {individual.id}: {e}")
    if model is None and os.path.exists(model_path):
        logger.info(f"Attempting to load existing model for Individual {individual.id} from {model_path}")
        try:
            model = PPO.load(model_path, env=vec_env, reset_num_timesteps=False)
            logger.info(f"Resumed model has {model.num_timesteps} timesteps.")
        except Exception as e:
            logger.error(f"Error loading full model for {individual.id}: {e}")
    if model is None:
        logger.info(f"No valid saved model found for Individual {individual.id}. Creating a new PPO model.")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=os.path.join(get_tensorboard_gen_log_dir(current_gen), f"ind_{individual.id}"),
            learning_rate=validated_chromosome[0],
            gamma=forced_gamma,
            ent_coef=forced_ent_coef,
            policy_kwargs=policy_kwargs,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            n_steps=2048,
        )

    if do_train and model.num_timesteps < min_train_timesteps:
        from callbacks import TensorBoardCallback, CheckpointCallback, EarlyStoppingCallback
        tensorboard_callback = TensorBoardCallback(
            model_name="PPO_TradingBot",
            run_id=f"gen_{current_gen}_ind_{individual.id}",
            log_dir=get_tensorboard_gen_log_dir(current_gen),
        )
        checkpoint_callback = CheckpointCallback(save_freq=10000, model_path=model_path, verbose=1)
        early_stopping_callback = EarlyStoppingCallback(patience=100, repeat_threshold=12, verbose=1)
        callback_list = CallbackList([tensorboard_callback, checkpoint_callback, early_stopping_callback])
        try:
            logger.info(f"Training model for Individual {individual.id} starting from {model.num_timesteps} timesteps.")
            model.learn(total_timesteps=total_timesteps, callback=callback_list, reset_num_timesteps=False)
        except KeyboardInterrupt:
            logger.info(f"Training interrupted for Individual {individual.id}. Saving current state.")
            step_str = f"_step{model.num_timesteps}"
            partial_model_path = model_path.replace(".zip", f"{step_str}.zip")
            save_model_atomically(model, partial_model_path)
            raise
    else:
        logger.info(f"Skipping training for Individual {individual.id} (do_train={do_train} and model.num_timesteps={model.num_timesteps}).")

    save_model_atomically(model, model_path)
    logger.info(f"Model saved successfully at {model_path} for Individual {individual.id}")

    obs = vec_env.reset()
    rewards = []
    abnormal_flag = False
    for _ in range(total_timesteps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, infos = vec_env.step(action)
        rewards.append(reward[0])
        if isinstance(done, np.ndarray):
            if done.any():
                info0 = infos[0]
                if info0.get("abnormal", False):
                    abnormal_flag = True
                    break
                obs = vec_env.reset()
        elif done:
            abnormal_flag = True
            break

    total_reward = sum(rewards)
    sharpe_ratio = calculate_sharpe_ratio(rewards)
    volatility = np.std(rewards)
    if abnormal_flag:
        logger.warning(f"Individual {individual.id} terminated abnormally during evaluation.")
        return (-1e6, -1e6, 1e6)
    logger.info(
        f"Fitness for Individual {individual.id}: Total Reward={total_reward:.2f}, Sharpe={sharpe_ratio:.2f}, Volatility={volatility:.2f}"
    )
    return (total_reward, sharpe_ratio, -volatility)

def validate_chromosome(chromosome, input_dim, window_size=16, padding=2, use_feature_extractor=True):
    """
    Validate & clamp each chromosome gene to prevent shape errors.
    Also auto-adjust attention heads if use_attention is off or if lstm_hidden is not divisible.
    """
    if not isinstance(chromosome, list):
        raise TypeError("Chromosome must be a list.")
    if len(chromosome) < 18:
        raise ValueError(f"Chromosome must have at least 18 elements, got {len(chromosome)}.")

    use_attention = bool(int(chromosome[15]))

    # Clamp primary hyperparameters.
    chromosome[0] = max(1e-8, min(1e-2, chromosome[0]))      # LR
    chromosome[1] = max(0.3, min(0.9999, chromosome[1]))     # gamma
    chromosome[2] = max(1e-7, min(0.1, chromosome[2]))       # ent_coef
    chromosome[3] = max(3, min(128, int(chromosome[3])))      # cnn_num_layers
    chromosome[4] = max(4, min(512, int(chromosome[4])))      # cnn_filters
    chromosome[5] = max(4, min(window_size, int(chromosome[5])))  # cnn_kernel_size
    chromosome[6] = max(1, min(5, int(chromosome[6])))       # cnn_stride
    chromosome[7] = max(12, min(412, int(chromosome[7])))    # lstm_hidden
    chromosome[8] = max(8, min(128, int(chromosome[8])))     # lstm_layers
    chromosome[9] = max(6, min(256, int(chromosome[9])))     # fc_hidden
    chromosome[10] = max(80000, min(120000, int(chromosome[10])))  # timesteps
    chromosome[11] = max(1000, min(5000, int(chromosome[11])))  # eval_freq

    # Clamp attention heads if attention is enabled.
    if use_attention:
        max_heads = len(CONTINUOUS_FEATURES)
        chromosome[12] = max(1, min(max_heads, int(chromosome[12])))
    else:
        chromosome[12] = 0

    chromosome[13] = max(0.01, min(3.0, chromosome[13]))  # reward_multiplier
    chromosome[14] = max(0.1, min(0.5, chromosome[14]))    # dropout

    if use_attention:
        lstm_hidden = chromosome[7]
        heads = chromosome[12]
        if heads > lstm_hidden:
            heads = lstm_hidden
        if lstm_hidden % heads != 0:
            heads = 1
        chromosome[12] = max(1, heads)
    
    # Validate post-fusion LSTM hyperparameters using a separate range.
    chromosome[16] = max(4, min(256, int(chromosome[16])))   # post-fusion lstm_hidden
    chromosome[17] = max(4, min(128, int(chromosome[17])))    # post-fusion lstm_layers

    return list(chromosome)
