"""
This module provides the evaluation function for a given individual.
It creates (or resumes) a PPO model based on the chromosome (via the DynamicHybridFeatureExtractor)
and trains/evaluates it.
Heavy penalties are returned if known errors occur (e.g. abnormal terminations).
"""

import os
import glob
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from ga_utils import get_tensorboard_gen_log_dir, calculate_sharpe_ratio, save_ga_state
from chromosome import validate_chromosome, modular_shape_is_valid, random_chromosome
from models import DynamicHybridFeatureExtractor
from envs.multi_stock_env import MultiStockTradingEnv  # Adjust if needed
from model_saver import save_model_atomically
from config import TENSORBOARD_LOG_DIR, CONTINUOUS_FEATURES, BINARY_FEATURES, TOTAL_FEATURE_COUNT

# Import our custom hybrid policy.
from hybrid_policy import HybridPolicy

HEAVY_PENALTY = (-1e6, -1e6, 1e6)

def evaluate(individual, **kwargs):
    """
    Train and evaluate an individual model using a single environment instance.
    
    Parameters:
        individual: The individual (chromosome) to evaluate.
        kwargs: Optional keyword arguments:
            - do_train (bool): Whether to train the model.
            - min_train_timesteps (int): Minimum timesteps required for training.
            - stock_data: A dictionary of stock data.
            - window_size (int): The window size for observations.
            - current_gen (int): Current GA generation.
            - LOG_DIR (str): Base directory for logs.
    
    Returns:
        tuple: (total_reward, sharpe, -volatility) or a heavy penalty tuple if errors occur.
    """
    # Retrieve keyword arguments
    do_train = kwargs.get("do_train", True)
    min_train_timesteps = kwargs.get("min_train_timesteps", 50000)
    stock_data = kwargs.get("stock_data", None)
    window_size = kwargs.get("window_size", 16)
    current_gen = kwargs.get("current_gen", 1)
    log_dir = kwargs.get("LOG_DIR", os.path.join(os.getcwd(), "logs"))
    
    if stock_data is None:
        print("No stock_data provided to evaluate()!")
        return HEAVY_PENALTY

    print(f"\n=== Evaluating Individual {individual.id} ===")
    # Log basic evaluation info
    print("Starting environment feature detection...")
    
    # 1) Create a temporary environment to detect per-timestep feature count.
    try:
        dummy_chrom = random_chromosome(n_features=TOTAL_FEATURE_COUNT)
        print("Dummy chromosome:", dummy_chrom)
        tmp_env = MultiStockTradingEnv(
            stock_data=stock_data,
            window_size=window_size,
            chromosome=dummy_chrom,
            include_past_actions=False
        )
        flat_shape = tmp_env.observation_space.shape[0]
        print("Flat observation space shape:", tmp_env.observation_space.shape)
        per_timestep_features = flat_shape // window_size
        print("Per-timestep feature count detected:", per_timestep_features)
        tmp_env.close()
    except Exception as e:
        print(f"Env creation error (feature detection): {e}")
        return HEAVY_PENALTY

    # 2) Validate the chromosome.
    try:
        validated_chrom = validate_chromosome(individual, input_dim=per_timestep_features, window_size=window_size)
        print("Validated chromosome:", validated_chrom)
    except Exception as e:
        print(f"Chromosome validation error for {individual.id}: {e}")
        return HEAVY_PENALTY

    if not modular_shape_is_valid(validated_chrom, window_size, input_dim=per_timestep_features):
        print(f"Invalid architecture shape for {individual.id}")
        return HEAVY_PENALTY

    total_timesteps = int(validated_chrom[-3])
    print("Total timesteps for training:", total_timesteps)
    forced_gamma = 0.99
    forced_ent_coef = 0.02

    # Build policy keyword arguments.
    policy_kwargs = {
        "features_extractor_class": DynamicHybridFeatureExtractor,
        "features_extractor_kwargs": {
            "chromosome": validated_chrom,
            "features_dim": 256,  # Force latent dimension to 256
            "device": 'cuda'
        },
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        "lambda_aux": 0.5
    }
    print("Policy kwargs:", policy_kwargs)

    # Environment keyword arguments (disable past actions for consistency)
    env_kwargs = {
        "stock_data": stock_data,
        "window_size": window_size,
        "initial_balance": 10000.0,
        "scaler_path": os.path.join(log_dir, "scaler.pkl"),
        "chromosome": validated_chrom,
        "include_past_actions": False
    }
    print("Environment kwargs:", env_kwargs)

    # 3) Create a single (non-vectorized) environment.
    try:
        env = MultiStockTradingEnv(**env_kwargs)
        print("Single environment created successfully.")
    except Exception as e:
        print(f"Error initializing environment for {individual.id}: {e}")
        return HEAVY_PENALTY

    model_path = individual.model_path
    model = None
    partial_checkpoints = sorted(
        glob.glob(model_path.replace(".zip", "_step*.zip")),
        key=lambda x: int(x.split("_step")[1].split(".zip")[0])
    )
    if partial_checkpoints:
        latest_partial = partial_checkpoints[-1]
        print(f"Resuming from partial checkpoint: {latest_partial}")
        try:
            model = PPO.load(latest_partial, env=env, reset_num_timesteps=False)
        except Exception as e:
            print(f"Error loading partial checkpoint: {e}")
    if model is None and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        try:
            model = PPO.load(model_path, env=env, reset_num_timesteps=False)
        except Exception as e:
            print(f"Error loading model: {e}")
    if model is None:
        print(f"Creating new PPO model for {individual.id}")
        model = PPO(
            policy=HybridPolicy,
            env=env,
            verbose=1,
            tensorboard_log=get_tensorboard_gen_log_dir(current_gen, log_dir=TENSORBOARD_LOG_DIR) + f"/ind_{individual.id}",
            learning_rate=validated_chrom[0],
            gamma=forced_gamma,
            ent_coef=forced_ent_coef,
            policy_kwargs=policy_kwargs,
            device='cuda',
            n_steps=2048,
        )
        print("New PPO model created.")

    # 4) Training phase.
    try:
        if do_train and model.num_timesteps < min_train_timesteps:
            from callbacks import TensorBoardCallback, CheckpointCallback, EarlyStoppingCallback
            tb_callback = TensorBoardCallback(
                model_name="PPO_TradingHybrid",
                run_id=f"gen_{current_gen}_ind_{individual.id}",
                log_dir=get_tensorboard_gen_log_dir(current_gen, log_dir=TENSORBOARD_LOG_DIR)
            )
            ckpt_callback = CheckpointCallback(save_freq=5000, model_path=model_path, verbose=1)
            early_stop = EarlyStoppingCallback(patience=20, repeat_threshold=32, verbose=1)
            callback_list = CallbackList([tb_callback, ckpt_callback, early_stop])
            print(f"Training model for {individual.id} starting from {model.num_timesteps} timesteps.")
            model.learn(total_timesteps=total_timesteps, callback=callback_list, reset_num_timesteps=False)
            print(f"Training completed. Model reached {model.num_timesteps} timesteps.")
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt: saving partial model for {individual.id}")
        step_str = f"_step{model.num_timesteps}"
        partial_model_path = model_path.replace(".zip", f"{step_str}.zip")
        save_model_atomically(model, partial_model_path)
        raise
    except Exception as e:
        print(f"Training error for {individual.id}: {e}")
        return HEAVY_PENALTY

    save_model_atomically(model, model_path)
    print(f"Model saved at {model_path} for {individual.id}")

    # 5) Evaluation phase.
    try:
        obs, _ = env.reset()
        print("Initial observation shape:", obs.shape)  # Expected shape: (1, obs_dim)
        rewards = []
        abnormal = False
        normal_termination = False
        for step in range(total_timesteps):
            action, _ = model.predict(obs, deterministic=True)
            print(f"Step {step}: Action shape: {action.shape}")  # Expected shape: (1, action_dim)

            if action.ndimension() == 1:
                action = action.unsqueeze(0)
            print(f"Step {step}: Action reshaped to: {action.shape}")

            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {step}: New obs shape: {obs.shape}, Reward: {reward}, Done: {done}")
            rewards.append(reward if isinstance(reward, float) else reward[0])
            print("Info:", info)

            if done.any():
                if any(info_item.get("end_of_data", False) for info_item in info):
                    print("End of data reached. Ending episode normally.")
                    normal_termination = True
                    break
                elif any(info_item.get("abnormal", False) for info_item in info):
                    abnormal = True
                    break
                obs, _ = env.reset()
                print("Environment reset. New obs shape:", obs.shape)

        if abnormal:
            print("Abnormal termination detected. Applying heavy penalty.")
            return HEAVY_PENALTY
        
        total_reward = sum(rewards)
        sharpe = calculate_sharpe_ratio(rewards)
        volatility = np.std(rewards)
        print(f"Individual {individual.id} evaluation complete => Reward={total_reward:.2f}, Sharpe={sharpe:.2f}, Volatility={volatility:.2f}")
        return (total_reward, sharpe, -volatility)

    except Exception as e:
        print(f"Evaluation error for {individual.id}: {e}")
        return HEAVY_PENALTY
