import os
import random
from deap import base, creator, tools
from itertools import count
from config import TOTAL_FEATURE_COUNT, EXTRA_FEATURES_PER_TIMESTEP
import logging
import glob
import pickle
import numpy as np

logger = logging.getLogger(__name__)

# Global GA parameters
MAX_BLOCKS = 6
BLOCK_TYPES = {
    0: "CNN", 1: "GRU", 2: "LSTM", 3: "TCN",
    4: "TRANSFORMER", 5: "MLP", 6: "NONE"
}

# Constants for top-level hyperparameters.
PARAM_CONSTRAINTS = {
    'learning_rate': (1e-5, 1e-2),
    'gamma': (0.8, 0.999),
    'ent_coef': (1e-6, 0.05),
    'timesteps': (80000, 180000),
    'reward_multiplier': (0.1, 2.0),
    'dropout': (0.1, 0.4)
}

# New constraints for the compression transformer parameters.
COMP_TRANSFORMER_CONSTRAINTS = {
    'num_layers': (24, 512),
    'nhead': (3, 248),
    'dropout': (0.1, 0.4)
}

# New constraints for the output transformer parameters.
OUT_TRANSFORMER_CONSTRAINTS = {
    'num_layers': (2, 256),
    'nhead': (2, 24),
    'hidden': (6, 32)
}

# Architectural constraints for the modular blocks.
ARCH_CONSTRAINTS = {
    'min_hidden_size': 8,
    'max_hidden_size': 256,
    'max_kernel_size': 16,
    'max_heads': 8,
    'max_layers': 256
}

# Number of extra genes for the compression transformer.
COMP_TRANSFORMER_PARAMS_COUNT = 3
# Number of extra genes for the output transformer.
OUT_TRANSFORMER_PARAMS_COUNT = 3

global_id_counter = count(1)

def init_individual(individual):
    """Initialize an individual with a unique ID and model path."""
    individual.id = f"ind_{next(global_id_counter):08x}"
    individual.evaluated = False
    individual.model_path = os.path.join("models", f"{individual.id}.zip")
    return individual

def random_chromosome(n_features=TOTAL_FEATURE_COUNT):
    """
    Generate a random chromosome with architectural constraints.
    Chromosome structure:
      - Genes 0-3: learning_rate, gamma, ent_coef, num_blocks.
      - Genes 4 to (4 + MAX_BLOCKS*5 - 1): Modular block parameters (5 genes per block).
      - Next 3 genes: timesteps, reward_multiplier, dropout.
      - Next 3 genes: comp_num_layers, comp_nhead, comp_dropout.
      - Final 3 genes: out_num_layers, out_nhead, out_hidden.
    Total length = 4 + (MAX_BLOCKS * 5) + 3 + COMP_TRANSFORMER_PARAMS_COUNT + OUT_TRANSFORMER_PARAMS_COUNT.
    """
    current_dim = n_features
    chrom = [
        random.uniform(*PARAM_CONSTRAINTS['learning_rate']),
        random.uniform(*PARAM_CONSTRAINTS['gamma']),
        random.uniform(*PARAM_CONSTRAINTS['ent_coef']),
        random.randint(1, MAX_BLOCKS)
    ]
    for i in range(MAX_BLOCKS):
        if i < chrom[3]:
            block_type = random.randint(0, len(BLOCK_TYPES) - 2)  # Exclude NONE.
            layer_config = get_layer_config(block_type, current_dim)
            chrom.extend(layer_config)
            current_dim = layer_config[1]
        else:
            chrom.extend([6, current_dim, 2, 1, 1])
    chrom.extend([
        random.randint(*PARAM_CONSTRAINTS['timesteps']),
        random.uniform(*PARAM_CONSTRAINTS['reward_multiplier']),
        random.uniform(*PARAM_CONSTRAINTS['dropout'])
    ])
    chrom.extend([
        random.randint(*COMP_TRANSFORMER_CONSTRAINTS['num_layers']),
        random.randint(*COMP_TRANSFORMER_CONSTRAINTS['nhead']),
        random.uniform(*COMP_TRANSFORMER_CONSTRAINTS['dropout'])
    ])
    chrom.extend([
        random.randint(*OUT_TRANSFORMER_CONSTRAINTS['num_layers']),
        random.randint(*OUT_TRANSFORMER_CONSTRAINTS['nhead']),
        random.randint(*OUT_TRANSFORMER_CONSTRAINTS['hidden'])
    ])
    return chrom

def get_layer_config(block_type, current_dim):
    """Generate valid layer configuration for a specific block type."""
    block_name = BLOCK_TYPES[block_type]
    if block_name in ["CNN", "TCN"]:
        return [
            block_type,
            current_dim,  # Keep same dimension for CNN/TCN
            random.choice([3, 5, 7]),
            1,
            random.randint(1, ARCH_CONSTRAINTS['max_layers'])
        ]
    if block_name == "TRANSFORMER":
        hidden_size = random.choice([current_dim, current_dim * 2])
        valid_heads = [h for h in range(1, ARCH_CONSTRAINTS['max_heads'] + 1) if hidden_size % h == 0]
        heads = random.choice(valid_heads) if valid_heads else 1
        return [
            block_type,
            hidden_size,
            0,
            heads,
            random.randint(1, ARCH_CONSTRAINTS['max_layers'])
        ]
    if block_name in ["GRU", "LSTM"]:
        return [
            block_type,
            random.choice([current_dim, current_dim // 2, current_dim * 2]),
            0,
            1,
            random.randint(1, ARCH_CONSTRAINTS['max_layers'])
        ]
    if block_name == "MLP":
        return [
            block_type,
            random.choice([max(current_dim // 2, ARCH_CONSTRAINTS['min_hidden_size']), current_dim * 2]),
            0,
            1,
            random.randint(1, ARCH_CONSTRAINTS['max_layers'])
        ]
    return [6, current_dim, 0, 0, 0]

def modular_shape_is_valid(chromosome, window_size, input_dim):
    """Perform comprehensive architectural validation with dimension tracking."""
    num_blocks = int(chromosome[3])
    current_dim = input_dim
    for i in range(num_blocks):
        idx = 4 + i * 5
        block_type = BLOCK_TYPES.get(int(chromosome[idx]))
        hidden_size = int(chromosome[idx + 1])
        kernel_size = int(chromosome[idx + 2])
        num_heads = int(chromosome[idx + 3])
        if block_type == "TRANSFORMER" and hidden_size % num_heads != 0:
            return False
        if block_type in ["CNN", "TCN"]:
            if kernel_size > window_size or hidden_size != current_dim:
                return False
        elif block_type in ["GRU", "LSTM", "MLP"]:
            if not (ARCH_CONSTRAINTS['min_hidden_size'] <= hidden_size <= ARCH_CONSTRAINTS['max_hidden_size']):
                return False
        if block_type == "NONE" and hidden_size != current_dim:
            return False
        if block_type != "NONE":
            current_dim = hidden_size
    return True

def validate_chromosome(chromosome, input_dim, window_size=16):
    """
    Clamp each gene in the chromosome and enforce a consistent chain of dimensions.
    Uses the expected per-timestep feature count from the config.
    """
    logger.debug(f"Expected feature count: {input_dim}")
    expected_features = TOTAL_FEATURE_COUNT
    if input_dim != expected_features:
        raise ValueError(f"Feature mismatch: Env has {input_dim}, expected {expected_features}")
    required_len = 4 + MAX_BLOCKS * 5 + 3 + COMP_TRANSFORMER_PARAMS_COUNT + OUT_TRANSFORMER_PARAMS_COUNT
    if len(chromosome) != required_len:
        raise ValueError("Invalid chromosome length")
    # Clamp top-level hyperparameters.
    chromosome[0] = max(PARAM_CONSTRAINTS['learning_rate'][0],
                        min(PARAM_CONSTRAINTS['learning_rate'][1], chromosome[0]))
    chromosome[1] = max(PARAM_CONSTRAINTS['gamma'][0],
                        min(PARAM_CONSTRAINTS['gamma'][1], chromosome[1]))
    chromosome[2] = max(PARAM_CONSTRAINTS['ent_coef'][0],
                        min(PARAM_CONSTRAINTS['ent_coef'][1], chromosome[2]))
    num_blocks = int(chromosome[3])
    chromosome[3] = max(1, min(MAX_BLOCKS, num_blocks))
    current_dim = input_dim
    block_start = 4
    for i in range(chromosome[3]):
        idx = block_start + i * 5
        chromosome[idx] = int(max(0, min(len(BLOCK_TYPES) - 1, chromosome[idx])))
        block_type = BLOCK_TYPES[chromosome[idx]]
        # For NONE blocks, force out_dim to equal current_dim.
        best = current_dim if block_type == "NONE" else current_dim
        chromosome[idx + 1] = best
        chromosome[idx + 2] = int(max(2, min(window_size, chromosome[idx + 2])))
        heads = int(chromosome[idx + 3])
        heads = max(1, min(ARCH_CONSTRAINTS['max_heads'], heads))
        chromosome[idx + 3] = heads
        layers = int(chromosome[idx + 4])
        layers = max(1, min(ARCH_CONSTRAINTS['max_layers'], layers))
        chromosome[idx + 4] = layers
        if block_type == "TRANSFORMER" and (best % heads != 0):
            chromosome[idx + 3] = 1
        if block_type != "NONE":
            current_dim = best
    for i in range(chromosome[3], MAX_BLOCKS):
        idx = block_start + i * 5
        chromosome[idx:idx+5] = [6, current_dim, 2, 1, 1]
    t = 4 + MAX_BLOCKS * 5
    chromosome[t]   = int(max(PARAM_CONSTRAINTS['timesteps'][0],
                               min(PARAM_CONSTRAINTS['timesteps'][1], chromosome[t])))
    chromosome[t+1] = max(PARAM_CONSTRAINTS['reward_multiplier'][0],
                          min(PARAM_CONSTRAINTS['reward_multiplier'][1], chromosome[t+1]))
    chromosome[t+2] = max(PARAM_CONSTRAINTS['dropout'][0],
                          min(PARAM_CONSTRAINTS['dropout'][1], chromosome[t+2]))
    chromosome[t+3] = random.randint(*COMP_TRANSFORMER_CONSTRAINTS['num_layers']) if chromosome[t+3] is None else int(chromosome[t+3])
    chromosome[t+4] = random.randint(*COMP_TRANSFORMER_CONSTRAINTS['nhead']) if chromosome[t+4] is None else int(chromosome[t+4])
    chromosome[t+5] = max(COMP_TRANSFORMER_CONSTRAINTS['dropout'][0],
                          min(COMP_TRANSFORMER_CONSTRAINTS['dropout'][1], chromosome[t+5]))
    chromosome[t+6] = random.randint(*OUT_TRANSFORMER_CONSTRAINTS['num_layers']) if chromosome[t+6] is None else int(chromosome[t+6])
    chromosome[t+7] = random.randint(*OUT_TRANSFORMER_CONSTRAINTS['nhead']) if chromosome[t+7] is None else int(chromosome[t+7])
    chromosome[t+8] = random.randint(*OUT_TRANSFORMER_CONSTRAINTS['hidden']) if chromosome[t+8] is None else int(chromosome[t+8])
    return list(chromosome)

def find_valid_divisor(number):
    """Find largest valid divisor for transformer heads."""
    for i in range(min(8, number), 0, -1):
        if number % i == 0:
            return i
    return 1

def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

def custom_mutate_modular(individual, indpb=0.2, window_size=16, n_features=TOTAL_FEATURE_COUNT):
    """Constrained mutation operator with repair."""
    top_level = [0, 1, 2, 3]
    block_start = 4
    block_end = block_start + MAX_BLOCKS * 5
    trailing = [block_end, block_end + 1, block_end + 2]
    comp_indices = [block_end + 3, block_end + 4, block_end + 5]
    out_indices = [block_end + 6, block_end + 7, block_end + 8]
    for i in range(len(individual)):
        if random.random() < indpb:
            old_val = individual[i]
            if i in top_level:
                if i == 0:
                    individual[i] = max(PARAM_CONSTRAINTS['learning_rate'][0],
                                          min(PARAM_CONSTRAINTS['learning_rate'][1], individual[i] * random.uniform(0.5, 2)))
                elif i == 1:
                    individual[i] = max(PARAM_CONSTRAINTS['gamma'][0],
                                          min(PARAM_CONSTRAINTS['gamma'][1], individual[i] + random.uniform(-0.05, 0.05)))
                elif i == 2:
                    individual[i] = max(PARAM_CONSTRAINTS['ent_coef'][0],
                                          min(PARAM_CONSTRAINTS['ent_coef'][1], individual[i] * random.uniform(0.5, 2)))
                elif i == 3:
                    individual[i] = max(1, min(MAX_BLOCKS, individual[i] + random.choice([-1, 1])))
            elif block_start <= i < block_end:
                pos = (i - block_start) % 5
                if pos == 0:
                    individual[i] = random.randint(0, len(BLOCK_TYPES) - 1)
                elif pos == 1:
                    individual[i] = clamp(individual[i] * random.uniform(0.5, 2),
                                            ARCH_CONSTRAINTS['min_hidden_size'],
                                            ARCH_CONSTRAINTS['max_hidden_size'])
                elif pos == 2:
                    individual[i] = max(2, min(window_size, individual[i] + random.choice([-1, 1])))
                elif pos == 3:
                    individual[i] = max(1, min(ARCH_CONSTRAINTS['max_heads'], individual[i] + random.choice([-1, 1])))
                elif pos == 4:
                    individual[i] = max(1, min(ARCH_CONSTRAINTS['max_layers'], individual[i] + random.choice([-1, 1])))
            elif i in trailing:
                if i == trailing[0]:
                    individual[i] = max(PARAM_CONSTRAINTS['timesteps'][0],
                                          min(PARAM_CONSTRAINTS['timesteps'][1], individual[i] + random.choice([-10000, 10000])))
                elif i == trailing[1]:
                    individual[i] = max(PARAM_CONSTRAINTS['reward_multiplier'][0],
                                          min(PARAM_CONSTRAINTS['reward_multiplier'][1], individual[i] * random.uniform(0.8, 1.2)))
                elif i == trailing[2]:
                    individual[i] = max(PARAM_CONSTRAINTS['dropout'][0],
                                          min(PARAM_CONSTRAINTS['dropout'][1], individual[i] + random.uniform(-0.05, 0.05)))
            elif i in comp_indices:
                if i == comp_indices[0]:
                    individual[i] = random.randint(*COMP_TRANSFORMER_CONSTRAINTS['num_layers'])
                elif i == comp_indices[1]:
                    individual[i] = random.randint(*COMP_TRANSFORMER_CONSTRAINTS['nhead'])
                elif i == comp_indices[2]:
                    individual[i] = max(COMP_TRANSFORMER_CONSTRAINTS['dropout'][0],
                                          min(COMP_TRANSFORMER_CONSTRAINTS['dropout'][1], individual[i] + random.uniform(-0.05, 0.05)))
            elif i in out_indices:
                if i == out_indices[0]:
                    individual[i] = random.randint(*OUT_TRANSFORMER_CONSTRAINTS['num_layers'])
                elif i == out_indices[1]:
                    individual[i] = random.randint(*OUT_TRANSFORMER_CONSTRAINTS['nhead'])
                elif i == out_indices[2]:
                    individual[i] = random.randint(*OUT_TRANSFORMER_CONSTRAINTS['hidden'])
            if not modular_shape_is_valid(individual, window_size, n_features):
                individual[i] = old_val
    return individual,

def setup_toolbox(n_features=TOTAL_FEATURE_COUNT):
    """Configure the DEAP toolbox with constraints and operators."""
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
    except Exception:
        pass
    try:
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    except Exception:
        pass

    from deap import base, tools

    def init_gen_ind(icls):
        chrom = random_chromosome(n_features=n_features)
        ind = icls(chrom)
        return init_individual(ind)

    toolbox = base.Toolbox()
    toolbox.register("individual", init_gen_ind, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate_modular, indpb=0.2, window_size=16, n_features=n_features)
    toolbox.register("select", tools.selNSGA2)
    return toolbox
