import os
import random
from deap import base, creator, tools
from itertools import count
from config import FEATURE_COLUMNS, CONTINUOUS_FEATURES, BINARY_FEATURES, TOTAL_FEATURE_COUNT, EXTRA_FEATURES_PER_TIMESTEP
import logging
import glob
import pickle
import numpy as np
from skopt import gp_minimize
from skopt.space import Real

logger = logging.getLogger(__name__)

def set_seed(seed_value=42):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import torch
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    except ImportError:
        pass

def get_tensorboard_gen_log_dir(current_gen, log_dir):
    """
    Return a log directory path for the current generation's TensorBoard logs.
    """
    return os.path.join(log_dir, f"generation_{current_gen}")

def calculate_sharpe_ratio(rewards, risk_free_rate=0.0):
    arr = np.array(rewards)
    mean_r = arr.mean()
    std_r = arr.std()
    if std_r < 1e-9:
        return 0.0
    return (mean_r - risk_free_rate) / std_r

def save_ga_state(population, generation, filepath):
    """
    Save the current GA state (population and generation) to a pickle file.
    """
    try:
        lock_path = f"{filepath}.lock"
        with open(lock_path, 'w') as _:
            pass
        with open(filepath, 'wb') as f:
            pickle.dump({'population': population, 'generation': generation}, f)
        logger.info(f"GA state saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save GA state to {filepath}: {e}")

def load_latest_ga_state(base_filepath, lock_timeout=10):
    """
    Load the latest GA state from checkpoint files.
    """
    checkpoint_dir = os.path.dirname(base_filepath) or os.getcwd()
    base_name, ext = os.path.splitext(os.path.basename(base_filepath))
    pattern = f"{base_name}_gen_*{ext}"
    checkpoint_files = sorted(
        glob.glob(os.path.join(checkpoint_dir, pattern)),
        key=lambda x: int(x.split('_gen_')[1].split(ext)[0])
    )
    if not checkpoint_files:
        logger.info("No GA state found. Starting fresh.")
        return None, 1
    latest = checkpoint_files[-1]
    lock_path = f"{latest}.lock"
    try:
        with open(latest, 'rb') as f:
            state = pickle.load(f)
        population = state['population']
        generation = state['generation']
        logger.info(f"Loaded GA state from {latest}: Generation {generation}, Population size {len(population)}")
        return population, generation
    except Exception as e:
        logger.error(f"Error loading GA state from {latest}: {e}")
        return None, 1

def bayesian_optimize_parameters(history):
    """
    Use Bayesian optimization (via gp_minimize) to propose a new mutation probability (indpb)
    based on historical evaluations. `history` is a list of tuples: (mutation_probability, average_fitness).
    We define the objective as the negative of the average fitness (to maximize fitness).
    """
    def objective(x):
        mp = x[0]
        best_fit = None
        best_diff = float('inf')
        for mp_val, fitness in history:
            diff = abs(mp_val - mp)
            if diff < best_diff:
                best_fit = fitness
                best_diff = diff
        return -best_fit if best_fit is not None else 0.0

    space = [Real(0.05, 0.5, name='indpb')]
    res = gp_minimize(objective, space, n_calls=10, random_state=42)
    optimized_indpb = res.x[0]
    logger.info(f"Bayesian optimization: optimized mutation probability (indpb) = {optimized_indpb:.4f}")
    return optimized_indpb

def init_deap_creators():
    """
    Initialize DEAP creator classes for multi-objective optimization.
    This should be called once in your main script.
    Note: FitnessMulti weights: (1.0, 1.0, -1.0)
    Chromosome length should be: 4 + (MAX_BLOCKS * 5) + 3 + COMP_TRANSFORMER_PARAMS_COUNT + OUT_TRANSFORMER_PARAMS_COUNT.
    """
    from deap import creator, base
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)
