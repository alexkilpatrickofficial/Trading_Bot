#!/usr/bin/env python3
import os
import glob
import pickle
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import LOG_DIR, GA_STATE_FILE, CONTINUOUS_FEATURES
from envs.multi_stock_env import MultiStockTradingEnv

# Set up logging.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_full_dataset():
    """
    Loads the full preprocessed dataset from a pickle file.
    Adjust the path as necessary.
    """
    data_path = os.path.join(LOG_DIR, "preprocessed_data.pkl")
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_pickle(data_path)
        logger.info(f"Loaded full dataset from {data_path} with shape {df.shape}")
        return {"BTCUSD": df}
    else:
        raise FileNotFoundError(f"Preprocessed data file not found at {data_path}")

def calculate_sharpe_ratio(rewards, risk_free_rate=0.0):
    rewards = np.array(rewards)
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    if std_reward == 0:
        return 0.0
    return (mean_reward - risk_free_rate) / std_reward

def evaluate_model(model_path, env_kwargs, min_timesteps=80000):
    """
    Loads a PPO model from model_path using DummyVecEnv and evaluates it on the full dataset.
    Returns a dict with keys: 'total_reward', 'sharpe', 'volatility', and 'profit_curve'
    if the model has at least min_timesteps; otherwise returns None.
    """
    try:
        # Create a temporary environment for loading.
        temp_env = DummyVecEnv([lambda: MultiStockTradingEnv(**env_kwargs)])
        model = PPO.load(model_path, env=temp_env, reset_num_timesteps=False)
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

    if model.num_timesteps < min_timesteps:
        logger.info(f"Model {model_path} has only {model.num_timesteps} timesteps (< {min_timesteps}); skipping evaluation.")
        return None

    logger.info(f"Evaluating model {model_path} with {model.num_timesteps} timesteps.")
    # Create a fresh evaluation environment.
    env = DummyVecEnv([lambda: MultiStockTradingEnv(**env_kwargs)])
    obs = env.reset()
    rewards = []
    profit_curve = []  # Record adjusted balance at each step.
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards.append(reward[0])
        # Record adjusted balance if available.
        if info and isinstance(info, list) and "adjusted_balance" in info[0]:
            profit_curve.append(info[0]["adjusted_balance"])
        else:
            profit_curve.append(0)
        if done[0]:
            break

    total_reward = sum(rewards)
    sharpe = calculate_sharpe_ratio(rewards)
    volatility = np.std(rewards)
    return {
        "total_reward": total_reward,
        "sharpe": sharpe,
        "volatility": volatility,
        "profit_curve": profit_curve
    }

def plot_combined_profit_curve(evaluated_models, save_path="all_models_profit_curve.png"):
    """
    Plots a single chart that overlays the profit curves of all evaluated models.
    """
    plt.figure(figsize=(12, 6))
    for model_id, metrics in evaluated_models.items():
        profit_curve = metrics["profit_curve"]
        plt.plot(profit_curve, label=model_id)
    plt.xlabel("Time Steps")
    plt.ylabel("Adjusted Balance")
    plt.title("Combined Profit Curves for Evaluated Models")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved combined profit curve chart to {save_path}")
    return save_path

def compute_gene_ranges(chromosomes):
    """
    Computes the min and max for each gene across the GA population.
    Returns a list of tuples: [(min_0, max_0), (min_1, max_1), ...].
    """
    if not chromosomes:
        logger.error("No chromosomes available for range computation.")
        return []
    num_genes = len(chromosomes[0])
    mins = [float('inf')] * num_genes
    maxs = [float('-inf')] * num_genes
    for chrom in chromosomes:
        for i, gene in enumerate(chrom):
            try:
                val = float(gene)
            except Exception:
                continue
            if val < mins[i]:
                mins[i] = val
            if val > maxs[i]:
                maxs[i] = val
    ranges = [(mins[i], maxs[i]) for i in range(num_genes)]
    return ranges

def export_ranges_to_csv(ranges, csv_path="hyperparameter_ranges.csv"):
    """
    Exports the per-gene min/max ranges to a CSV file.
    """
    headers = [f"Gene_{i}_min" for i in range(len(ranges))] + [f"Gene_{i}_max" for i in range(len(ranges))]
    row = []
    for r in ranges:
        row.append(r[0])
    for r in ranges:
        row.append(r[1])
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(row)
    logger.info(f"Exported hyperparameter ranges to {csv_path}")

def get_population_chromosomes(ga_state_path):
    """
    Loads the GA state from file and returns the population (list of individuals).
    Each individual should have an attribute 'id' and be a list of genes.
    """
    if os.path.exists(ga_state_path):
        with open(ga_state_path, 'rb') as f:
            state = pickle.load(f)
        population = state.get('population', [])
        return population
    else:
        logger.warning(f"No GA state available at {ga_state_path}.")
        return []

def export_hyperparameters_to_csv(population, evaluated_ids, csv_path="evaluated_models_hyperparameters.csv"):
    """
    Exports the hyperparameters (chromosome values) for individuals whose IDs are in evaluated_ids.
    Assumes each individual is a list of 18 values and has an attribute 'id'.
    """
    headers = [
        "model_id", "learning_rate", "gamma", "ent_coef",
        "cnn_num_layers", "cnn_filters", "cnn_kernel_size", "cnn_stride",
        "lstm_hidden_size", "lstm_num_layers", "fc_hidden_size",
        "total_timesteps", "eval_freq", "num_heads",
        "reward_multiplier", "dropout_prob", "use_attention_flag",
        "post_lstm_hidden_size", "post_lstm_num_layers"
    ]
    rows = []
    for individual in population:
        if hasattr(individual, "id") and individual.id in evaluated_ids:
            row = [individual.id] + individual
            rows.append(row)
    if rows:
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)
        logger.info(f"Exported hyperparameters for {len(rows)} evaluated models to {csv_path}")
    else:
        logger.warning("No evaluated models found in the GA state; hyperparameter export skipped.")

def main():
    # Load the full dataset.
    stock_data = load_full_dataset()

    # Set up environment parameters for evaluation.
    env_kwargs = {
        "stock_data": stock_data,
        # The chromosome is only used for reward_multiplier in evaluation.
        "chromosome": [0] * 18,
        "window_size": 16,
        "initial_balance": 10000.0,
        "scaler_path": os.path.join(LOG_DIR, "scaler.pkl"),
        "fee_rate": 0.001,
        "slippage": 0.001,
        "spread": 0.001,
        "max_fill_fraction": 0.8,
    }

    # Export hyperparameter ranges from the GA state.
    ga_state_path = os.path.join(LOG_DIR, GA_STATE_FILE)
    population = get_population_chromosomes(ga_state_path)
    if population:
        ranges = compute_gene_ranges(population)
        export_ranges_to_csv(ranges, csv_path="hyperparameter_ranges.csv")
    else:
        logger.warning("No GA state available; skipping hyperparameter ranges export.")

    # Search for model files in the "models" directory.
    model_pattern = os.path.join("models", "ind_*.zip")
    model_paths = glob.glob(model_pattern)
    if not model_paths:
        logger.error("No models found in the 'models' directory.")
        return

    evaluated_models = {}
    evaluated_ids = []

    logger.info(f"Found {len(model_paths)} model files. Evaluating only models with >= 80k timesteps...")
    for model_path in model_paths:
        model_id = os.path.splitext(os.path.basename(model_path))[0]
        logger.info(f"Evaluating model {model_id}...")
        result = evaluate_model(model_path, env_kwargs, min_timesteps=80000)
        if result is not None:
            evaluated_models[model_id] = result
            evaluated_ids.append(model_id)
            logger.info(f"Model {model_id}: Total Reward = {result['total_reward']:.2f}, Sharpe = {result['sharpe']:.2f}, Volatility = {result['volatility']:.2f}")
        else:
            logger.info(f"Model {model_id} did not meet the minimum timesteps requirement or failed evaluation.")

    if not evaluated_models:
        logger.error("No models were evaluated successfully. Exiting.")
        return

    # Create a single chart that overlays the profit curves of all evaluated models.
    plot_combined_profit_curve(evaluated_models, save_path="all_models_profit_curve.png")

    # Export hyperparameters of evaluated models.
    if population:
        export_hyperparameters_to_csv(population, evaluated_ids, csv_path="evaluated_models_hyperparameters.csv")
    else:
        logger.warning("No GA state available; skipping hyperparameter export.")

if __name__ == '__main__':
    main()
