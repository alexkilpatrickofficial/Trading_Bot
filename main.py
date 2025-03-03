# main.py
import os
import time
import logging
import numpy as np
import pandas as pd
import subprocess
import webbrowser
from deap import tools
from logging_config import setup_logging
from data_preprocessing import load_crypto_data_optimized, train_and_save_scaler, save_to_hdf5

# GA and chromosome imports
from ga_utils import (
    init_deap_creators,
    get_tensorboard_gen_log_dir,
    calculate_sharpe_ratio,
    save_ga_state,
    load_latest_ga_state
)
from chromosome import setup_toolbox, init_individual
from evaluation import evaluate
from models import ModularBlock, DynamicHybridFeatureExtractor

from config import LOG_DIR, CACHE_FILE_NAME, OUTPUT_H5_FILE
from callbacks import TensorBoardCallback, CheckpointCallback, EarlyStoppingCallback

logger = logging.getLogger(__name__)

def launch_tensorboard(log_dir, port=6006):
    """
    Launch TensorBoard in a subprocess.
    """
    try:
        tb_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", str(port)])
        time.sleep(5)
        webbrowser.open(f"http://localhost:{port}")
        logger.info(f"TensorBoard launched at http://localhost:{port}")
        return tb_process
    except Exception as e:
        logger.error(f"Failed to launch TensorBoard: {e}")
        return None

def reinitialize_individual(toolbox, stock_data, log_dir, max_attempts=3):
    """
    Generate a new individual until its evaluation does not yield a heavy penalty.
    """
    HEAVY_PENALTY = (-1e6, -1e6, 1e6)
    for attempt in range(max_attempts):
        new_ind = toolbox.individual()
        fitness = toolbox.evaluate(new_ind, stock_data=stock_data, LOG_DIR=log_dir)
        if fitness[0] != HEAVY_PENALTY[0]:
            new_ind.fitness.values = fitness
            logger.info(f"Reinitialization attempt {attempt+1} succeeded for new individual {new_ind.id}.")
            return new_ind
        else:
            logger.info(f"Reinitialization attempt {attempt+1} produced heavy penalty for new individual {new_ind.id}.")
    new_ind = toolbox.individual()
    new_ind.fitness.values = toolbox.evaluate(new_ind, stock_data=stock_data, LOG_DIR=log_dir)
    logger.info(f"Max reinitialization attempts reached. New individual {new_ind.id} selected with heavy penalty.")
    return new_ind

def main():
    setup_logging()
    logger.info("Starting GA optimization process...")

    # 1) Initialize DEAP creator classes
    init_deap_creators()

    # 2) Data loading and preprocessing
    raw_data_path = os.path.expanduser("~/Downloads/Gemini_BTCUSD_1h.csv")
    if not os.path.exists(raw_data_path):
        logger.error(f"Data file not found: {raw_data_path}")
        return

    try:
        raw_data = pd.read_csv(raw_data_path)
        logger.info("Raw data loaded successfully.")
        logger.debug(f"Columns in raw data: {list(raw_data.columns)}")
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        return

    try:
        preprocessed_data = load_crypto_data_optimized(raw_data, cache_file=os.path.join(LOG_DIR, CACHE_FILE_NAME))
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        return

    try:
        save_to_hdf5(preprocessed_data, os.path.join(LOG_DIR, OUTPUT_H5_FILE))
        logger.info("Preprocessed data saved to HDF5 successfully.")
    except Exception as e:
        logger.error(f"Error saving preprocessed data to HDF5: {e}")

    scaler_path = os.path.join(LOG_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        try:
            train_and_save_scaler({"BTCUSD": preprocessed_data}, scaler_path)
            logger.info(f"Scaler trained and saved at {scaler_path}")
        except Exception as e:
            logger.error(f"Error training or saving scaler: {e}")
            return

    stock_data = {"BTCUSD": preprocessed_data}

    # 3) GA Setup
    toolbox = setup_toolbox()  # n_features is set by TOTAL_FEATURE_COUNT by default
    toolbox.register("evaluate", evaluate)

    ga_checkpoint_dir = os.path.join(LOG_DIR, "ga_checkpoints")
    os.makedirs(ga_checkpoint_dir, exist_ok=True)

    saved_population, start_gen = load_latest_ga_state(ga_checkpoint_dir)
    if saved_population:
        population = saved_population
        current_gen = start_gen + 1
        logger.info(f"Resuming GA from generation {current_gen}")
    else:
        population = toolbox.population(n=15)
        for ind in population:
            init_individual(ind)
        current_gen = 1
        logger.info(f"Initialized new population with {len(population)} individuals.")

    n_generations = 12
    tb_process = launch_tensorboard(os.path.join(LOG_DIR, "tensorboard_logs"), port=6006)
    HEAVY_PENALTY = (-1e6, -1e6, 1e6)

    # 4) GA Evolution Loop
    for gen in range(current_gen, n_generations + 1):
        logger.info(f"=== Generation {gen} ===")

        # Evaluate individuals with invalid fitness.
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid_ind:
            fitness = toolbox.evaluate(ind, stock_data=stock_data, LOG_DIR=LOG_DIR)
            if fitness[0] == HEAVY_PENALTY[0]:
                logger.info(f"Individual {ind.id} received heavy penalty. Reinitializing (up to 3 attempts).")
                new_ind = reinitialize_individual(toolbox, stock_data, LOG_DIR)
                index = population.index(ind)
                population[index] = new_ind
            else:
                ind.fitness.values = fitness
            ind.evaluated = True

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        # Apply mutation.
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        population[:] = offspring

        # Re-evaluate individuals with invalid fitness.
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid_ind:
            fitness = toolbox.evaluate(ind, stock_data=stock_data, LOG_DIR=LOG_DIR)
            if fitness[0] == HEAVY_PENALTY[0]:
                logger.info(f"Individual {ind.id} received heavy penalty on re-evaluation. Replacing with new individual.")
                new_ind = reinitialize_individual(toolbox, stock_data, LOG_DIR)
                index = population.index(ind)
                population[index] = new_ind
            else:
                ind.fitness.values = fitness

        try:
            fits = [ind.fitness.values[0] for ind in population]
        except Exception as e:
            logger.error(f"Error computing fitness values: {e}")
            return

        mean_fit = np.mean(fits)
        max_fit = np.max(fits)
        logger.info(f"Generation {gen}: mean fitness = {mean_fit:.4f}, max fitness = {max_fit:.4f}")
        ga_state_path = os.path.join(ga_checkpoint_dir, f"ga_state_gen_{gen}.pkl")
        save_ga_state(population, gen, ga_state_path)

    best_ind = tools.selBest(population, k=1)[0]
    logger.info(f"Best Individual: {best_ind}, Fitness: {best_ind.fitness.values}")

    if tb_process:
        tb_process.terminate()
        logger.info("TensorBoard process terminated.")

if __name__ == "__main__":
    main()
