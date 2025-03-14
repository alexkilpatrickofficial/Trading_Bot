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

# Import cleanup functions from model_cleanup.py
from model_cleanup import scan_and_cleanup_models, cleanup_logs, scan_and_merge_tensorboard_logs

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

    # --- Duplicate Scan & Cleanup Before Training ---
    run_cleanup = input("Do you want to scan for duplicate model checkpoints and logs? [y/N]: ").strip().lower()
    if run_cleanup == 'y':
        bulk_cleanup = input("Run in bulk cleanup mode? (Automatically delete/merge duplicates) [y/N]: ").strip().lower() == 'y'
        # Clean up duplicate model checkpoints
        models_dir = os.path.join(os.getcwd(), "models")
        logger.info("Scanning for duplicate model checkpoints in: %s", models_dir)
        scan_and_cleanup_models(models_dir, bulk=bulk_cleanup)
        
        # Clean up rotated log files
        logs_dir = LOG_DIR  # Using the main logs folder from config
        logger.info("Scanning for duplicate rotated log files in: %s", logs_dir)
        cleanup_logs(logs_dir, "trading_env.log.*", keep=1)
        cleanup_logs(logs_dir, "events.out.tfevents.*", keep=1)
        
        # Merge duplicate TensorBoard log subdirectories
        # Assuming TensorBoard logs are stored under: LOG_DIR/tensorboard_logs/generation_X/
        tb_generation_dir = os.path.join(LOG_DIR, "tensorboard_logs", "generation_1")  # Adjust generation as needed
        logger.info("Merging duplicate TensorBoard log subdirectories in: %s", tb_generation_dir)
        scan_and_merge_tensorboard_logs(tb_generation_dir, bulk=bulk_cleanup)
    # --- End Duplicate Scan & Cleanup ---

    # 1) Initialize DEAP creator classes
    init_deap_creators()

    # 2) Data loading and preprocessing
    raw_data_path = os.path.expanduser("~/Downloads/Gemini_BTCUSD_1h.csv")
    if not os.path.exists(raw_data_path):
        logger.error("Data file not found: %s", raw_data_path)
        return

    try:
        raw_data = pd.read_csv(raw_data_path)
        logger.info("Raw data loaded successfully with shape: %s", raw_data.shape)
        logger.debug("Columns in raw data: %s", list(raw_data.columns))
    except Exception as e:
        logger.exception("Error loading raw data: %s", e)
        return

    try:
        preprocessed_data = load_crypto_data_optimized(raw_data, cache_file=os.path.join(LOG_DIR, CACHE_FILE_NAME))
        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.exception("Error during data preprocessing: %s", e)
        return

    try:
        save_to_hdf5(preprocessed_data, os.path.join(LOG_DIR, OUTPUT_H5_FILE))
        logger.info("Preprocessed data saved to HDF5 successfully.")
    except Exception as e:
        logger.exception("Error saving preprocessed data to HDF5: %s", e)

    scaler_path = os.path.join(LOG_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        try:
            train_and_save_scaler({"BTCUSD": preprocessed_data}, scaler_path)
            logger.info("Scaler trained and saved at %s", scaler_path)
        except Exception as e:
            logger.exception("Error training or saving scaler: %s", e)
            return

    stock_data = {"BTCUSD": preprocessed_data}

    # 3) GA Setup
    toolbox = setup_toolbox()
    toolbox.register("evaluate", evaluate)

    ga_checkpoint_dir = os.path.join(LOG_DIR, "ga_checkpoints")
    os.makedirs(ga_checkpoint_dir, exist_ok=True)
    logger.info("GA checkpoint directory: %s", ga_checkpoint_dir)

    saved_population, start_gen = load_latest_ga_state(ga_checkpoint_dir)
    if saved_population:
        population = saved_population
        current_gen = start_gen + 1
        logger.info("Resuming GA from generation %d", current_gen)
    else:
        population = toolbox.population(n=15)
        for ind in population:
            init_individual(ind)
        current_gen = 1
        logger.info("Initialized new population with %d individuals: %s", len(population), [ind.id for ind in population])

    n_generations = 12
    tb_log_dir = os.path.join(LOG_DIR, "tensorboard_logs")
    tb_process = launch_tensorboard(tb_log_dir, port=6006)
    HEAVY_PENALTY = (-1e6, -1e6, 1e6)

    # 4) GA Evolution Loop
    for gen in range(current_gen, n_generations + 1):
        logger.info("=== Generation %d ===", gen)
        generation_start = time.time()

        # Evaluate individuals with invalid fitness.
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid_ind:
            fitness = toolbox.evaluate(ind, stock_data=stock_data, LOG_DIR=LOG_DIR)
            if fitness[0] == HEAVY_PENALTY[0]:
                logger.info("Individual %s received heavy penalty. Reinitializing (up to 3 attempts).", ind.id)
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
                logger.info("Individual %s received heavy penalty on re-evaluation. Replacing with new individual.", ind.id)
                new_ind = reinitialize_individual(toolbox, stock_data, LOG_DIR)
                index = population.index(ind)
                population[index] = new_ind
            else:
                ind.fitness.values = fitness

        try:
            fits = [ind.fitness.values[0] for ind in population]
        except Exception as e:
            logger.exception("Error computing fitness values: %s", e)
            return

        mean_fit = np.mean(fits)
        max_fit = np.max(fits)
        logger.info("Generation %d: mean fitness = %.4f, max fitness = %.4f", gen, mean_fit, max_fit)
        generation_elapsed = time.time() - generation_start
        logger.info("Generation %d completed in %.2f seconds.", gen, generation_elapsed)

        ga_state_path = os.path.join(ga_checkpoint_dir, f"ga_state_gen_{gen}.pkl")
        save_ga_state(population, gen, ga_state_path)
        logger.info("GA state saved to %s", ga_state_path)

    best_ind = tools.selBest(population, k=1)[0]
    logger.info("Best Individual: %s, Fitness: %s", best_ind, best_ind.fitness.values)

    if tb_process:
        tb_process.terminate()
        logger.info("TensorBoard process terminated.")

if __name__ == "__main__":
    main()
