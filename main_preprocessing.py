import os
import logging
import pandas as pd
from data_preprocessing import load_crypto_data_optimized, save_to_hdf5, train_and_save_scaler
from config import LOG_DIR, CACHE_FILE_NAME, OUTPUT_H5_FILE, FEATURE_COLUMNS, CONTINUOUS_FEATURES

import tkinter as tk
from tkinter import filedialog, messagebox

SAVED_LOCATION_FILE = os.path.join(LOG_DIR, "dataset_location.txt")

def setup_logging():
    """
    Configures logging for the application.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "preprocessing.log"))
        ]
    )

def load_saved_dataset_location():
    """
    Loads a previously saved dataset location from file if available.
    """
    if os.path.exists(SAVED_LOCATION_FILE):
        with open(SAVED_LOCATION_FILE, "r") as f:
            path = f.read().strip()
            if os.path.isfile(path):
                return path
    return None

def save_dataset_location(path):
    """
    Saves the dataset location to a file.
    """
    with open(SAVED_LOCATION_FILE, "w") as f:
        f.write(path)

def choose_dataset_location():
    """
    Opens a file dialog to choose the dataset CSV file.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window.
    file_path = filedialog.askopenfilename(
        title="Select Dataset CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def main_preprocessing():
    """
    Main function to process raw data and prepare it for modeling.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting preprocessing...")

    # Try to load a previously saved dataset location.
    INPUT_CSV_PATH = load_saved_dataset_location()
    if INPUT_CSV_PATH:
        logger.info(f"Found saved dataset location: {INPUT_CSV_PATH}")
        # Ask the user if they want to use the saved location or pick a new file.
        root = tk.Tk()
        root.withdraw()
        use_saved = messagebox.askyesno("Dataset Location", 
                                        f"Use saved dataset location?\n{INPUT_CSV_PATH}")
        root.destroy()
        if not use_saved:
            INPUT_CSV_PATH = None

    # If no valid saved path, prompt the user.
    if not INPUT_CSV_PATH:
        INPUT_CSV_PATH = choose_dataset_location()
        if not INPUT_CSV_PATH:
            logger.error("No dataset selected. Exiting preprocessing.")
            return
        # Ask if the user wants to save this location for future use.
        root = tk.Tk()
        root.withdraw()
        save_choice = messagebox.askyesno("Save Location",
                                          "Would you like to save this dataset location for future runs?")
        root.destroy()
        if save_choice:
            save_dataset_location(INPUT_CSV_PATH)
            logger.info(f"Dataset location saved: {INPUT_CSV_PATH}")

    CACHE_FILE = os.path.join(LOG_DIR, CACHE_FILE_NAME)
    OUTPUT_H5_PATH = os.path.join(LOG_DIR, OUTPUT_H5_FILE)
    SCALER_PATH = os.path.join(LOG_DIR, "scaler.pkl")

    # Step 1: Load raw data
    try:
        raw_data = pd.read_csv(INPUT_CSV_PATH)
        logger.info("Raw data loaded successfully.")
        logger.debug(f"Columns in raw data: {list(raw_data.columns)}")
    except FileNotFoundError:
        logger.error(f"File not found: {INPUT_CSV_PATH}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading raw data: {e}")
        return

    # Step 2: Preprocess data
    try:
        preprocessed_data = load_crypto_data_optimized(raw_data, cache_file=CACHE_FILE)
        logger.info("Data preprocessing completed successfully.")
        logger.debug(f"Columns in preprocessed data: {list(preprocessed_data.columns)}")
        
        # Verify required features are present
        missing_features = [feature for feature in FEATURE_COLUMNS if feature not in preprocessed_data.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise KeyError(f"Missing features in data: {missing_features}")
        logger.info("All required features are present in the processed data.")
        
        # Keep only the features to be used
        preprocessed_data = preprocessed_data[FEATURE_COLUMNS]
        logger.debug(f"Data shape after subsetting to FEATURE_COLUMNS: {preprocessed_data.shape}")
    except KeyError as e:
        logger.error(f"Feature verification failed: {e}")
        return
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    # Step 3: Save preprocessed data to HDF5
    try:
        save_to_hdf5(preprocessed_data, OUTPUT_H5_PATH)
        logger.info(f"Preprocessed data saved to {OUTPUT_H5_PATH}")
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")
        return

    # Step 4: Train and save scaler
    try:
        train_and_save_scaler({"BTCUSD": preprocessed_data}, SCALER_PATH)
        logger.info(f"Scaler trained and saved to {SCALER_PATH}")
    except Exception as e:
        logger.error(f"Error training or saving scaler: {e}")
        return

    logger.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    main_preprocessing()
