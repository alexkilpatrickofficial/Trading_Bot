import os
import logging
import pandas as pd
from data_preprocessing import load_crypto_data_optimized, save_to_hdf5, train_and_save_scaler
from config import LOG_DIR, CACHE_FILE_NAME, OUTPUT_H5_FILE, FEATURE_COLUMNS, CONTINUOUS_FEATURES

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

def verify_processed_data(data, required_features):
    """
    Verifies that all required features are present in the DataFrame.
    """
    missing_features = [feature for feature in required_features if feature not in data.columns]
    if missing_features:
        logging.error(f"Missing features: {missing_features}")
        raise KeyError(f"Missing features in data: {missing_features}")
    logging.info("All required features are present in the processed data.")

def main_preprocessing():
    """
    Main function to process raw data and prepare it for modeling.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting preprocessing...")

    INPUT_CSV_PATH = r"C:\Users\Alex\Downloads\Gemini_BTCUSD_1h.csv"
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
        verify_processed_data(preprocessed_data, FEATURE_COLUMNS)
        
        # Drop extra columns: keep only the features you plan to use.
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
