# logging_utils.py

import logging
import os
from config import LOG_DIR

def setup_logging(level=logging.INFO, log_filename="trading_bot.log"):
    """
    Sets up logging configuration.

    Args:
        level (int): Logging level.
        log_filename (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_filename)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
