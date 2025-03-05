import logging
import os
from logging.handlers import RotatingFileHandler
from config import LOG_DIR

def setup_logging(level=None, log_filename="trading_bot.log", max_bytes=10*1024*1024, backup_count=5):
    """
    Sets up logging configuration.

    Args:
        level (int or str): Logging level (e.g., logging.INFO or 'INFO'). Defaults to environment variable LOG_LEVEL or INFO.
        log_filename (str): Name of the log file.
        max_bytes (int): Maximum bytes per log file before rotation.
        backup_count (int): Number of backup log files to keep.

    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_filename)

    # Determine logging level from parameter or environment variable
    if level is None:
        level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    if isinstance(level, str):
        level = getattr(logging, level, logging.INFO)

    # Prevent duplicate logging handlers
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Setup Rotating File Handler
    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)

    # Setup Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[file_handler, console_handler]
    )

    logger = logging.getLogger(__name__)
    return logger
