import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir='logs', log_file='trading_env.log', max_bytes=10*1024*1024, backup_count=5, level=None):
    """
    Configures logging for the project.

    Args:
        log_dir (str): Directory where log files will be stored.
        log_file (str): Name of the log file.
        max_bytes (int): Maximum size in bytes of the log file before rotating.
        backup_count (int): Number of backup log files to keep.
        level (str): Logging level as a string (e.g. 'DEBUG', 'INFO'); defaults to environment variable LOG_LEVEL or 'DEBUG'.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine log level
    if level is None:
        level = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
    log_level = getattr(logging, level, logging.DEBUG)

    logger = logging.getLogger()
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # Rotating file handler
    file_handler = RotatingFileHandler(os.path.join(log_dir, log_file), maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )

    # Optional: Set specific log levels for certain modules if needed.
    # For example:
    # logging.getLogger('some_module').setLevel(logging.INFO)

