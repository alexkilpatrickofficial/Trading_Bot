# logging_config.py

import logging
import os

def setup_logging(log_dir='logs', log_file='trading_env.log'):
    """
    Configures logging for the project.

    Args:
        log_dir (str): Directory where log files will be stored.
        log_file (str): Name of the log file.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )

    # Optional: Configure specific loggers if needed
    # For example, to set a different level for a specific module:
    # logging.getLogger('some_module').setLevel(logging.INFO)
