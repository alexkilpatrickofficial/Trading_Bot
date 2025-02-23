# model_saver.py
import os
import tempfile
import logging
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

def save_model_atomically(model: PPO, target_path: str) -> None:
    """
    Saves the model atomically by writing to a temporary file first.
    """
    try:
        logger.info(f"Starting atomic save for model at {target_path}")
        dir_name = os.path.dirname(target_path)
        os.makedirs(dir_name, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".tmp.zip") as tmp_file:
            temp_path = tmp_file.name

        logger.info(f"Saving model to temporary file: {temp_path}")
        model.save(temp_path)

        logger.info(f"Replacing temporary file with final file: {target_path}")
        os.replace(temp_path, target_path)
        logger.info(f"Model saved atomically at {target_path}")
    except Exception as e:
        logger.error(f"Failed to atomically save model: {e}", exc_info=True)
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.debug(f"Removed temporary file {temp_path} after failure.")
