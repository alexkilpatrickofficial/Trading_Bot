import time
import os
import numpy as np
import logging
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from torch.utils.tensorboard import SummaryWriter
from config import TENSORBOARD_LOG_DIR
from model_saver import save_model_atomically  # Ensure this function is defined

logger = logging.getLogger(__name__)

class TensorBoardCallback(BaseCallback):
    """
    Custom callback for logging to TensorBoard.
    """
    def __init__(self, model_name: str, run_id: str = None, log_dir: str = TENSORBOARD_LOG_DIR):
        super().__init__(verbose=1)
        self.model_name = model_name
        self.run_id = run_id or f"run_{int(time.time())}"
        self.log_dir = os.path.join(log_dir, self.model_name, self.run_id)
        self.writer = None
        self.performance_log = []

    def _on_training_start(self) -> None:
        logger.info("TensorBoardCallback: Training started.")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def _on_step(self) -> bool:
        step = self.num_timesteps
        infos = self.locals.get("infos", [{}])
        if infos:
            info = infos[0]
            reward = info.get("reward", 0.0)
            total_profit = info.get("total_profit", 0.0)
            realized_profit = info.get("realized_profit", 0.0)
            balance = info.get("balance", 0.0)
            if self.writer:
                self.writer.add_scalar("Metrics/Reward", reward, step)
                self.writer.add_scalar("Metrics/Realized_Profit", realized_profit, step)
                self.writer.add_scalar("Metrics/Total_Profit", total_profit, step)
                self.writer.add_scalar("Metrics/Balance", balance, step)
            self.performance_log.append([step, balance, total_profit, realized_profit, reward])
        return True

    def _on_training_end(self) -> None:
        logger.info("TensorBoardCallback: Training ended.")
        if self.writer:
            self.writer.flush()
            self.writer.close()
        if self.performance_log:
            df = pd.DataFrame(
                self.performance_log, 
                columns=["Step", "Balance", "Total_Profit", "Realized_Profit", "Reward"]
            )
            csv_path = os.path.join(self.log_dir, "performance_log.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Performance log saved at: {csv_path}")

class CheckpointCallback(BaseCallback):
    """
    Custom callback that saves the model at regular intervals during training.
    """
    def __init__(self, save_freq: int, model_path: str, verbose: int = 1):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.model_path = model_path  # e.g. 'models/ind_12345abcd.zip'
        self.last_step_save = 0

    def _init_callback(self) -> None:
        if self.model is None:
            raise ValueError("No model found. Callback must be used with an existing model.")
        dir_name = os.path.dirname(self.model_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def _on_step(self) -> bool:
        if (self.n_calls - self.last_step_save) >= self.save_freq:
            self.last_step_save = self.n_calls
            step_str = f"_step{self.num_timesteps}"
            partial_model_path = self.model_path.replace(".zip", f"{step_str}.zip")
            save_model_atomically(self.model, partial_model_path)
            if self.verbose > 0:
                logger.info(f"[CheckpointCallback] Model checkpoint saved at {partial_model_path}")
        return True

    # No _on_training_end override needed here

class EarlyStoppingCallback(BaseCallback):
    """
    Early-stops training if consecutive episodes terminate 'badly'.
    A 'bad' episode is one that ends due to high repeated action counts or abnormal conditions.
    Terminations due solely to end-of-data are considered normal and do not count.
    
    This callback updates its counter only once per episode boundary, and ignores episodes that are too short.
    """
    def __init__(
        self,
        patience: int = 20,
        repeat_threshold: int = 16,
        min_episode_length: int = 10,  # Only count episodes longer than this as "meaningful"
        verbose: int = 0
    ):
        """
        :param patience: Number of consecutive bad episodes before stopping.
        :param repeat_threshold: If repeated_action_count >= this value, count as a bad episode.
        :param min_episode_length: Minimum episode length to be considered for early stopping.
        :param verbose: Logging verbosity level.
        """
        super().__init__(verbose)
        self.patience = patience
        self.repeat_threshold = repeat_threshold
        self.min_episode_length = min_episode_length
        self.counter = 0

    def _init_callback(self) -> None:
        if self.model is None:
            raise ValueError("No model found. Callback must be used with an existing model.")
        logger.info(
            f"EarlyStoppingCallback initialized: patience={self.patience}, "
            f"repeat_threshold={self.repeat_threshold}, min_episode_length={self.min_episode_length}"
        )

    def _on_step(self) -> bool:
        # Retrieve done flags and episode info from the vectorized environment.
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", [{}])
        if dones is None:
            return True

        if any(dones):
            # Assume a single episode termination for now.
            info = infos[0]
            episode_length = info.get("episode_length", None)
            # If episode_length is not provided, assume it's long enough.
            if episode_length is None or episode_length >= self.min_episode_length:
                repeated_count = info.get("repeated_action_count", 0)
                is_abnormal = info.get("abnormal", False)
                # Only count if termination is not due to end-of-data.
                if not info.get("end_of_data", False) and (repeated_count >= self.repeat_threshold or is_abnormal):
                    self.counter += 1
                    if self.verbose > 0:
                        logger.info(
                            f"EarlyStopping: Episode terminated badly (length={episode_length}, repeated_action_count={repeated_count}, abnormal={is_abnormal}). Counter: {self.counter}/{self.patience}"
                        )
                else:
                    self.counter = 0
            else:
                self.counter = 0

        if self.counter >= self.patience:
            if self.verbose > 0:
                logger.info("EarlyStopping: Patience exceeded, stopping training.")
            return False  # Stops training.
        return True
