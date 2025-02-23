import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import pickle
import os
from collections import deque
import itertools
import time
import torch

from config import (
    FEATURE_COLUMNS,
    CONTINUOUS_FEATURES,  # Ensure the first element is the target price (e.g. "close")
    BINARY_FEATURES,
    EXTRA_FEATURES_PER_TIMESTEP  # e.g., 6
)
from config import ACCOUNT_METRICS  # For dynamic computation.
from evaluation import calculate_sharpe_ratio  # Ensure this function exists

logger = logging.getLogger(__name__)

class MultiStockTradingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": None}

    def __init__(
        self,
        stock_data: dict,
        chromosome: list,
        window_size: int = 16,
        initial_balance: float = 10000.0,
        render_mode=None,
        scaler_path: str = "scaler.pkl",
        fee_rate: float = 0.001,
        slippage: float = 0.001,
        spread: float = 0.001,
        max_fill_fraction: float = 0.8,
        mavg_weight3: float = 0.001,
        mavg_weight6: float = 0.0005,
        mavg_weight12: float = 0.0001,
        include_past_actions: bool = True,  # Optionally include past actions
        use_aux_reward: bool = True,        # Enable separate auxiliary reward
        main_reward_weight: float = 1.0,     # Weight for main reward
        aux_reward_weight: float = 0.5       # Weight for auxiliary reward
    ):
        super().__init__()
        # Core parameters
        self.stock_data = stock_data
        self.chromosome = chromosome
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.render_mode = render_mode
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.spread = spread
        self.max_fill_fraction = max_fill_fraction

        self.minimum_trade_amount = 5.0
        self.max_position = 2000.0
        self.risk_threshold = 0.0

        self.last_action_type = None
        self.repeated_action_count = 0
        self.episode_length = 0
        self.max_episode_length = 8766

        self.hold_counter = 0
        self.inactivity_steps = 0

        self.total_profit = 0.0
        self.realized_profit = 0.0
        self.position = 0.0
        self.avg_buy_price = 0.0

        self.reward_multiplier = self.chromosome[13] if len(self.chromosome) > 13 else 1.0

        # For auxiliary reward: store last price for prediction comparison.
        self.last_price = None

        # Reward weighting parameters.
        self.use_aux_reward = use_aux_reward
        self.main_reward_weight = main_reward_weight
        self.aux_reward_weight = aux_reward_weight

        # Load scaler if available.
        self.scaler = None
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler loaded from {scaler_path}")
            except Exception as e:
                logger.error(f"Failed to load scaler: {e}")

        # Compute feature count per timestep.
        base_feature_count = len(CONTINUOUS_FEATURES) + len(BINARY_FEATURES) + len(ACCOUNT_METRICS)
        total_feature_count = base_feature_count + EXTRA_FEATURES_PER_TIMESTEP

        # Extra action features if enabled.
        extra_action_features = 32 if include_past_actions else 0

        # Define observation space (flattened vector).
        obs_dim = self.window_size * total_feature_count + extra_action_features
        self.observation_space = spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([3, 10])

        self.current_step = 0
        self.np_random = np.random.RandomState()

        # Moving average weights.
        self.mavg_weight3 = mavg_weight3
        self.mavg_weight6 = mavg_weight6
        self.mavg_weight12 = mavg_weight12

        # Initialize action history.
        self.include_past_actions = include_past_actions
        if self.include_past_actions:
            self.action_history = deque([(0, 0)] * 16, maxlen=16)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        self.balance = self.initial_balance
        self.total_profit = 0.0
        self.realized_profit = 0.0
        self.position = 0.0
        self.avg_buy_price = 0.0
        self.episode_length = 0
        self.hold_counter = 0
        self.inactivity_steps = 0

        if self.include_past_actions:
            self.action_history = deque([(0, 0)] * 16, maxlen=16)

        self.prices = self.stock_data["BTCUSD"]["close"].values
        self.sell_profit_history = deque(maxlen=12)
        max_start = len(self.prices) - self.window_size - 1
        if max_start <= self.window_size:
            raise ValueError("Not enough data to create a valid episode.")

        self.current_step = self.np_random.randint(self.window_size, max_start)
        if "close" not in self.stock_data["BTCUSD"].columns:
            raise KeyError("The 'close' column is missing from the dataset.")

        self.prev_price = self.prices[self.current_step - 1]
        self.previous_adjusted_balance = self.initial_balance
        self.last_price = self.prev_price

        logger.debug(f"Reset: Starting at step {self.current_step} with prev_price {self.prev_price}")

        obs = self._get_observation()
        info = self._construct_info(reward=0.0)
        return obs, info

    def _construct_info(self, reward=0.0, terminate=False):
        current_price = self._get_price(self.current_step)
        unrealized_profit = (current_price - self.avg_buy_price) * self.position if self.position > 0 else 0.0
        return {
            "total_profit": self.total_profit,
            "realized_profit": self.realized_profit,
            "balance": self.balance,
            "unrealized_profit": unrealized_profit,
            "adjusted_balance": self.balance + unrealized_profit,
            "hold_counter": self.hold_counter,
            "inactivity_steps": self.inactivity_steps,
            "reward": reward,
            "repeated_action_count": self.repeated_action_count,
            "terminate": terminate
        }

    def _get_observation(self):
        start_idx = self.current_step - self.window_size
        data_slice = self.stock_data["BTCUSD"].iloc[start_idx:self.current_step]
        
        # Ensure continuous_data is 2D even if only one column is selected.
        continuous_data = np.array(data_slice[CONTINUOUS_FEATURES].values, dtype=np.float32)
        if continuous_data.ndim == 1:
            continuous_data = continuous_data.reshape(-1, 1)
        
        # Ensure binary_data is 2D.
        binary_data = np.array(data_slice[BINARY_FEATURES].values, dtype=np.float32)
        if binary_data.ndim == 1:
            binary_data = binary_data.reshape(-1, 1)
        
        account_metrics = np.array(
            [[self.balance, self.position, self.total_profit]] * self.window_size,
            dtype=np.float32
        )

        base_features = np.concatenate([continuous_data, binary_data, account_metrics], axis=1)
        extra_features = np.zeros((self.window_size, EXTRA_FEATURES_PER_TIMESTEP), dtype=np.float32)
        features = np.concatenate([base_features, extra_features], axis=1)
        logger.debug(f"Observation matrix shape (before flatten): {features.shape}")
        observation = features.flatten()

        if self.include_past_actions:
            past_actions = np.array(list(self.action_history), dtype=np.float32).flatten()
            logger.debug(f"Past actions shape: {past_actions.shape}")
            observation = np.concatenate([observation, past_actions])

        observation = np.nan_to_num(observation, nan=0.0, posinf=1e9, neginf=-1e9)
        assert observation.shape == self.observation_space.shape, (
            f"Observation shape mismatch: expected {self.observation_space.shape}, got {observation.shape}"
        )
        return observation

    def _get_price(self, step_index):
        return self.stock_data["BTCUSD"]["close"].values[step_index]

    def step(self, action):
        try:
            # --------------------- Debug & Shape Handling ---------------------
            logger.debug(f"[step] Received raw action: {action} (type: {type(action)})")
            if isinstance(action, torch.Tensor):
                logger.debug(f"[step] Action is a torch.Tensor with shape {action.shape}. Converting to NumPy.")
                action = action.detach().cpu().numpy()

            # If the action is still a np.ndarray, check its shape
            if isinstance(action, np.ndarray):
                logger.debug(f"[step] Action is now a NumPy array with shape {action.shape}.")
                # If shape is (1,2), flatten it to (2,)
                if action.shape == (1, 2):
                    logger.debug("[step] Flattening action from (1,2) to (2,).")
                    action = action[0]  # shape now (2,)

                # If shape is just (), interpret it as a scalar
                elif action.shape == ():
                    logger.debug("[step] Action shape is (), interpreting as a single scalar action.")
                    action = [int(action)]

            # Convert to list or tuple if not already
            if not hasattr(action, '__iter__'):
                action = [action]  # single scalar case
            elif isinstance(action, np.ndarray):
                action = action.tolist()

            logger.debug(f"[step] Action after shape fix: {action} (length: {len(action)})")

            # Final assertion: we need exactly 2 elements
            if len(action) != 2:
                raise ValueError(f"[step] Expected action of length 2, got length {len(action)}. "
                                f"Action content: {action}")

            # Unpack two elements
            action_type, trade_size_raw = action

            # --------------- Step Logic ---------------
            if self.current_step >= len(self.prices) - 1:
                # End-of-data: treat as done.
                return self._terminate_episode(reward=0.0, error="end_of_data")

            # Convert discrete trade_size into fraction.
            trade_fraction = (trade_size_raw + 1) / 10.0
            current_price = self._get_price(self.current_step)
            prev_price = self._get_price(self.current_step - 1) if self.current_step > 0 else current_price

            if np.isnan(current_price) or np.isnan(prev_price):
                raise ValueError("NaN detected in price data.")

            # Track repeated actions
            if self.last_action_type is None or action_type != self.last_action_type:
                self.last_action_type = action_type
                self.repeated_action_count = 1
            else:
                self.repeated_action_count += 1

            # Calculate base reward.
            step_reward = 0.0
            realized_profit_this_step = 0.0

            if action_type == 1:  # Buy
                step_reward = self._buy(current_price, trade_fraction)
            elif action_type == 2:  # Sell
                realized_profit_this_step = self._sell(current_price, trade_fraction)
                step_reward = realized_profit_this_step
            elif action_type == 0:  # Hold
                step_reward = self._hold(prev_price, current_price)
                self.hold_counter += 1
                if self.hold_counter >= 10:
                    self.inactivity_steps += 1
                    self.hold_counter = 0
            else:
                logger.warning(f"Unknown action type: {action_type}")

            if action_type != 0:
                self.hold_counter = 0

            if self.include_past_actions:
                self.action_history.append((action_type, trade_size_raw))

            # Update profit and step counters
            self.total_profit += realized_profit_this_step
            self.realized_profit += realized_profit_this_step
            self.current_step += 1
            self.episode_length += 1

            # Compute incremental reward based on adjusted balance change
            current_price = self._get_price(self.current_step)
            unrealized_profit = (current_price - self.avg_buy_price) * self.position if self.position > 0 else 0.0
            adjusted_balance = self.balance + unrealized_profit
            incremental_reward = (adjusted_balance - self.previous_adjusted_balance) / self.initial_balance
            self.previous_adjusted_balance = adjusted_balance

            # Compute auxiliary reward components
            main_reward, aux_reward = self._calculate_reward(realized_profit_this_step, action_type)
            combined_reward = self.main_reward_weight * main_reward
            if self.use_aux_reward:
                combined_reward += self.aux_reward_weight * aux_reward

            final_reward = step_reward + incremental_reward + combined_reward

            # Determine if episode is done
            done = self._check_termination(adjusted_balance)
            obs = self._get_observation()
            info = self._construct_info(reward=final_reward, terminate=done)
            info["main_reward"] = main_reward
            info["aux_reward"] = aux_reward

            if done:
                return self._terminate_episode(reward=final_reward)
            return obs, final_reward, False, False, info

        except Exception as e:
            logger.error(f"Error in step: {e}", exc_info=True)
            return self._terminate_episode(reward=0.0, error=str(e))

    def _apply_slippage(self, price, is_buy=True):
        adjustment = np.random.normal(0, self.slippage)
        return price * (1 + adjustment if is_buy else 1 - adjustment)

    def _buy(self, current_price, trade_fraction):
        ask_price = current_price * (1 + self.spread / 2)
        effective_price = self._apply_slippage(ask_price, is_buy=True)
        invest_amount = self.balance * trade_fraction

        if invest_amount < self.minimum_trade_amount:
            logger.debug("Investment amount below minimum trade amount.")
            return -0.1

        units_to_buy = (invest_amount / effective_price) * self.max_fill_fraction
        cost = units_to_buy * effective_price
        fee = cost * self.fee_rate
        total_cost = cost + fee

        if total_cost > self.balance:
            logger.debug("Insufficient balance for trade after fee.")
            return -0.1

        if self.position + units_to_buy > self.max_position:
            units_to_buy = self.max_position - self.position
            if units_to_buy <= 0:
                logger.debug("Max position reached; cannot buy more.")
                return -0.2
            cost = units_to_buy * effective_price
            fee = cost * self.fee_rate
            total_cost = cost + fee

        total_cost_basis = self.avg_buy_price * self.position + cost + fee
        new_position = self.position + units_to_buy
        self.avg_buy_price = total_cost_basis / new_position if new_position > 0 else effective_price
        self.position = new_position
        self.balance -= total_cost
        self.balance = max(0.0, round(self.balance, 2))

        logger.debug(f"Bought {units_to_buy:.4f} units at effective price {effective_price:.2f}; "
                     f"Fee: {fee:.2f}, New balance: {self.balance:.2f}")
        return -fee * 0.5

    def _sell(self, current_price, trade_fraction):
        if self.position <= 0:
            logger.debug("No position to sell.")
            return -0.1

        bid_price = current_price * (1 - self.spread / 2)
        effective_price = self._apply_slippage(bid_price, is_buy=False)
        units_to_sell = self.position * trade_fraction * self.max_fill_fraction

        if units_to_sell < 1e-6:
            logger.debug("Trade fraction too small; no units sold.")
            return -0.05

        revenue = units_to_sell * effective_price
        fee = revenue * self.fee_rate
        net_revenue = revenue - fee

        realized_profit = (effective_price - self.avg_buy_price) * units_to_sell - fee
        self.balance += net_revenue
        self.balance = round(self.balance, 2)
        self.position -= units_to_sell

        if self.position <= 1e-6:
            self.position = 0.0
            self.avg_buy_price = 0.0

        logger.debug(f"Sold {units_to_sell:.4f} units at effective price {effective_price:.2f}; "
                     f"Fee: {fee:.2f}, Realized profit: {realized_profit:.2f}, "
                     f"New balance: {self.balance:.2f}")
        self.sell_profit_history.append(realized_profit)
        return realized_profit

    def _hold(self, prev_price, current_price):
        price_diff = current_price - prev_price
        if self.position > 0:
            return price_diff * self.position * 0.001
        else:
            return -0.01

    def _calculate_reward(self, realized_profit, action):
        profit_reward = realized_profit / self.initial_balance
        sell_history = (np.array(self.sell_profit_history) / self.initial_balance
                        if len(self.sell_profit_history) > 0
                        else np.array([0]))
        mavg3 = np.mean(sell_history[-3:]) if len(sell_history) >= 3 else 0.0
        mavg6 = np.mean(sell_history[-6:]) if len(sell_history) >= 6 else 0.0
        mavg_reward = (self.mavg_weight3 * mavg3 + self.mavg_weight6 * mavg6)

        inactivity_penalty = -0.005 * self.inactivity_steps
        repeated_penalty = -0.01 * max(0, self.repeated_action_count - 5)
        if self.position > 0:
            current_price = self._get_price(self.current_step)
            unrealized_loss = max(0, self.avg_buy_price - current_price)
            unrealized_penalty = - (unrealized_loss * self.position) / self.initial_balance
        else:
            unrealized_penalty = 0.0

        main_reward = (profit_reward + mavg_reward + inactivity_penalty + repeated_penalty + unrealized_penalty)

        current_price = self._get_price(self.current_step)
        predicted_price = self.last_price if self.last_price is not None else current_price
        price_error = abs(current_price - predicted_price) / current_price
        aux_reward = max(0, 1 - price_error)

        self.last_price = current_price
        return main_reward, aux_reward

    def _check_termination(self, adjusted_balance):
        if self.balance <= self.risk_threshold:
            logger.warning(f"Terminating: Balance {self.balance} below risk threshold.")
            return True
        if self.inactivity_steps >= 5:
            logger.warning("Terminating: Inactivity steps exceeded threshold.")
            return True
        if self.repeated_action_count >= 12:
            logger.warning("Terminating: Repeated action count exceeded threshold.")
            return True
        if self.episode_length >= self.max_episode_length:
            logger.info("Terminating: Maximum episode length reached.")
            return True
        return False

    def _terminate_episode(self, reward, error=None):
        done = True
        current_price = self._get_price(self.current_step)
        final_unrealized = (current_price - self.avg_buy_price) * self.position if self.position > 0 else 0.0
        final_adjusted_balance = self.balance + final_unrealized

        abnormal = False
        if error is not None and error != "end_of_data":
            abnormal = True
        if abnormal:
            logger.warning("Abnormal termination detected. Applying heavy penalty.")
            reward -= 25.0

        final_info = {
            "total_profit": self.total_profit,
            "realized_profit": self.realized_profit,
            "balance": self.balance,
            "unrealized_profit": 0.0,
            "adjusted_balance": final_adjusted_balance,
            "hold_counter": self.hold_counter,
            "inactivity_steps": self.inactivity_steps,
            "reward": reward,
            "abnormal": abnormal,
            "reason": "Error encountered." if error and error != "end_of_data" else "Normal termination."
        }
        if error and error != "end_of_data":
            final_info["error"] = error
            logger.error(f"Terminating episode due to error: {error}")

        self._terminate_episode_info = final_info
        obs = self._get_observation()
        logger.debug(f"Terminating episode. Final info: {final_info}")
        # Return five values: obs, reward, terminated, truncated, final_info.
        return obs, reward, True, False, final_info

    def render(self):
        if self.render_mode == "human":
            current_price = self._get_price(self.current_step)
            unrealized = (current_price - self.avg_buy_price) * self.position if self.position > 0 else 0.0
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
                  f"Position: {self.position:.4f}, AvgBuyPrice: {self.avg_buy_price:.2f}, "
                  f"Unrealized: {unrealized:.2f}, TotalProfit: {self.total_profit:.2f}")
