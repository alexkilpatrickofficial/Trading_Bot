import os

# Base directories and output filenames
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Original continuous features (8 features)
ORIGINAL_CONTINUOUS_FEATURES = [
    "rsi",           # Momentum
    "macd",          # Trend
    "macd_signal",   # Trend signal
    "sma50",         # Short-term trend
    "sma200",        # Long-term trend
    "vwap",          # Volume-weighted average price
    "atr",           # Volatility
    "ema"            # Short-term trend
]

# Additional continuous features (8 features)
ADDITIONAL_CONTINUOUS_FEATURES = [
    "bollinger_pctb",   # Derived Bollinger %B indicator
    "price_sma50",      # Ratio of price to SMA50 (trend strength)
    "close_log",        # Log-transformed close for scaling stability
    "day_sin", "day_cos",       # Cyclical day-of-week features
    "month_sin", "month_cos",   # Cyclical month-of-year features
    "ut_trailing_stop"  # UT Bot trailing stop (volatility-based dynamic stop)
]

# Combine continuous features (16 total)
CONTINUOUS_FEATURES = ORIGINAL_CONTINUOUS_FEATURES + ADDITIONAL_CONTINUOUS_FEATURES

# Binary features (2 features)
BINARY_FEATURES = [
    "ut_buy_signal",    # UT Bot buy signal flag
    "ut_sell_signal"    # UT Bot sell signal flag
]

# All feature columns (16 continuous + 2 binary = 18 features)
FEATURE_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES

# Account metrics (3 features)
ACCOUNT_METRICS = ["balance", "position", "total_profit"]

# Base feature count per timestep: 18 (feature columns) + 3 (account metrics) = 21
BASE_FEATURE_COUNT = len(FEATURE_COLUMNS) + len(ACCOUNT_METRICS)

# Extra features per timestep (distributed extra features, e.g., for past actions)
EXTRA_FEATURES_PER_TIMESTEP = 6

# Total features per timestep must be 27 (21 + 6)
TOTAL_FEATURE_COUNT = BASE_FEATURE_COUNT + EXTRA_FEATURES_PER_TIMESTEP

# Output and cache filenames
OUTPUT_H5_FILE = "processed_data.h5"
CACHE_FILE_NAME = "preprocessed_data.pkl"
SPLIT_DATA_H5 = "train_test_data.h5"
SCALER_FILE_NAME = "scaler.pkl"
GA_STATE_FILE = "ga_state.pkl"
PERFORMANCE_LOG_FILE = "performance_log.csv"

# TensorBoard logging directory
TENSORBOARD_LOG_DIR = os.path.join(LOG_DIR, "tensorboard_logs")
