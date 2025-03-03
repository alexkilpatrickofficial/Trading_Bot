import pandas as pd
import numpy as np
import logging
from ta import trend, momentum, volatility
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def calculate_rsi(series, window):
    """
    Calculates the Relative Strength Index (RSI).
    """
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_required_indicators(data):
    """
    Calculates only the required technical indicators using a 16-period window (unless noted otherwise):
      - RSI and EMA for momentum/trend.
      - MACD and MACD signal (fast=8, slow=16, signal=4).
      - SMA50 and SMA200 for longer-term trend.
      - VWAP (if 'volumeusd' and 'volume' are available).
      - ATR (16-period) for volatility.
      - Bollinger %B indicator (16-period, 2 standard deviations).
      - Price relative to SMA50.
    """
    logger.info("Calculating required indicators...")

    try:
        # RSI and EMA (16-period)
        data['rsi'] = momentum.RSIIndicator(close=data['close'], window=16).rsi()
        data['ema'] = trend.EMAIndicator(close=data['close'], window=16).ema_indicator()

        # MACD and MACD signal (using fast=8, slow=16, signal=4)
        macd = trend.MACD(close=data['close'], window_slow=16, window_fast=8, window_sign=4)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()

        # SMA50 and SMA200
        data['sma50'] = data['close'].rolling(window=50, min_periods=1).mean()
        data['sma200'] = data['close'].rolling(window=200, min_periods=1).mean()

        # VWAP (requires 'volumeusd' and 'volume')
        if 'volumeusd' in data.columns and 'volume' in data.columns:
            data['vwap'] = (data['volumeusd'] / data['volume']).fillna(0)
        else:
            logger.error("Columns 'volumeusd' and 'volume' are required for VWAP calculation.")
            raise ValueError("Missing columns for VWAP calculation.")

        # ATR using a 16-period window
        atr_indicator = volatility.AverageTrueRange(
            high=data['high'], low=data['low'], close=data['close'], window=16
        )
        data['atr'] = atr_indicator.average_true_range()

        # Bollinger %B indicator (16-period, 2 standard deviations)
        bollinger = volatility.BollingerBands(close=data['close'], window=16, window_dev=2)
        data['bollinger_pctb'] = bollinger.bollinger_pband()

        # Price relative to SMA50
        data['price_sma50'] = data['close'] / data['sma50']

        logger.info("Required indicators calculated successfully.")
        return data

    except Exception as e:
        logger.error(f"Error in calculate_required_indicators: {e}", exc_info=True)
        raise

def apply_global_scaling(data, cols_to_scale=None):
    """
    Applies a single-stage scaling approach to specified columns.
    Logs how many NaNs remain.
    """
    if cols_to_scale is None:
        # Example: scale these daily-based features
        cols_to_scale = ['close', 'high', 'low', 'open', 'vwap']

    logger.info(f"Applying single-stage scaling to columns: {cols_to_scale}")

    # Log-transform raw price columns to reduce skew
    for c in cols_to_scale:
        if (data[c] <= 0).any():
            logger.warning(f"Column '{c}' has zero or negative values. Clipping to 1e-9 for log transform.")
            data[c] = data[c].clip(lower=1e-9)
        data[f"{c}_log"] = np.log1p(data[c])

    log_cols = [f"{c}_log" for c in cols_to_scale]
    scaler = StandardScaler()
    scaler.fit(data[log_cols])

    # Check for NaNs before and after scaling
    for col in log_cols:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column {col} has {nan_count} NaN values before scaling")
    data[log_cols] = scaler.transform(data[log_cols])
    for col in log_cols:
        nan_count_after = data[col].isna().sum()
        if nan_count_after > 0:
            logger.warning(f"Column {col} has {nan_count_after} NaNs after scaling")

    logger.info("Single-stage scaling complete. Created/log-transformed columns:")
    logger.info(f"{log_cols}")
    return data, scaler

def verify_features(data, feature_list):
    """
    Verifies that all required features are present in the DataFrame and contain no NaN values.
    """
    missing_features = [feature for feature in feature_list if feature not in data.columns]
    if missing_features:
        logger.error(f"Missing features in data: {missing_features}")
        raise KeyError(f"Missing features in data: {missing_features}")

    for feature in feature_list:
        if data[feature].isnull().any():
            logger.error(f"Feature '{feature}' contains NaN values.")
            raise ValueError(f"Feature '{feature}' contains NaN values.")

    logger.info("All required features are present and valid.")