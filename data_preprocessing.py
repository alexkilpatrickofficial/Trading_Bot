import os
import logging
import pandas as pd
import pickle
import h5py
import numpy as np
from sklearn.preprocessing import RobustScaler  # Use RobustScaler by default

# We'll no longer use the full calculate_indicators from data_utils.
# from data_utils import calculate_indicators, verify_features
from config import FEATURE_COLUMNS, CONTINUOUS_FEATURES, LOG_DIR

logger = logging.getLogger(__name__)


def save_to_hdf5(df, h5_file_path, dataset_name='data', chunk_size=1000):
    """
    Saves the DataFrame to an HDF5 file, including column names.
    Only numeric columns are saved.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    num_rows, num_features = numeric_df.shape
    with h5py.File(h5_file_path, 'w') as h5f:
        dset = h5f.create_dataset(
            dataset_name,
            shape=(num_rows, num_features),
            dtype='float32',
            chunks=(chunk_size, num_features),
            compression="gzip"
        )
        for i in range(0, num_rows, chunk_size):
            end = i + chunk_size
            dset[i:end, :] = numeric_df.iloc[i:end].values.astype('float32')
        h5f.attrs['columns'] = ','.join(numeric_df.columns)
    logger.info(f"DataFrame successfully saved to HDF5: {h5_file_path}")


def calculate_required_indicators(data):
    """
    Calculates only the technical indicators required by our model:
      - RSI, EMA, MACD, MACD signal (using a 16-period window)
      - SMA50 and SMA200
      - VWAP (if available)
      - ATR (16-period)
      - Bollinger %B indicator (16-period)
      - Price relative to SMA50
    """
    try:
        import ta
    except ImportError:
        logger.error("Please install the 'ta' library for technical indicator calculations.")
        raise

    # RSI and EMA (16-period)
    data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=16).rsi()
    data['ema'] = ta.trend.EMAIndicator(close=data['close'], window=16).ema_indicator()

    # MACD and MACD signal (using fast=8, slow=16, signal=4)
    macd = ta.trend.MACD(close=data['close'], window_slow=16, window_fast=8, window_sign=4)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()

    # SMA50 and SMA200
    data['sma50'] = data['close'].rolling(window=50, min_periods=1).mean()
    data['sma200'] = data['close'].rolling(window=200, min_periods=1).mean()

    # VWAP: requires both 'volumeusd' and 'volume'
    if 'volumeusd' in data.columns and 'volume' in data.columns:
        data['vwap'] = (data['volumeusd'] / data['volume']).fillna(0)
    else:
        logger.error("Columns 'volumeusd' and 'volume' are required for VWAP calculation.")
        raise ValueError("Missing columns for VWAP calculation.")

    # ATR using a 16-period window
    atr_indicator = ta.volatility.AverageTrueRange(
        high=data['high'], low=data['low'], close=data['close'], window=16
    )
    data['atr'] = atr_indicator.average_true_range()

    # Bollinger %B indicator (16-period, 2 std dev)
    bollinger = ta.volatility.BollingerBands(close=data['close'], window=16, window_dev=2)
    data['bollinger_pctb'] = bollinger.bollinger_pband()

    # Price relative to SMA50
    data['price_sma50'] = data['close'] / data['sma50']

    return data


def calculate_ut_bot_candles(data, a=1, atr_period=10):
    """
    Calculates UT Bot–style trailing stop and generates candle-based buy/sell signals.
    
    New columns added:
      - 'ut_prev_close': previous bar's close.
      - 'ut_tr': True Range.
      - 'ut_atr': ATR over `atr_period`.
      - 'ut_nloss': a * ut_atr.
      - 'ut_trailing_stop': dynamically calculated trailing stop.
      - 'ut_buy_signal': boolean flag for a buy signal.
      - 'ut_sell_signal': boolean flag for a sell signal.
    """
    data = data.copy()
    data['ut_prev_close'] = data['close'].shift(1)
    data['ut_tr'] = data.apply(
        lambda row: max(
            row['high'] - row['low'],
            abs(row['high'] - row['ut_prev_close']) if pd.notnull(row['ut_prev_close']) else 0,
            abs(row['low'] - row['ut_prev_close']) if pd.notnull(row['ut_prev_close']) else 0
        ),
        axis=1
    )
    data['ut_atr'] = data['ut_tr'].rolling(window=atr_period, min_periods=1).mean()
    data['ut_nloss'] = a * data['ut_atr']

    ts = np.zeros(len(data))
    close_prices = data['close'].values
    nloss = data['ut_nloss'].values

    for i in range(len(data)):
        if i == 0:
            ts[i] = close_prices[i] - nloss[i]
        else:
            prev_ts = ts[i - 1]
            if (close_prices[i] > prev_ts) and (close_prices[i - 1] > prev_ts):
                ts[i] = max(prev_ts, close_prices[i] - nloss[i])
            elif (close_prices[i] < prev_ts) and (close_prices[i - 1] < prev_ts):
                ts[i] = min(prev_ts, close_prices[i] + nloss[i])
            else:
                ts[i] = close_prices[i] - nloss[i] if close_prices[i] > prev_ts else close_prices[i] + nloss[i]
    data['ut_trailing_stop'] = ts

    # Generate UT Bot signals:
    ut_buy_signals = [False] * len(data)
    ut_sell_signals = [False] * len(data)
    for i in range(1, len(data)):
        if (close_prices[i - 1] <= ts[i - 1]) and (close_prices[i] > ts[i]):
            ut_buy_signals[i] = True
        if (close_prices[i - 1] >= ts[i - 1]) and (close_prices[i] < ts[i]):
            ut_sell_signals[i] = True
    data['ut_buy_signal'] = ut_buy_signals
    data['ut_sell_signal'] = ut_sell_signals

    logger.info("UT Bot candle signals calculated successfully.")
    return data


def load_crypto_data_optimized(data, start_date='2020-01-01', end_date=None, cache_file=None, scaler_type="robust"):
    """
    Loads and preprocesses cryptocurrency data by:
      1) Normalizing column names.
      2) Renaming columns (volume, close, etc.).
      3) Parsing and filtering the 'date' column (default: from 2020 onward).
      4) Creating cyclical features (day_sin, day_cos, month_sin, month_cos).
      5) Creating a log-transformed 'close_log'.
      6) Calculating required technical indicators.
      7) Calculating UT Bot candle signals.
      8) Verifying that all required features exist.
      9) Replacing infinite values and filling NAs.
      10) Scaling the CONTINUOUS_FEATURES.
    """
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading preprocessed data from cache: {cache_file}")
        return pd.read_pickle(cache_file)

    # 1) Normalize column names.
    data.columns = data.columns.str.lower()

    # 2) Rename columns.
    data.rename(
        columns={
            "volume btc": "volume",
            "volume usd": "volumeusd",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close"
        },
        inplace=True
    )
    logger.debug(f"Columns after renaming: {list(data.columns)}")

    # 3) Date parsing and filtering.
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data.dropna(subset=['date'], inplace=True)
        data.sort_values(by='date', inplace=True)
        data.reset_index(drop=True, inplace=True)
        logger.debug(f"Shape after date parsing: {data.shape}")

        if start_date is not None:
            data = data[data['date'] >= pd.to_datetime(start_date)]
            data.reset_index(drop=True, inplace=True)
            logger.debug(f"Shape after filtering from start_date ({start_date}): {data.shape}")
        if end_date is not None:
            data = data[data['date'] <= pd.to_datetime(end_date)]
            data.reset_index(drop=True, inplace=True)
            logger.debug(f"Shape after filtering to end_date: {data.shape}")

        # 4) Create cyclical day and month features.
        day_of_week = data['date'].dt.dayofweek  # 0..6
        data.loc[:, 'day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        data.loc[:, 'day_cos'] = np.cos(2 * np.pi * day_of_week / 7)


        month_of_year = data['date'].dt.month - 1  # 0..11
        data['month_sin'] = np.sin(2 * np.pi * month_of_year / 12)
        data['month_cos'] = np.cos(2 * np.pi * month_of_year / 12)

    # 5) Create 'close_log'.
    if "close" not in data.columns:
        logger.error("Column 'close' missing, cannot create 'close_log'.")
    else:
        data["close_log"] = np.log1p(data["close"].clip(lower=1e-9))

    # 6) Calculate required technical indicators.
    data = calculate_required_indicators(data)

    # 7) Calculate UT Bot candle signals.
    data = calculate_ut_bot_candles(data, a=1, atr_period=10)

    # 8) Verify that all required features exist.
    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in data.columns]
    if missing_features:
        logger.error(f"Missing features in data: {missing_features}")
        raise KeyError(f"Missing features in data: {missing_features}")
    else:
        logger.info("All required features are present in the processed data.")

    # 9) Replace infinite values and fill NAs.
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # 10) Scale continuous features.
    if scaler_type.lower() == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    data[CONTINUOUS_FEATURES] = scaler.fit_transform(data[CONTINUOUS_FEATURES])
    logger.info(f"Data shape after preprocessing: {data.shape}")

    if cache_file:
        cache_dir = os.path.dirname(cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        data.to_pickle(cache_file)
        logger.info(f"Preprocessed data cached at {cache_file}")

    return data


def train_and_save_scaler(data_dict, scaler_path, scaler_type="robust"):
    """
    Trains a scaler on the provided data and saves it to disk.
    Data_dict: e.g. {"BTCUSD": df}
    """
    try:
        combined_data = pd.concat(data_dict.values(), ignore_index=True)
        for feature in CONTINUOUS_FEATURES:
            if feature not in combined_data.columns:
                raise ValueError(f"Feature '{feature}' not found in preprocessed data.")

        scaler = RobustScaler() if scaler_type.lower() == "robust" else None
        if scaler is None:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        scaler.fit(combined_data[CONTINUOUS_FEATURES])

        os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        logger.info(f"Scaler trained and saved at {scaler_path}")
    except Exception as e:
        logger.error(f"Failed to train and save scaler: {e}")
        raise


# Note: Since this project is modular, the following block is for local testing.
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Starting preprocessing...")

    INPUT_CSV_PATH = r"C:\Users\Alex\Downloads\Gemini_BTCUSD_1h.csv"
    CACHE_FILE = os.path.join(LOG_DIR, "preprocessed_data.pkl")
    OUTPUT_H5_PATH = os.path.join(LOG_DIR, "processed_data.h5")
    SCALER_PATH = os.path.join(LOG_DIR, "scaler.pkl")

    try:
        raw_data = pd.read_csv(INPUT_CSV_PATH)
        logger.info("Raw data loaded successfully.")
        logger.debug(f"Columns in raw data: {list(raw_data.columns)}")
    except Exception as e:
        logger.error(f"Error loading raw data: {e}")
        exit(1)

    try:
        # The start_date is now defaulted to 2020-01-01.
        preprocessed_data = load_crypto_data_optimized(raw_data)
        logger.info("Data preprocessing completed successfully.")
        logger.debug(f"Columns in preprocessed data: {list(preprocessed_data.columns)}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        exit(1)

    try:
        save_to_hdf5(preprocessed_data, OUTPUT_H5_PATH)
        logger.info(f"Preprocessed data saved to HDF5 at {OUTPUT_H5_PATH}")
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")

    try:
        train_and_save_scaler({"BTCUSD": preprocessed_data}, SCALER_PATH)
        logger.info(f"Scaler trained and saved to {SCALER_PATH}")
    except Exception as e:
        logger.error(f"Error training or saving scaler: {e}")

    logger.info("Preprocessing completed successfully!")