#pine_test.py

# A simple python script to evaluate a models performance using the alpaca API with a paper trading account.

# Update any API keys, links, or parameters as neccessary.


import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi

# ----- CONFIGURATION -----
API_KEY = 'yourapikey'
API_SECRET = 'yourapikey'
BASE_URL = 'https://paper-api.alpaca.markets'  # Alpaca paper trading URL

# Trading parameters
SYMBOL = 'BTC'           # For crypto use "BTC/USD", for equities e.g. "AAPL"
TIMEFRAME = '1Min'           # Trade by minute
ATR_PERIOD = 10              # ATR period (c)
SENSITIVITY = 1              # Sensitivity factor (a)
TRADE_QTY = 1                # Number of shares (or units) to trade

# Mode selection: 'live' for paper trading, 'backtest' for historical simulation.
MODE = 'live'   # Change to 'backtest' if you want to run a backtest

# For backtesting, fetch a larger number of bars (max. allowed by Alpaca)
HISTORICAL_LIMIT = 1000

# Initialize Alpaca API (v2)
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')


def get_recent_bars(symbol, timeframe, limit):
    """
    Fetch recent bars for the given symbol.
    If the symbol contains '/', assume crypto and use get_crypto_bars.
    Otherwise, use get_bars for equities.
    Returns a DataFrame with a DatetimeIndex and columns: open, high, low, close, volume.
    """
    if '/' in symbol:
        # For crypto, remove the slash (e.g. "BTC/USD" -> "BTCUSD")
        crypto_symbol = symbol.replace('/', '')
        bars = api.get_crypto_bars(crypto_symbol, timeframe, limit=limit).df
    else:
        bars = api.get_bars(symbol, timeframe, limit=limit).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.droplevel(0)
    bars.sort_index(inplace=True)
    return bars


def compute_trailing_stop(df, sensitivity, atr_period):
    """
    Given a DataFrame with columns: high, low, close,
    compute ATR and trailing stop.
    """
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df.apply(lambda row: max(
        row['high'] - row['low'],
        abs(row['high'] - row['prev_close']) if pd.notnull(row['prev_close']) else 0,
        abs(row['low'] - row['prev_close']) if pd.notnull(row['prev_close']) else 0
    ), axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period, min_periods=1).mean()
    df['nloss'] = sensitivity * df['atr']

    # Calculate trailing stop (ts)
    ts = np.zeros(len(df))
    close_prices = df['close'].values
    nloss = df['nloss'].values

    for i in range(len(df)):
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

    df['ts'] = ts
    return df


def evaluate_signals(df):
    """
    Look at the last two bars and decide whether a buy or sell signal is generated.
    Returns (signal_buy, signal_sell)
    """
    if len(df) < 2:
        return (False, False)

    prev, latest = df.iloc[-2], df.iloc[-1]

    # Buy signal: previous close <= previous ts and current close > current ts
    signal_buy = (prev['close'] <= prev['ts']) and (latest['close'] > latest['ts'])
    # Sell signal: previous close >= previous ts and current close < current ts
    signal_sell = (prev['close'] >= prev['ts']) and (latest['close'] < latest['ts'])

    return signal_buy, signal_sell


def get_current_position(symbol):
    """
    Get the current position for the symbol.
    Returns a float (qty) or 0 if none.
    """
    try:
        position = api.get_position(symbol)
        return float(position.qty)
    except Exception:
        return 0.0


def wait_for_order_fill(order_id, timeout=30):
    """
    Polls the order status until it is filled or the timeout is reached.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        order = api.get_order(order_id)
        if order.status == 'filled':
            return True
        time.sleep(1)
    return False


def place_order(symbol, qty, side):
    """
    Place a market order.
    For crypto (symbol contains '/') use time_in_force 'gtc',
    otherwise use 'day'. Then wait for the order to fill.
    """
    tif = 'gtc' if '/' in symbol else 'day'
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force=tif
        )
        print(f"Order submitted: {side.upper()} {qty} {symbol} at {datetime.datetime.now()}")
        filled = wait_for_order_fill(order.id)
        if filled:
            print(f"Order {order.id} filled.")
        else:
            print(f"Order {order.id} not filled within timeout. Consider cancelling.")
        return order
    except Exception as e:
        print(f"Order error: {e}")
        return None


def run_live_trading():
    """
    Live trading mode for 24-hour markets.
    For equities, it checks that the market is open.
    For crypto (identified by '/' in SYMBOL), trading is allowed 24/7 so it skips the clock check.
    """
    while True:
        try:
            # Determine if the symbol is for a 24-hour market (crypto)
            is_crypto = '/' in SYMBOL

            # Only check market hours for non-crypto assets.
            if not is_crypto:
                clock = api.get_clock()
                if not clock.is_open:
                    print(f"Market is closed. Next open at {clock.next_open}. Waiting...")
                    time.sleep(60)
                    continue

            df = get_recent_bars(SYMBOL, TIMEFRAME, limit=50)
            df = compute_trailing_stop(df, SENSITIVITY, ATR_PERIOD)
            signal_buy, signal_sell = evaluate_signals(df)

            latest_bar_time = df.index[-1]
            latest_close = df['close'].iloc[-1]
            latest_ts = df['ts'].iloc[-1]
            print(f"{datetime.datetime.now()} - Latest bar ({latest_bar_time}): Close = {latest_close:.2f}, TS = {latest_ts:.2f}")
            print(f"Signals - BUY: {signal_buy}, SELL: {signal_sell}")

            current_position = get_current_position(SYMBOL)
            print(f"Current position: {current_position}")

            # Execute orders based on signals
            if signal_buy and current_position == 0:
                print("Signal BUY detected: Entering long position.")
                place_order(SYMBOL, TRADE_QTY, 'buy')
            elif signal_buy and current_position < 0:
                print("Signal BUY detected: Exiting short and entering long.")
                place_order(SYMBOL, abs(current_position), 'buy')  # Exit short
                time.sleep(1)
                place_order(SYMBOL, TRADE_QTY, 'buy')               # Enter long

            if signal_sell and current_position == 0:
                print("Signal SELL detected: Entering short position.")
                place_order(SYMBOL, TRADE_QTY, 'sell')
            elif signal_sell and current_position > 0:
                print("Signal SELL detected: Exiting long and entering short.")
                place_order(SYMBOL, current_position, 'sell')  # Exit long
                time.sleep(1)
                place_order(SYMBOL, TRADE_QTY, 'sell')           # Enter short

            print("Waiting for next bar...\n")
            time.sleep(60)  # Wait 1 minute

        except Exception as e:
            print(f"Error encountered: {e}")
            time.sleep(60)


def run_backtest():
    """
    Backtesting mode:
    Fetches a large set of historical minute bars and simulates your trailing stop strategy.
    """
    df = get_recent_bars(SYMBOL, TIMEFRAME, limit=HISTORICAL_LIMIT)
    df = compute_trailing_stop(df, SENSITIVITY, ATR_PERIOD)

    # Generate signals based on trailing stop crossovers
    df['buy_signal'] = False
    df['sell_signal'] = False
    close_prices = df['close'].values
    ts = df['ts'].values

    for i in range(1, len(df)):
        if (close_prices[i - 1] <= ts[i - 1]) and (close_prices[i] > ts[i]):
            df.iloc[i, df.columns.get_loc('buy_signal')] = True
        if (close_prices[i - 1] >= ts[i - 1]) and (close_prices[i] < ts[i]):
            df.iloc[i, df.columns.get_loc('sell_signal')] = True

    # ----- SIMULATE TRADING -----
    position = 0         # 1 for long, -1 for short, 0 for flat
    entry_price = 0.0
    cumulative_profit = 0.0
    equity_curve = []    # To store cumulative profit over time
    trade_log = []       # To record trade details

    for i in range(len(df)):
        price = close_prices[i]
        stop = ts[i]
        signal_buy = df['buy_signal'].iloc[i]
        signal_sell = df['sell_signal'].iloc[i]

        # Exit conditions for long
        if position == 1 and price < stop:
            exit_price = stop
            profit = exit_price - entry_price
            cumulative_profit += profit
            trade_log.append({
                'type': 'long_exit_stop',
                'entry': entry_price,
                'exit': exit_price,
                'profit': profit,
                'date': df.index[i]
            })
            position = 0
            entry_price = 0.0

        # Exit conditions for short
        if position == -1 and price > stop:
            exit_price = stop
            profit = entry_price - exit_price
            cumulative_profit += profit
            trade_log.append({
                'type': 'short_exit_stop',
                'entry': entry_price,
                'exit': exit_price,
                'profit': profit,
                'date': df.index[i]
            })
            position = 0
            entry_price = 0.0

        # Entry conditions
        if signal_buy:
            if position == -1:
                exit_price = price
                profit = entry_price - exit_price
                cumulative_profit += profit
                trade_log.append({
                    'type': 'short_exit_signal',
                    'entry': entry_price,
                    'exit': exit_price,
                    'profit': profit,
                    'date': df.index[i]
                })
                position = 0
                entry_price = 0.0
            if position == 0:
                position = 1
                entry_price = price
                trade_log.append({
                    'type': 'long_entry',
                    'entry': entry_price,
                    'date': df.index[i]
                })

        if signal_sell:
            if position == 1:
                exit_price = price
                profit = exit_price - entry_price
                cumulative_profit += profit
                trade_log.append({
                    'type': 'long_exit_signal',
                    'entry': entry_price,
                    'exit': exit_price,
                    'profit': profit,
                    'date': df.index[i]
                })
                position = 0
                entry_price = 0.0
            if position == 0:
                position = -1
                entry_price = price
                trade_log.append({
                    'type': 'short_entry',
                    'entry': entry_price,
                    'date': df.index[i]
                })

        # Mark-to-market
        if position == 1:
            unrealized = price - entry_price
        elif position == -1:
            unrealized = entry_price - price
        else:
            unrealized = 0
        equity_curve.append(cumulative_profit + unrealized)

    df['equity'] = equity_curve

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity'], label="Equity Curve")
    plt.title("UT Bot Strategy Profit Over Time (Backtest)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    trade_df = pd.DataFrame(trade_log)
    print(trade_df)


# ----- MAIN -----
if __name__ == '__main__':
    if MODE == 'live':
        run_live_trading()
    elif MODE == 'backtest':
        run_backtest()
    else:
        print("Invalid MODE selected. Please set MODE to 'live' or 'backtest'.")
