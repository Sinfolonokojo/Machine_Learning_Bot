"""
Fetch Historical Data from MetaTrader 5
========================================

This script downloads historical OHLCV data from MT5 for training the DRL agent.

Requirements:
    pip install MetaTrader5

Usage:
    python fetch_mt5_data.py

Author: DRL Trading System
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os


def fetch_mt5_data(
    symbol: str = "EURUSD",
    timeframe: str = "M15",  # 15-minute candles
    num_candles: int = 50000,  # ~1.5 years of 15min data
    output_path: str = "data/EURUSD_15m.csv"
):
    """
    Fetch historical data from MT5.

    Args:
        symbol: Trading symbol (e.g., "EURUSD", "GBPUSD")
        timeframe: Timeframe ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
        num_candles: Number of historical candles to fetch
        output_path: Where to save the CSV file

    Returns:
        DataFrame with OHLCV data
    """
    print("=" * 80)
    print("FETCHING DATA FROM METATRADER 5")
    print("=" * 80)

    # Initialize MT5 connection
    print("\n[1] Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print("[ERROR] MT5 initialization failed!")
        print("Make sure MetaTrader 5 is installed and running.")
        print("Error code:", mt5.last_error())
        return None

    print("[OK] Connected to MT5")
    print(f"    Terminal: {mt5.terminal_info().name}")
    print(f"    Version: {mt5.version()}")

    # Map timeframe string to MT5 constant
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    if timeframe not in timeframe_map:
        print(f"[ERROR] Invalid timeframe: {timeframe}")
        print(f"Valid options: {list(timeframe_map.keys())}")
        mt5.shutdown()
        return None

    mt5_timeframe = timeframe_map[timeframe]

    # Fetch historical data
    print(f"\n[2] Fetching {num_candles} candles of {symbol} ({timeframe})...")

    # Get current time and calculate start time
    now = datetime.now()

    # Fetch data
    rates = mt5.copy_rates_from(symbol, mt5_timeframe, now, num_candles)

    if rates is None or len(rates) == 0:
        print("[ERROR] Failed to fetch data!")
        print("Error code:", mt5.last_error())
        print("\nPossible reasons:")
        print("  1. Symbol name is incorrect (check MT5 Market Watch)")
        print("  2. Symbol is not available in your broker")
        print("  3. No historical data available for this symbol")
        mt5.shutdown()
        return None

    print(f"[OK] Fetched {len(rates)} candles")

    # Convert to DataFrame
    print("\n[3] Converting to DataFrame...")
    df = pd.DataFrame(rates)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')

    # Rename columns to match our format
    df = df.rename(columns={
        'tick_volume': 'volume'
    })

    # Select only required columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"[OK] DataFrame created")
    print(f"    Shape: {df.shape}")
    print(f"    Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"    Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

    # Save to CSV
    print(f"\n[4] Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("[OK] Data saved successfully!")

    # Display sample
    print("\n[5] Sample data (first 5 rows):")
    print(df.head())

    print("\n[6] Sample data (last 5 rows):")
    print(df.tail())

    # Shutdown MT5
    mt5.shutdown()
    print("\n[OK] MT5 connection closed")

    print("\n" + "=" * 80)
    print("DATA FETCH COMPLETE!")
    print("=" * 80)
    print(f"\nYour data is ready at: {output_path}")
    print(f"Total candles: {len(df)}")
    print(f"\nNext step:")
    print(f"  python src/training/train_ppo.py")
    print("=" * 80)

    return df


def list_available_symbols():
    """
    List all available symbols in MT5.
    """
    print("=" * 80)
    print("AVAILABLE SYMBOLS IN MT5")
    print("=" * 80)

    if not mt5.initialize():
        print("[ERROR] Cannot connect to MT5")
        return

    symbols = mt5.symbols_get()

    if symbols is None or len(symbols) == 0:
        print("No symbols found")
        mt5.shutdown()
        return

    # Filter forex symbols
    forex_symbols = [s for s in symbols if s.name in [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF',
        'NZDUSD', 'USDCAD', 'EURJPY', 'GBPJPY', 'EURGBP'
    ]]

    print("\nMajor Forex Pairs:")
    for symbol in forex_symbols:
        print(f"  {symbol.name:10} - {symbol.description}")

    print(f"\nTotal symbols available: {len(symbols)}")
    print("(Showing only major forex pairs above)")

    mt5.shutdown()
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MT5 DATA FETCHER FOR DRL TRADING BOT")
    print("=" * 80)

    # Check if MT5 is installed
    try:
        import MetaTrader5 as mt5
        print("\n[OK] MetaTrader5 package found")
    except ImportError:
        print("\n[ERROR] MetaTrader5 package not found!")
        print("\nInstall it with:")
        print("  pip install MetaTrader5")
        print("\nAlso ensure MetaTrader 5 terminal is installed:")
        print("  Download from: https://www.metatrader5.com/en/download")
        exit(1)

    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Fetch EURUSD 15-minute data (recommended)")
    print("2. Fetch GBPUSD 15-minute data")
    print("3. List available symbols")
    print("4. Custom symbol/timeframe")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        fetch_mt5_data(
            symbol="EURUSD",
            timeframe="M15",
            num_candles=50000,
            output_path="data/EURUSD_15m.csv"
        )

    elif choice == "2":
        fetch_mt5_data(
            symbol="GBPUSD",
            timeframe="M15",
            num_candles=50000,
            output_path="data/GBPUSD_15m.csv"
        )

    elif choice == "3":
        list_available_symbols()

    elif choice == "4":
        symbol = input("Enter symbol (e.g., EURUSD): ").strip().upper()
        timeframe = input("Enter timeframe (M1/M5/M15/M30/H1/H4/D1): ").strip().upper()
        num_candles = int(input("Enter number of candles (e.g., 50000): ").strip())
        output_path = f"data/{symbol}_{timeframe}.csv"

        fetch_mt5_data(
            symbol=symbol,
            timeframe=timeframe,
            num_candles=num_candles,
            output_path=output_path
        )

    else:
        print("Invalid choice!")
