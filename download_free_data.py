"""
Download Free Historical Forex Data
====================================

Downloads real historical data from free sources and converts to our format.

Sources:
- Dukascopy (Switzerland bank, high quality data)
- HistData.com (free historical tick data)

Author: DRL Trading System
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import os
from io import StringIO


def download_dukascopy_data(
    symbol: str = "EURUSD",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-15",
    timeframe: str = "15m",
    output_path: str = "data/EURUSD_15m.csv"
):
    """
    Download data from Dukascopy (free, high quality).

    Note: This is a simplified version. For production, use the
    dukascopy-node package or manual download from:
    https://www.dukascopy.com/swiss/english/marketwatch/historical/

    Args:
        symbol: Currency pair (e.g., "EURUSD")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        output_path: Where to save CSV
    """
    print("=" * 80)
    print("DOWNLOAD FREE HISTORICAL DATA")
    print("=" * 80)

    print("\n[INFO] For now, please manually download data from:")
    print("\nOption A - Dukascopy (Recommended):")
    print("  1. Go to: https://www.dukascopy.com/swiss/english/marketwatch/historical/")
    print(f"  2. Select: {symbol}")
    print(f"  3. Select timeframe: {timeframe}")
    print(f"  4. Select date range: {start_date} to {end_date}")
    print("  5. Download as CSV")
    print(f"  6. Save to: {output_path}")
    print("\nOption B - HistData.com:")
    print("  1. Go to: http://www.histdata.com/download-free-forex-data/")
    print(f"  2. Select: {symbol}")
    print("  3. Select timeframe: M15")
    print("  4. Download ZIP files")
    print("  5. Extract and combine CSVs")

    print("\nOption C - MetaTrader 5 (Best!):")
    print("  1. Install MT5: https://www.metatrader5.com/en/download")
    print("  2. Open MT5 terminal")
    print("  3. Right-click on chart -> 'Export' -> 'Export to CSV'")
    print(f"  4. Save to: {output_path}")

    print("\n" + "=" * 80)
    print("MANUAL DOWNLOAD GUIDE")
    print("=" * 80)

    return None


def convert_mt5_export_to_format(input_csv: str, output_csv: str):
    """
    Convert MT5 exported CSV to our required format.

    MT5 export format:
        Date,Time,Open,High,Low,Close,Volume

    Our required format:
        timestamp,open,high,low,close,volume
    """
    print(f"\n[1] Reading MT5 export: {input_csv}")

    # Try different MT5 export formats
    try:
        # Format 1: Date,Time,Open,High,Low,Close,Volume
        df = pd.read_csv(input_csv)

        if 'Date' in df.columns and 'Time' in df.columns:
            # Combine Date and Time
            df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
        elif 'timestamp' in df.columns or 'Timestamp' in df.columns:
            # Already has timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df['Timestamp'])
        else:
            # Try first column as timestamp
            df['timestamp'] = pd.to_datetime(df.iloc[:, 0])

        # Select required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Save
        print(f"[2] Saving to: {output_csv}")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

        print(f"[OK] Conversion complete!")
        print(f"    Rows: {len(df)}")
        print(f"    Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        return df

    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        print("\nPlease ensure your CSV has these columns:")
        print("  Date,Time,Open,High,Low,Close,Volume")
        print("  OR")
        print("  timestamp,open,high,low,close,volume")
        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FREE FOREX DATA DOWNLOADER")
    print("=" * 80)

    print("\nWhat would you like to do?")
    print("1. View download instructions (manual download)")
    print("2. Convert MT5 exported CSV to our format")
    print("3. Convert HistData CSV to our format")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        download_dukascopy_data()

    elif choice == "2":
        print("\n[INFO] Converting MT5 export...")
        input_file = input("Enter path to MT5 CSV file: ").strip()
        output_file = input("Enter output path (e.g., data/EURUSD_15m.csv): ").strip()

        if not output_file:
            output_file = "data/EURUSD_15m.csv"

        convert_mt5_export_to_format(input_file, output_file)

    elif choice == "3":
        print("\n[INFO] HistData format conversion...")
        print("HistData CSVs are usually already in the correct format!")
        print("Just ensure they have: timestamp,open,high,low,close,volume")

        input_file = input("Enter path to HistData CSV: ").strip()
        output_file = input("Enter output path (e.g., data/EURUSD_15m.csv): ").strip()

        if not output_file:
            output_file = "data/EURUSD_15m.csv"

        convert_mt5_export_to_format(input_file, output_file)

    else:
        print("Invalid choice!")

    print("\n" + "=" * 80)
    print("Once you have your data in data/EURUSD_15m.csv:")
    print("  python src/training/train_ppo.py")
    print("=" * 80)
