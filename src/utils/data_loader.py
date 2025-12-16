"""
Data Loading and Validation Utilities
======================================

This module handles loading Forex OHLCV data from CSV files and performing
validation checks to ensure data quality for RL training.

Expected CSV format:
    timestamp,open,high,low,close,volume
    2024-01-01 00:00:00,1.0950,1.0955,1.0948,1.0952,1234
    ...

Author: DRL Trading System
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings


class ForexDataLoader:
    """
    Loads and validates Forex OHLCV data from CSV files.

    This class ensures data quality by checking for:
    - Missing values
    - Chronological ordering
    - Valid OHLC relationships (high >= low, etc.)
    - Sufficient data points
    """

    def __init__(self, min_rows: int = 100):
        """
        Initialize the data loader.

        Args:
            min_rows: Minimum number of rows required for valid dataset
        """
        self.min_rows = min_rows

    def load_forex_data(self, csv_path: str, validate: bool = True) -> pd.DataFrame:
        """
        Load Forex data from CSV file.

        Args:
            csv_path: Path to CSV file
            validate: Whether to run validation checks

        Returns:
            DataFrame with timestamp index and OHLCV columns

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data validation fails
        """
        # Check file exists
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"Loading data from: {csv_path}")

        # Load CSV
        try:
            df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        except KeyError:
            # Try alternative timestamp column names
            df = pd.read_csv(csv_path)
            timestamp_cols = ['timestamp', 'datetime', 'time', 'date', 'Date', 'Time']
            found_col = None
            for col in timestamp_cols:
                if col in df.columns:
                    found_col = col
                    break

            if found_col:
                df['timestamp'] = pd.to_datetime(df[found_col])
                df = df.drop(columns=[found_col])
            else:
                raise ValueError("No timestamp column found. Expected 'timestamp', 'datetime', or 'time'")

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Ensure columns are lowercase
        df.columns = df.columns.str.lower()

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Reorder columns
        df = df[required_cols]

        print(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")

        # Run validation checks
        if validate:
            self.validate_data(df)

        return df

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data quality.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        print("\nRunning data validation checks...")

        # 1. Check minimum rows
        if len(df) < self.min_rows:
            raise ValueError(f"Insufficient data: {len(df)} rows < {self.min_rows} minimum")
        print(f"[OK] Sufficient data: {len(df)} rows")

        # 2. Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"[WARN] Missing values detected:\n{null_counts[null_counts > 0]}")
            warnings.warn("Data contains missing values - will be forward-filled")
        else:
            print("[OK] No missing values")

        # 3. Check chronological ordering
        if not df.index.is_monotonic_increasing:
            raise ValueError("Data is not chronologically ordered")
        print("[OK] Data is chronologically ordered")

        # 4. Check for duplicate timestamps
        if df.index.duplicated().any():
            num_duplicates = df.index.duplicated().sum()
            print(f"[WARN] {num_duplicates} duplicate timestamps found - removing...")
            df = df[~df.index.duplicated(keep='first')]
        else:
            print("[OK] No duplicate timestamps")

        # 5. Validate OHLC relationships
        invalid_high = (df['high'] < df['low']).sum()
        invalid_close_high = (df['close'] > df['high']).sum()
        invalid_close_low = (df['close'] < df['low']).sum()
        invalid_open_high = (df['open'] > df['high']).sum()
        invalid_open_low = (df['open'] < df['low']).sum()

        total_invalid = (invalid_high + invalid_close_high + invalid_close_low +
                        invalid_open_high + invalid_open_low)

        if total_invalid > 0:
            print(f"[WARN] {total_invalid} rows with invalid OHLC relationships")
            if invalid_high > 0:
                print(f"  - {invalid_high} rows where high < low")
            if invalid_close_high > 0:
                print(f"  - {invalid_close_high} rows where close > high")
            if invalid_close_low > 0:
                print(f"  - {invalid_close_low} rows where close < low")
        else:
            print("[OK] Valid OHLC relationships")

        # 6. Check for zero or negative prices
        zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if zero_prices > 0:
            raise ValueError(f"{zero_prices} rows contain zero or negative prices")
        print("[OK] All prices are positive")

        # 7. Check for extreme price changes (potential data errors)
        returns = df['close'].pct_change()
        extreme_returns = (np.abs(returns) > 0.1).sum()  # 10% change in one candle
        if extreme_returns > 0:
            print(f"[WARN] {extreme_returns} candles with >10% price change (possible data errors)")
        else:
            print("[OK] No extreme price movements detected")

        print("\nValidation complete!\n")

    def split_data(self, df: pd.DataFrame,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets (chronological split).

        For time series, we MUST use chronological splitting, not random.

        Args:
            df: DataFrame to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), \
            "Ratios must sum to 1.0"

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"Data split:")
        print(f"  Train: {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"  Val:   {len(val_df)} rows ({val_df.index[0]} to {val_df.index[-1]})")
        print(f"  Test:  {len(test_df)} rows ({test_df.index[0]} to {test_df.index[-1]})")

        return train_df, val_df, test_df

    def create_sample_data(self, output_path: str, num_rows: int = 10000,
                          timeframe: str = '15min', seed: int = 42) -> pd.DataFrame:
        """
        Create sample Forex data for testing (realistic random walk).

        This is useful for initial development and testing before obtaining real data.

        Args:
            output_path: Path to save CSV file
            num_rows: Number of candles to generate
            timeframe: Candle timeframe (e.g., '15min', '1h', '4h', '1d')
            seed: Random seed for reproducibility

        Returns:
            Generated DataFrame
        """
        np.random.seed(seed)

        # Generate timestamps
        dates = pd.date_range('2023-01-01', periods=num_rows, freq=timeframe)

        # Simulate realistic EURUSD price movement
        # Starting price around 1.0950
        initial_price = 1.0950
        drift = 0.00001  # Slight upward drift
        volatility = 0.0002  # ~2 pips per candle volatility

        # Generate close prices (geometric random walk)
        returns = np.random.randn(num_rows) * volatility + drift
        close_prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLC from close prices
        high_add = np.abs(np.random.randn(num_rows)) * volatility * 0.5
        low_sub = np.abs(np.random.randn(num_rows)) * volatility * 0.5
        open_offset = np.random.randn(num_rows) * volatility * 0.3

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + open_offset,
            'high': close_prices + high_add,
            'low': close_prices - low_sub,
            'close': close_prices,
            'volume': np.random.randint(1000, 5000, num_rows)
        })

        # Ensure OHLC relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Sample data created: {output_path}")
        print(f"  Rows: {num_rows}")
        print(f"  Timeframe: {timeframe}")
        print(f"  Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

        return df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the ForexDataLoader.

    This demonstrates:
    1. Creating sample data
    2. Loading and validating data
    3. Splitting into train/val/test sets
    """

    loader = ForexDataLoader()

    print("=" * 80)
    print("FOREX DATA LOADER - TEST MODE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # OPTION 1: Create sample data (for testing without real data)
    # -------------------------------------------------------------------------
    print("\n[1] Creating sample EURUSD data...")
    sample_path = "data/EURUSD_15m_sample.csv"
    sample_df = loader.create_sample_data(
        output_path=sample_path,
        num_rows=5000,
        timeframe='15min',
        seed=42
    )

    # -------------------------------------------------------------------------
    # OPTION 2: Load real data (uncomment when you have real CSV files)
    # -------------------------------------------------------------------------
    # print("\n[2] Loading real EURUSD data...")
    # real_path = "data/EURUSD_15m.csv"  # Replace with your real data path
    # df = loader.load_forex_data(real_path, validate=True)

    # -------------------------------------------------------------------------
    # Load the sample data we just created
    # -------------------------------------------------------------------------
    print(f"\n[2] Loading data from CSV...")
    df = loader.load_forex_data(sample_path, validate=True)

    # -------------------------------------------------------------------------
    # Split data
    # -------------------------------------------------------------------------
    print(f"\n[3] Splitting data into train/val/test...")
    train_df, val_df, test_df = loader.split_data(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print("\n" + "=" * 80)
    print("DATA READY FOR TRAINING")
    print("=" * 80)
    print("\nTo use real data:")
    print("1. Place your CSV file in the data/ folder")
    print("2. Ensure format: timestamp,open,high,low,close,volume")
    print("3. Update the csv_path in train_ppo.py")
    print("=" * 80)
