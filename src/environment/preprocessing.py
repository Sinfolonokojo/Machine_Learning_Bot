"""
Preprocessing Pipeline for Forex Trading Data
==============================================

This module transforms raw OHLCV price data into stationary, normalized features
suitable for Deep Reinforcement Learning. All features are designed to avoid
non-stationarity issues inherent in raw price data.

Key transformations:
- Log returns instead of raw prices
- Normalized technical indicators (scaled to [0,1] or [-1,1])
- Volatility-adjusted metrics using ATR

Author: DRL Trading System
"""

import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class ForexPreprocessor:
    """
    Preprocesses OHLCV data for stationary feature extraction.

    This class handles all feature engineering, ensuring the neural network
    receives only normalized, stationary inputs (no raw prices).
    """

    def __init__(self,
                 sma_periods: list = [20, 50],
                 ema_periods: list = [12, 26],
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 bb_period: int = 20,
                 bb_std: int = 2,
                 stoch_period: int = 14,
                 atr_period: int = 14):
        """
        Initialize preprocessor with indicator parameters.

        Args:
            sma_periods: List of SMA window sizes
            ema_periods: List of EMA window sizes
            rsi_period: RSI lookback period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            stoch_period: Stochastic oscillator period
            atr_period: Average True Range period
        """
        self.sma_periods = sma_periods
        self.ema_periods = ema_periods
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.stoch_period = stoch_period
        self.atr_period = atr_period

    def calculate_log_returns(self, prices: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate log returns to ensure stationarity.

        Log returns are preferred over simple returns for:
        - Time-additivity
        - Better statistical properties
        - Normality assumption

        Args:
            prices: Price series (close prices)
            periods: Number of periods for return calculation

        Returns:
            Log returns series
        """
        return np.log(prices / prices.shift(periods))

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI measures momentum and is normalized to [0, 100].
        We further normalize by dividing by 100 to get [0, 1].

        Args:
            prices: Price series
            period: RSI period (default 14)

        Returns:
            RSI values normalized to [0, 1]
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi / 100.0  # Normalize to [0, 1]

    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD shows the relationship between two moving averages.
        Returns three components: MACD line, signal line, and histogram.

        Args:
            prices: Price series

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands and %B position.

        %B indicates where price is relative to the bands:
        - 1.0 = at upper band
        - 0.5 = at middle band (SMA)
        - 0.0 = at lower band

        Args:
            prices: Price series

        Returns:
            Tuple of (upper_band, middle_band, lower_band, percent_b)
        """
        middle_band = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()

        upper_band = middle_band + (self.bb_std * std)
        lower_band = middle_band - (self.bb_std * std)

        # %B calculation
        percent_b = (prices - lower_band) / (upper_band - lower_band + 1e-10)

        return upper_band, middle_band, lower_band, percent_b

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D).

        Stochastic compares closing price to price range over a period.
        Already bounded [0, 100], we normalize to [0, 1].

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Tuple of (%K, %D) normalized to [0, 1]
        """
        lowest_low = low.rolling(window=self.stoch_period).min()
        highest_high = high.rolling(window=self.stoch_period).max()

        # %K calculation
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # %D is 3-period SMA of %K
        stoch_d = stoch_k.rolling(window=3).mean()

        return stoch_k / 100.0, stoch_d / 100.0  # Normalize to [0, 1]

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR measures volatility and is used to normalize other indicators.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            ATR series
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()

        return atr

    def normalize_by_atr(self, value: pd.Series, price: pd.Series, atr: pd.Series) -> pd.Series:
        """
        Normalize a value by ATR for volatility adjustment.

        This makes indicators comparable across different volatility regimes.

        Args:
            value: Value to normalize (e.g., distance from moving average)
            price: Reference price
            atr: ATR series

        Returns:
            Normalized series
        """
        return value / (atr + 1e-10)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps to a DataFrame.

        This is the main method that creates all features from raw OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with normalized features (NO raw prices)
        """
        df = df.copy()

        # Extract OHLC
        close = df['close']
        high = df['high']
        low = df['low']

        # 1. LOG RETURNS (multiple periods for temporal context)
        df['log_return_1'] = self.calculate_log_returns(close, periods=1)
        df['log_return_3'] = self.calculate_log_returns(close, periods=3)
        df['log_return_5'] = self.calculate_log_returns(close, periods=5)

        # 2. ATR (used for normalization)
        atr = self.calculate_atr(high, low, close)
        df['atr_normalized'] = atr / (close + 1e-10)  # Normalize by price level

        # 3. RSI
        df['rsi'] = self.calculate_rsi(close, self.rsi_period)

        # 4. MOVING AVERAGES (normalized by ATR)
        for period in self.sma_periods:
            sma = close.rolling(window=period).mean()
            df[f'sma_{period}_dist'] = self.normalize_by_atr(close - sma, close, atr)

        for period in self.ema_periods:
            ema = close.ewm(span=period, adjust=False).mean()
            df[f'ema_{period}_dist'] = self.normalize_by_atr(close - ema, close, atr)

        # 5. MACD (normalized by ATR)
        macd_line, signal_line, histogram = self.calculate_macd(close)
        df['macd_line'] = self.normalize_by_atr(macd_line, close, atr)
        df['macd_signal'] = self.normalize_by_atr(signal_line, close, atr)
        df['macd_hist'] = self.normalize_by_atr(histogram, close, atr)

        # 6. BOLLINGER BANDS
        upper_band, middle_band, lower_band, percent_b = self.calculate_bollinger_bands(close)
        df['bb_percent_b'] = percent_b  # Already normalized [0, 1]

        # 7. STOCHASTIC
        stoch_k, stoch_d = self.calculate_stochastic(high, low, close)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        # Drop raw OHLCV columns (CRITICAL: don't feed raw prices to NN)
        df = df.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore')

        # Forward fill any NaN values created by indicators
        df = df.fillna(method='ffill').fillna(0)

        return df

    def get_feature_names(self) -> list:
        """
        Get list of all feature names that will be generated.

        Returns:
            List of feature column names
        """
        features = [
            'log_return_1', 'log_return_3', 'log_return_5',
            'atr_normalized', 'rsi',
            'macd_line', 'macd_signal', 'macd_hist',
            'bb_percent_b', 'stoch_k', 'stoch_d'
        ]

        # Add SMA distances
        for period in self.sma_periods:
            features.append(f'sma_{period}_dist')

        # Add EMA distances
        for period in self.ema_periods:
            features.append(f'ema_{period}_dist')

        return features


# ============================================================================
# USAGE EXAMPLE (for testing/debugging)
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the ForexPreprocessor.

    To test with real data:
    1. Place a CSV file in the data/ folder with columns: timestamp, open, high, low, close, volume
    2. Update the file path below
    3. Run: python src/environment/preprocessing.py
    """

    # Create dummy data for testing
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')

    # Simulate realistic forex price movement (random walk with drift)
    close_prices = 1.0950 + np.cumsum(np.random.randn(1000) * 0.0001)

    dummy_df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(1000) * 0.0001,
        'high': close_prices + np.abs(np.random.randn(1000) * 0.0002),
        'low': close_prices - np.abs(np.random.randn(1000) * 0.0002),
        'close': close_prices,
        'volume': np.random.randint(1000, 5000, 1000)
    })

    print("=" * 80)
    print("FOREX PREPROCESSOR - TEST MODE")
    print("=" * 80)
    print(f"\nOriginal data shape: {dummy_df.shape}")
    print(f"Date range: {dummy_df['timestamp'].iloc[0]} to {dummy_df['timestamp'].iloc[-1]}")

    # Initialize preprocessor
    preprocessor = ForexPreprocessor()

    # Process data
    processed_df = preprocessor.process_dataframe(dummy_df.drop(columns=['timestamp']))

    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"Number of features: {len(preprocessor.get_feature_names())}")
    print(f"\nFeature names:\n{preprocessor.get_feature_names()}")

    print(f"\nFirst 5 rows of processed data:")
    print(processed_df.head())

    print(f"\nFeature statistics:")
    print(processed_df.describe())

    print("\n" + "=" * 80)
    print("CRITICAL CHECK: Verify no raw prices in features")
    print("=" * 80)
    raw_price_cols = ['open', 'high', 'low', 'close']
    has_raw_prices = any(col in processed_df.columns for col in raw_price_cols)
    print(f"Contains raw prices: {has_raw_prices}")

    if not has_raw_prices:
        print("✓ SUCCESS: No raw prices found - data is suitable for RL training")
    else:
        print("✗ ERROR: Raw prices detected - do not use for training!")

    print("\n" + "=" * 80)
