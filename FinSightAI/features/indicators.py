"""
Financial feature engineering module for FinSight AI
Implements RSI, MACD, and Moving Averages from scratch using Pandas
"""

import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) from scratch.
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss using exponential moving average
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) from scratch.
    
    Args:
        prices: Series of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
    
    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    # Create DataFrame
    macd_df = pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })
    
    return macd_df


def calculate_sma(prices: pd.Series, period: int = 50) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Series of closing prices
        period: SMA period (default 50)
    
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all financial features and add them to the DataFrame.
    
    Args:
        df: DataFrame with OHLCV data (must have 'Close' column)
    
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()
    
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must have 'Close' column")
    
    close_prices = df['Close']
    
    # Calculate RSI (14-period)
    df['RSI'] = calculate_rsi(close_prices, period=14)
    
    # Calculate MACD (12-26-9)
    macd_df = calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9)
    df['MACD'] = macd_df['MACD']
    df['MACD_Signal'] = macd_df['Signal']
    df['MACD_Histogram'] = macd_df['Histogram']
    
    # Calculate 50-Day SMA
    df['SMA_50'] = calculate_sma(close_prices, period=50)
    
    # Drop rows with NaN values (from initial periods of indicators)
    df.dropna(inplace=True)
    
    return df
