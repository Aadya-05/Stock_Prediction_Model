"""
Data fetching module for FinSight AI
Fetches historical S&P 500 stock data using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Time period to fetch ('5y', '10y', etc.)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data fetched for ticker {ticker}")
        
        # Reset index to have Date as a column
        df.reset_index(inplace=True)
        
        # Ensure Date column exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Handle missing values - forward fill then backward fill
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Drop any remaining NaN values
        df.dropna(inplace=True)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")


def prepare_supervised_data(
    df: pd.DataFrame,
    lookback: int = 60,
    forecast_horizon: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert time-series data into supervised learning dataset.
    
    Args:
        df: DataFrame with features (including Close price)
        lookback: Number of days to look back for LSTM input
        forecast_horizon: Number of days ahead to predict
    
    Returns:
        X_train, X_test, y_train, y_test arrays
    """
    # Extract features (all columns except Close which will be the target)
    feature_cols = [col for col in df.columns if col != 'Close']
    
    # Get the Close prices for labels (shifted forward by forecast_horizon)
    close_prices = df['Close'].values
    
    # Create sequences
    X, y = [], []
    
    for i in range(lookback, len(close_prices) - forecast_horizon + 1):
        # Input: features from lookback window
        X.append(df[feature_cols].iloc[i-lookback:i].values)
        # Output: close price forecast_horizon days ahead
        y.append(close_prices[i + forecast_horizon - 1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train/test (80/20)
    split_idx = int(len(X) * 0.8)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def prepare_linear_data(
    df: pd.DataFrame,
    forecast_horizon: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for linear regression model.
    Uses all features at time t to predict price at time t+forecast_horizon.
    
    Args:
        df: DataFrame with features
        forecast_horizon: Number of days ahead to predict
    
    Returns:
        X_train, X_test, y_train, y_test arrays
    """
    # Extract features
    feature_cols = [col for col in df.columns if col != 'Close']
    
    # Get close prices shifted forward
    close_prices = df['Close'].values
    y = close_prices[forecast_horizon:]
    X = df[feature_cols].iloc[:-forecast_horizon].values
    
    # Remove any rows with NaN
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Split into train/test (80/20)
    split_idx = int(len(X) * 0.8)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test
