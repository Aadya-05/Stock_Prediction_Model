"""
Evaluation metrics module for FinSight AI
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE value (as percentage)
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def improvement_percentage(linear_rmse: float, lstm_rmse: float) -> float:
    """
    Calculate improvement percentage of LSTM over Linear Regression.
    
    Args:
        linear_rmse: RMSE of Linear Regression model
        lstm_rmse: RMSE of LSTM model
    
    Returns:
        Improvement percentage
    """
    if linear_rmse == 0:
        return 0.0
    return ((linear_rmse - lstm_rmse) / linear_rmse) * 100
