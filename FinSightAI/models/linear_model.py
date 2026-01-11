"""
Linear Regression model for FinSight AI
Baseline model using Scikit-learn
"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


class LinearStockPredictor:
    """
    Linear Regression model for stock price prediction.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the linear regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        # Reshape y if needed
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        
        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train).ravel()
        
        # Train model
        self.model.fit(X_train_scaled, y_train_scaled)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predictions (in original scale)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler_X.transform(X)
        
        # Predict (in scaled space)
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform to original scale
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).ravel()
        
        return y_pred
    
    def save(self, filepath: str):
        """Save the model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, filepath: str):
        """Load the model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler_X = data['scaler_X']
            self.scaler_y = data['scaler_y']
            self.is_fitted = data['is_fitted']
