"""
LSTM model for FinSight AI
Primary model using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


class LSTMStockPredictor:
    """
    LSTM model for stock price prediction.
    """
    
    def __init__(self, lookback: int = 60, n_features: int = None, lstm_units: int = 50, dropout_rate: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            lookback: Number of time steps to look back
            n_features: Number of features (will be set during fit if None)
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        """
        self.lookback = lookback
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_fitted = False
    
    def _build_model(self):
        """Build the LSTM model architecture."""
        self.model = Sequential([
            LSTM(self.lstm_units, input_shape=(self.lookback, self.n_features), return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features (shape: [samples, lookback, features])
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        # Set n_features if not set
        if self.n_features is None:
            self.n_features = X_train.shape[2]
        
        # Reshape y if needed
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Scale features
        # Reshape for scaling: (samples * lookback, features)
        n_samples, lookback, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = self.scaler_X.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, lookback, n_features)
        
        # Scale targets
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        
        # Build model if not built
        if self.model is None:
            self._build_model()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            n_samples_val = X_val.shape[0]
            X_val_reshaped = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler_X.transform(X_val_reshaped)
            X_val_scaled = X_val_scaled.reshape(n_samples_val, lookback, n_features)
            
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            y_val_scaled = self.scaler_y.transform(y_val)
            
            validation_data = (X_val_scaled, y_val_scaled)
        
        # Train model
        self.model.fit(
            X_train_scaled,
            y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose,
            shuffle=False  # Important for time series
        )
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (shape: [samples, lookback, features])
        
        Returns:
            Predictions (in original scale)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        n_samples = X.shape[0]
        n_features = X.shape[2]
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, self.lookback, n_features)
        
        # Predict (in scaled space)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Inverse transform to original scale
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).ravel()
        
        return y_pred
    
    def save(self, filepath: str):
        """Save the model to disk."""
        self.model.save(filepath.replace('.pkl', '_model.h5'))
        with open(filepath, 'wb') as f:
            pickle.dump({
                'lookback': self.lookback,
                'n_features': self.n_features,
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'is_fitted': self.is_fitted,
                'model_path': filepath.replace('.pkl', '_model.h5')
            }, f)
    
    def load(self, filepath: str):
        """Load the model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.lookback = data['lookback']
            self.n_features = data['n_features']
            self.lstm_units = data['lstm_units']
            self.dropout_rate = data['dropout_rate']
            self.scaler_X = data['scaler_X']
            self.scaler_y = data['scaler_y']
            self.is_fitted = data['is_fitted']
            self.model = keras.models.load_model(data['model_path'])
