"""
Hybrid CNN-LSTM architecture for temporal classification.

Implements a deep learning model combining Convolutional Neural Networks (CNN)
for spatial feature extraction with LSTM layers for temporal sequence modeling.
"""

import tensorflow as tf
from tensorflow import keras
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CNNLSTMClassifier:
    """CNN-LSTM hybrid model for time-series classification.
    
    Architecture:
    - Input: (timesteps, features) = (26, 13)
    - Conv1D layers for spatial feature extraction
    - LSTM layers for temporal sequence modeling
    - Dense layers for classification
    - Output: 5 classes (Poppy, Wheat, Bare, Vegetation, Built/Water)
    """
    
    def __init__(self, n_timesteps: int = 26, n_features: int = 13,
                 n_classes: int = 5, dropout_rate: float = 0.3):
        """Initialize CNN-LSTM architecture.
        
        Args:
            n_timesteps (int): Number of time steps (default: 26)
            n_features (int): Number of input features (default: 13)
            n_classes (int): Number of output classes (default: 5)
            dropout_rate (float): Dropout rate (default: 0.3)
        """
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        logger.info(f"Initialized CNNLSTMClassifier (T={n_timesteps}, F={n_features}, C={n_classes})")
    
    def _build_model(self) -> keras.Model:
        """Build CNN-LSTM model architecture.
        
        Returns:
            keras.Model: Compiled Sequential model
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(self.n_timesteps, self.n_features)),
            keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.LSTM(128, return_sequences=False),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(self.dropout_rate),
            keras.layers.Dense(self.n_classes, activation='softmax'),
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("CNN-LSTM model compiled successfully")
        return model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 100, batch_size: int = 256,
            validation_split: float = 0.2, verbose: int = 1) -> keras.callbacks.History:
        """Train the model.
        
        Args:
            X_train (np.ndarray): Training features (samples, timesteps, features)
            y_train (np.ndarray): Training labels (one-hot encoded)
            epochs (int): Number of training epochs (default: 100)
            batch_size (int): Batch size (default: 256)
            validation_split (float): Validation split ratio (default: 0.2)
            verbose (int): Verbosity level (default: 1)
            
        Returns:
            keras.callbacks.History: Training history
        """
        logger.info(f"Training CNN-LSTM for {epochs} epochs")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        logger.info("CNN-LSTM training complete")
        return history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X_test (np.ndarray): Test features (samples, timesteps, features)
            
        Returns:
            np.ndarray: Predicted class labels
        """
        probabilities = self.model.predict(X_test)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Class probabilities
        """
        return self.model.predict(X_test)
    
    def save(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath (str): Output filepath for model
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model from file.
        
        Args:
            filepath (str): Input filepath for model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
