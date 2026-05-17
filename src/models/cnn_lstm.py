"""
cnn_lstm.py
===========
Hybrid 1D Convolutional Neural Network – Long Short-Term Memory (CNN-LSTM)
classifier for multi-temporal opium poppy cultivation detection.

Architecture follows Ji et al. (2019) and Rußwurm & Körner (2018), adapted
for the South Asian opium poppy detection problem. CNN layers extract local
temporal patterns; LSTM layers model sequential phenological dynamics.

Author : Fahad Hameed Khan
Paper  : Geospatial Machine Learning for Opium Poppy Cultivation Monitoring
         in Pakistan and Afghanistan: A Sentinel-2 Multi-Temporal Analysis
Journal: Land (MDPI), 2025

References
----------
Ji et al. (2019) Remote Sensing, 10(1), 75.
Rußwurm & Körner (2018) ISPRS Intl J. Geo-Information, 7(4), 129.
Hochreiter & Schmidhuber (1997) Neural Computation, 9(8), 1735–1780.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Class labels
CLASS_NAMES = ["Opium Poppy", "Wheat", "Bare/Fallow", "Other Vegetation", "Built/Water"]
N_CLASSES = 5


def build_cnn_lstm(
    n_timesteps: int = 26,
    n_features: int = 13,
    n_classes: int = N_CLASSES,
    conv_filters: Tuple[int, int] = (64, 128),
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_conv: float = 0.3,
    dropout_lstm: float = 0.2,
    kernel_size: int = 3,
) -> "keras.Model":
    """
    Build the hybrid CNN-LSTM architecture.

    Network design
    ──────────────
    Input (n_timesteps, n_features)
        │
        ├─ Conv1D(64, k=3) + BatchNorm + Dropout(0.3)
        ├─ Conv1D(128, k=3) + BatchNorm + Dropout(0.3)
        ├─ MaxPooling1D
        │
        ├─ LSTM(128, return_sequences=True) + Dropout(0.2)
        ├─ LSTM(128) + Dropout(0.2)
        │
        ├─ Dense(64, ReLU)
        └─ Dense(n_classes, Softmax)

    Parameters
    ----------
    n_timesteps  : number of bi-weekly time steps (default 26 for 7-month season)
    n_features   : spectral + phenological features per timestep (default 13)
    n_classes    : number of land cover classes (default 5)
    conv_filters : CNN filter counts per layer (default (64, 128))
    lstm_units   : LSTM hidden units per layer (default 128)
    dense_units  : dense layer units before classification head (default 64)
    dropout_conv : dropout rate after CNN layers (default 0.3)
    dropout_lstm : dropout rate after LSTM layers (default 0.2)
    kernel_size  : 1D convolution kernel size (default 3)

    Returns
    -------
    Uncompiled Keras Model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, Input
    except ImportError:
        raise ImportError(
            "TensorFlow is required for CNN-LSTM. "
            "Install with: pip install tensorflow>=2.10"
        )

    tf.random.set_seed(42)
    inputs = Input(shape=(n_timesteps, n_features), name="temporal_input")

    # ── CNN block ─────────────────────────────────────────────────────────────
    x = layers.Conv1D(conv_filters[0], kernel_size=kernel_size, padding="same",
                      activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout_conv, name="drop1")(x)

    x = layers.Conv1D(conv_filters[1], kernel_size=kernel_size, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(dropout_conv, name="drop2")(x)

    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

    # ── LSTM block ────────────────────────────────────────────────────────────
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm1")(x)
    x = layers.Dropout(dropout_lstm, name="drop_lstm1")(x)

    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm2")(x)
    x = layers.Dropout(dropout_lstm, name="drop_lstm2")(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_PoppyClassifier")
    return model


class CNNLSTMClassifier:
    """
    Wrapper class for training, evaluating, and deploying the CNN-LSTM classifier.

    Parameters
    ----------
    n_timesteps  : int — number of bi-weekly time steps
    n_features   : int — features per time step
    n_classes    : int — number of land cover classes
    learning_rate: float — Adam learning rate (default 0.0001)
    batch_size   : int — training batch size (default 256)
    max_epochs   : int — maximum training epochs (default 100)
    patience     : int — early stopping patience (default 20)
    model_path   : str or Path — where to save/load model weights
    """

    def __init__(
        self,
        n_timesteps: int = 26,
        n_features: int = 13,
        n_classes: int = N_CLASSES,
        learning_rate: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 20,
        model_path: Optional[str] = None,
        dropout_conv: float = 0.3,
        dropout_lstm: float = 0.2,
    ):
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.model_path = Path(model_path) if model_path else Path("results/models/cnn_lstm.h5")
        self.dropout_conv = dropout_conv
        self.dropout_lstm = dropout_lstm
        self.model = None
        self.history = None

    def _compute_class_weights(self, y: np.ndarray) -> dict:
        """
        Compute inverse-frequency class weights to address class imbalance.
        Opium poppy (class 0) is severely underrepresented.
        """
        from collections import Counter
        counts = Counter(y)
        total = len(y)
        weights = {cls: total / (self.n_classes * count)
                   for cls, count in counts.items()}
        logger.info("Class weights: " +
                    ", ".join(f"{CLASS_NAMES[k]}={v:.2f}" for k, v in weights.items()))
        return weights

    def build(self):
        """Build and compile the model."""
        import tensorflow as tf

        self.model = build_cnn_lstm(
            n_timesteps=self.n_timesteps,
            n_features=self.n_features,
            n_classes=self.n_classes,
            dropout_conv=self.dropout_conv,
            dropout_lstm=self.dropout_lstm,
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(f"Model built: {self.model.count_params():,} parameters.")
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        validation_split: float = 0.15,
    ) -> "CNNLSTMClassifier":
        """
        Train the CNN-LSTM model.

        Parameters
        ----------
        X_train : shape (n_samples, n_timesteps, n_features)
        y_train : shape (n_samples,) — integer class labels
        X_val   : optional validation features
        y_val   : optional validation labels
        validation_split : fraction of training data for validation if no X_val

        Returns
        -------
        self (fitted)
        """
        import tensorflow as tf

        if self.model is None:
            self.build()

        class_weights = self._compute_class_weights(y_train)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience,
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path),
                monitor="val_loss", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8,
                min_lr=1e-6, verbose=1
            ),
        ]

        val_data = (X_val, y_val) if X_val is not None else None
        val_split = 0.0 if val_data is not None else validation_split

        logger.info(f"Training CNN-LSTM: {X_train.shape[0]:,} samples, "
                    f"batch={self.batch_size}, max_epochs={self.max_epochs}")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            validation_split=val_split,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("✅ Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : shape (n_samples, n_timesteps, n_features)

        Returns
        -------
        Predicted class labels, shape (n_samples,)
        """
        probs = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates.

        Returns
        -------
        shape (n_samples, n_classes)
        """
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)

    def save(self, path: Optional[str] = None):
        """Save model weights to disk."""
        save_path = path or self.model_path
        self.model.save(str(save_path))
        logger.info(f"Model saved: {save_path}")

    def load(self, path: Optional[str] = None):
        """Load pre-trained model weights."""
        import tensorflow as tf
        load_path = path or self.model_path
        self.model = tf.keras.models.load_model(str(load_path))
        logger.info(f"Model loaded: {load_path}")
        return self

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training/validation loss and accuracy curves."""
        import matplotlib.pyplot as plt

        if self.history is None:
            raise ValueError("No training history. Run .fit() first.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("CNN-LSTM Training History — Opium Poppy Classifier",
                     fontsize=14, fontweight="bold")

        # Loss
        axes[0].plot(self.history.history["loss"], label="Train", color="#E74C3C", lw=2)
        axes[0].plot(self.history.history["val_loss"], label="Validation",
                     color="#2ECC71", lw=2, linestyle="--")
        axes[0].set_title("Loss (Sparse Categorical Crossentropy)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(self.history.history["accuracy"], label="Train",
                     color="#3498DB", lw=2)
        axes[1].plot(self.history.history["val_accuracy"], label="Validation",
                     color="#F39C12", lw=2, linestyle="--")
        axes[1].set_title("Classification Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Training curve saved: {save_path}")
        plt.show()
