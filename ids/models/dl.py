"""Keras DL‑модели: MLP, 1‑D CNN, BiLSTM (RNN)."""
from __future__ import annotations
from typing import Dict
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_mlp(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(input_dim),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_cnn(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),          
        layers.Conv1D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=3, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def build_bilstm(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(None, input_dim)),     
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def get_dl_models(input_dim: int) -> Dict[str, tf.keras.Model]:
    return {
        "mlp_dl":   build_mlp(input_dim),
        "cnn_1d":   build_cnn(input_dim),
        "bilstm":   build_bilstm(input_dim),
    }
