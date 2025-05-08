"""Keras DL models (MLP, 1‑D CNN, BiLSTM)."""
from __future__ import annotations

from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_mlp(input_dim: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Dense(128, activation="relu", input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(5, activation="softmax"),
    ])
    model.compile(optimizer=optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def get_dl_models(input_dim: int) -> Dict[str, tf.keras.Model]:
    return {"mlp": build_mlp(input_dim)}