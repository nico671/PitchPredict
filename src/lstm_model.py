from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from dvclive import Live
from dvclive.keras import DVCLiveCallback

params = Path("params.yaml")
with open(params, "r") as file:
    params = yaml.safe_load(file)
DROPOUT = params["train"]["dropout"]
PATIENCE = params["train"]["patience"]
KERN_REG = params["train"]["kernel_regularizer"]
BATCH_NORMALIZATION = params["train"]["batch_normalization"]
BATCH_SIZE = params["train"]["batch_size"]
EPOCHS = params["train"]["epochs"]
LSTM_UNITS = params["train"]["lstm_units"]


def compile_and_fit(model, X_train, y_train, X_val, y_val, pitcher_name):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,  # - learning_rate: controls how much to change the model in response to the estimated error each time the model weights are updated.
        clipnorm=1.0,  # - clipnorm: clips gradients by norm; helps prevent exploding gradients.
        weight_decay=1e-4,  # - weight_decay: adds a penalty to the loss function to prevent overfitting.
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=PATIENCE,
            min_lr=1e-6,
        ),
        DVCLiveCallback(live=Live(f"dvclive/{pitcher_name}_logs")),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    return history


def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=input_shape))
    model.add(
        tf.keras.layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
            dropout=DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
        )
    )
    if BATCH_NORMALIZATION:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
            dropout=DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
        )
    )
    if BATCH_NORMALIZATION:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.LSTM(
            LSTM_UNITS,
            dropout=DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
        )
    )
    if BATCH_NORMALIZATION:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return model


def calculate_class_weights(y):
    proportions = np.bincount(y) / len(y)
    for i, proportion in enumerate(proportions):
        if proportion == 0:
            proportions[i] = 1e-6
    inverseN = 1 / len(proportions)
    weights = [inverseN / proportion for proportion in proportions]
    return {i: w for i, w in enumerate(weights)}
