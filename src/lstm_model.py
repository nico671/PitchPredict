from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

params = Path("params.yaml")
with open(params, "r") as file:
    params = yaml.safe_load(file)
DROPOUT = params["train"]["dropout"]
PATIENCE = params["train"]["patience"]


def compile_and_fit(model, X_train, y_train, X_val, y_val, class_weight):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4, clipnorm=1.0, weight_decay=1e-4
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
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # early stopping typically occurs around 20 anyways
        batch_size=16,
        callbacks=callbacks,
        # class_weight=class_weight, # hurts performance
    )

    return history


def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=input_shape))
    model.add(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.LSTM(
            64,
            dropout=DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )
    )
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
