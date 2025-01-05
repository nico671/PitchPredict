from pathlib import Path

import tensorflow as tf
import yaml
from tensorflow.keras.layers import Attention  # type: ignore  # noqa: E402

from dvclive import Live  # type: ignore  # noqa: E402
from dvclive.keras import DVCLiveCallback  # type: ignore  # noqa: E402

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
        learning_rate=1e-4,
        clipnorm=1.0,
        weight_decay=1e-5,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            "accuracy",
        ],
    )

    callbacks = [
        # # early stopping callback to stop training when the model is not improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=PATIENCE // 2,
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
    input_seq = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True)(input_seq)
    attention = Attention()([x, x])
    x = tf.keras.layers.LSTM(LSTM_UNITS)(attention)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=input_seq, outputs=output)
    return model
