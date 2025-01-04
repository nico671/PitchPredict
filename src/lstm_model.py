from pathlib import Path

import tensorflow as tf
import yaml

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
        learning_rate=1e-4,  # Start higher
        clipnorm=1.0,
        weight_decay=1e-5,
        # momentum=0.8,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),  # Use focal loss
        metrics=["accuracy"],
    )

    callbacks = [
        # # early stopping callback to stop training when the model is not improving
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_accuracy",
        #     patience=PATIENCE,
        #     restore_best_weights=True,
        # ),
        # # reduce learning rate on plateau callback to reduce the learning rate when the model is not improving
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_accuracy",
        #     factor=0.7,
        #     patience=2,
        #     min_lr=1e-6,
        # ),
        # metric logging callback to log metrics to dvclive
        # DVCLiveCallback(live=Live(f"dvclive/{pitcher_name}_logs")),
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
    model.add(
        tf.keras.layers.LSTM(
            LSTM_UNITS // 2,
            return_sequences=True,
            dropout=DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
        )
    )
    model.add(
        tf.keras.layers.LSTM(
            LSTM_UNITS // 4,
            dropout=DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
        )
    )
    if BATCH_NORMALIZATION:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return model
