from pathlib import Path

import numpy as np
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
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Attention


def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Round predictions to 0 or 1
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)  # True Positives
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)  # False Positives
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)  # False Negatives

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)  # Return the mean F1 score across all classes


def calculate_balanced_weights(y_train, min_weight=0.01, epsilon=1e-6):
    # Convert one-hot to integer labels
    y_train_int = np.argmax(y_train, axis=1)

    # Get class counts with minimum 1 to avoid division by zero
    class_counts = np.maximum(np.bincount(y_train_int), epsilon)

    # Calculate inverse frequencies with numerical stability
    weights = 1.0 / (class_counts + epsilon)

    # Ensure minimum weight
    weights = np.maximum(weights, min_weight)

    # Normalize to sum to 1 using stable division
    weights = weights / (weights.sum() + epsilon)

    return dict(enumerate(weights))


def compile_and_fit(model, X_train, y_train, X_val, y_val, pitcher_name):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,  # Start higher
        clipnorm=1.0,
        weight_decay=1e-5,
        # momentum=0.8,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", f1_score],
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
        # DVCLiveCallback(live=Live(f"dvclive/{pitcher_name}_logs")),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        # class_weight=calculate_balanced_weights(y_train),
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


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.InputLayer(shape=input_shape))
# model.add(
#     tf.keras.layers.LSTM(
#         LSTM_UNITS,
#         return_sequences=True,
#         dropout=DROPOUT,
#         kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
#     )
# )
# if BATCH_NORMALIZATION:
#     model.add(tf.keras.layers.BatchNormalization())
# model.add(
#     tf.keras.layers.LSTM(
#         LSTM_UNITS,
#         return_sequences=True,
#         dropout=DROPOUT,
#         kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
#     )
# )
# if BATCH_NORMALIZATION:
#     model.add(tf.keras.layers.BatchNormalization())
# model.add(
#     tf.keras.layers.LSTM(
#         LSTM_UNITS,
#         dropout=DROPOUT,
#         return_sequences=True,
#         kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
#     )
# )
# if BATCH_NORMALIZATION:
#     model.add(tf.keras.layers.BatchNormalization())
# model.add(
#     tf.keras.layers.LSTM(
#         LSTM_UNITS,
#         dropout=DROPOUT,
#         return_sequences=True,
#         kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
#     )
# )
# if BATCH_NORMALIZATION:
#     model.add(tf.keras.layers.BatchNormalization())
# model.add(
#     tf.keras.layers.LSTM(
#         LSTM_UNITS,
#         dropout=DROPOUT,
#         kernel_regularizer=tf.keras.regularizers.l2(KERN_REG),
#     )
# )
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
# return model
