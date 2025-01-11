from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    Attention,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)

from dvclive import Live  # type: ignore  # noqa: E402
from dvclive.keras import DVCLiveCallback  # type: ignore  # noqa: E402


def calculate_class_weight(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    return dict(enumerate(class_weights))


# load hyperparameters from params.yaml, not sure if this is the best way to do this but it works
params = Path("params.yaml")
with open(params, "r") as file:
    params = yaml.safe_load(file)
PATIENCE = params["train"]["patience"]
BATCH_SIZE = params["train"]["batch_size"]
EPOCHS = params["train"]["epochs"]
LSTM_UNITS = params["train"]["lstm_units"]


def compile_and_fit(model, X_train, y_train, X_val, y_val, pitcher_name):
    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",  # categorical crossentropy is standard for classification tasks with multiple classes and one hot encoded labels
        metrics=[
            "sparse_categorical_accuracy",
        ],
    )

    callbacks = [  # noqa: F841
        # callback to end training early if the model stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        # callback to reduce the learning rate if the model stops improving, helps squeeze a bit more performance out of the model
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_sparse_categorical_accuracy",
            factor=0.5,
            patience=PATIENCE // 4,
            min_lr=1e-6,
        ),
        # callback to log metrics to DVC Live (the ML experiment tracking tool I used)
        DVCLiveCallback(live=Live(f"dvclive/{pitcher_name}_logs")),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),  # validation data to monitor model performance
        epochs=EPOCHS,  # given in params.yaml, hyperparameter that controls how many times the model will see the data
        batch_size=BATCH_SIZE,  # given in params.yaml, hyperparameter that controls how many samples are processed before the model weights are updated
        # callbacks=callbacks,  # list of callbacks to use during training
        # class_weight=calculate_class_weight(y_train),
    )

    return history


def create_model(
    input_shape,
    num_classes,
    lstm_units=LSTM_UNITS,
    dropout_rate=params["train"]["dropout"],
    kernel_regularizer=params["train"]["kernel_regularizer"],
):
    # input layer
    input_seq = Input(shape=input_shape)

    # first LSTM layer with dropout and batch normalization
    x = LSTM(
        lstm_units,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer),
    )(input_seq)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # attention layer
    attention = Attention()([x, x])

    # second LSTM layer with dropout and batch normalization
    x = LSTM(
        lstm_units, kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)
    )(attention)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # dense layer with softmax activation for multi-class classification
    output = Dense(num_classes, activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_seq, outputs=output)
