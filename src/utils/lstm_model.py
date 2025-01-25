from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.utils import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (  # type: ignore
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.models import Model  # type: ignore

from dvclive import Live
from dvclive.keras import DVCLiveCallback

tf.random.set_seed(1)
np.random.seed(1)


def get_sample_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))

    return class_weight_dict

    sample_weights = compute_sample_weight(class_weight_dict, y_train)
    return sample_weights


# load hyperparameters from params.yaml, not sure if this is the best way to do this but it works
params = Path("params.yaml")
with open(params, "r") as file:
    params = yaml.safe_load(file)
PATIENCE = params["train"]["patience"]
BATCH_SIZE = params["train"]["batch_size"]
EPOCHS = params["train"]["epochs"]
LSTM_UNITS = params["train"]["lstm_units"]


def compile_and_fit(model, X_train, y_train, X_val, y_val, pitcher_name):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE // 4,
            min_lr=1e-6,
            mode="min",
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
        # class_weight=get_sample_weights(y_train),
    )

    return history


def create_model(input_shape, num_classes, lstm_units=LSTM_UNITS):
    inputs = Input(shape=input_shape)

    x = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )
    )(inputs)
    # for i in range(3):
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.1)(x)
    #     x = Bidirectional(
    #         LSTM(
    #             lstm_units,
    #             return_sequences=True,
    #             kernel_regularizer=tf.keras.regularizers.l2(0.01),
    #         )
    #     )(x)

    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(
        LSTM(
            lstm_units,
            # return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )
    )(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)
