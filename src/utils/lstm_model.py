from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.utils import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    LSTM,
    Attention,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

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

    # Normalize weights to sum to 1
    total_weight = sum(class_weight_dict.values())
    class_weight_dict = {k: v / total_weight for k, v in class_weight_dict.items()}
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
    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE // 4,
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


def create_model(
    input_shape,
    num_classes,
    lstm_units=16,
    dropout_rate=0.1,
    kernel_regularizer=0.01,
):
    inputs = Input(shape=input_shape)

    # First LSTM layer with return_sequences=True
    x = LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        kernel_regularizer=l2(kernel_regularizer),
    )(inputs)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # Attention layer
    x = Attention()([x, x])

    x = LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        kernel_regularizer=l2(kernel_regularizer),
    )(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # Attention layer
    x = Attention()([x, x])

    x = LSTM(
        lstm_units,
        return_sequences=False,  # Changed to False
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        kernel_regularizer=l2(kernel_regularizer),
    )(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # Main output
    main_output = Dense(num_classes, activation="softmax", name="main_output")(x)

    model = Model(inputs=inputs, outputs=main_output)
    return model
