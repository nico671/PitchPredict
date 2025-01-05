from pathlib import Path

import tensorflow as tf
import yaml
from tensorflow.keras.layers import Attention  # type: ignore  # noqa: E402

from dvclive import Live  # type: ignore  # noqa: E402
from dvclive.keras import DVCLiveCallback  # type: ignore  # noqa: E402

# load hyperparameters from params.yaml, not sure if this is the best way to do this but it works
params = Path("params.yaml")
with open(params, "r") as file:
    params = yaml.safe_load(file)
PATIENCE = params["train"]["patience"]
BATCH_SIZE = params["train"]["batch_size"]
EPOCHS = params["train"]["epochs"]
LSTM_UNITS = params["train"]["lstm_units"]


def compile_and_fit(model, X_train, y_train, X_val, y_val, pitcher_name):
    # although i found some resources saying that SGD is best for pure model performance, Adam gives excellent results and requires far less tuning
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,  # hyperparameter controlling how much we are adjusting the weights of our network with respect the loss gradient
        clipnorm=1.0,  # hyperparameter that controls the maximum norm of the gradients
        weight_decay=1e-5,  # hyperparameter that adds a penalty to the loss function to prevent overfitting
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1  # hyperparameter that smooths the labels (this really helped with overfitting and generalization)
        ),  # categorical crossentropy is standard for classification tasks with multiple classes and one hot encoded labels
        metrics=[
            "accuracy",
        ],
    )

    callbacks = [
        # callback to end training early if the model stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
        # callback to reduce the learning rate if the model stops improving, helps squeeze a bit more performance out of the model
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=PATIENCE // 2,
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
        callbacks=callbacks,  # list of callbacks to use during training
    )

    return history


# LSTM model with attention layer
def create_model(input_shape, num_classes):
    # input layer
    input_seq = tf.keras.layers.Input(shape=input_shape)
    # first LSTM layer
    x = tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=True)(input_seq)
    # attention layer, this is the key to the model's performance as it allows the model to focus on the most important parts of the input sequence
    attention = Attention()([x, x])
    # second LSTM layer
    x = tf.keras.layers.LSTM(LSTM_UNITS)(attention)
    # dense layer with softmax activation for multi-class classification
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_seq, outputs=output)
