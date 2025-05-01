import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input

FEAT_DIR = "data/featurized"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(
    seq_len: int, num_features: int, num_classes: int, config=None
) -> tf.keras.Model:
    """Build a model for pitch prediction with configurable architecture.

    Args:
        seq_len: Length of input sequences
        num_features: Number of features per time step
        num_classes: Number of pitch classes to predict
        config: Dictionary with model configuration parameters
    """
    # Default configuration
    if config is None:
        config = {
            "lstm_units": [128, 64],  # Multiple LSTM layers with specified units
            "dropout_rate": 0.3,  # Dropout for regularization
            "recurrent_dropout": 0.2,  # Recurrent dropout for LSTM
            "use_bidirectional": True,  # Use bidirectional wrappers
            "dense_units": [32],  # Additional dense layers before output
            "learning_rate": 1e-4,
            "clipnorm": 1.0,
        }

    inputs = Input(shape=(seq_len, num_features))
    x = inputs

    # Add LSTM layers
    for i, units in enumerate(config["lstm_units"]):
        return_sequences = (
            i < len(config["lstm_units"]) - 1
        )  # Return sequences except for last layer

        lstm_layer = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=config["dropout_rate"],
            recurrent_dropout=config["recurrent_dropout"],
        )

        # Optionally wrap in bidirectional
        if config["use_bidirectional"]:
            lstm_layer = tf.keras.layers.Bidirectional(lstm_layer)

        x = lstm_layer(x)

        # Add batch normalization after each LSTM
        x = tf.keras.layers.BatchNormalization()(x)

    # Add dense layers before final classification
    for units in config["dense_units"]:
        x = Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(config["dropout_rate"])(x)

    # Final output layer
    outputs = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    opt = tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"], clipnorm=config["clipnorm"]
    )

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


if __name__ == "__main__":
    # Define model configurations to try
    model_configs = [
        {
            "name": "baseline",
            "config": {
                "lstm_units": [128],
                "dropout_rate": 0.3,
                "recurrent_dropout": 0.2,
                "use_bidirectional": False,
                "dense_units": [],
                "learning_rate": 1e-4,
                "clipnorm": 1.0,
            },
        },
        {
            "name": "bidirectional",
            "config": {
                "lstm_units": [128],
                "dropout_rate": 0.3,
                "recurrent_dropout": 0.2,
                "use_bidirectional": True,
                "dense_units": [],
                "learning_rate": 1e-4,
                "clipnorm": 1.0,
            },
        },
        {
            "name": "deep",
            "config": {
                "lstm_units": [128, 64],
                "dropout_rate": 0.3,
                "recurrent_dropout": 0.2,
                "use_bidirectional": True,
                "dense_units": [32],
                "learning_rate": 1e-4,
                "clipnorm": 1.0,
            },
        },
    ]

    # Choose which config to use (could be a command line arg)
    selected_model = "deep"  # Change this to try different architectures
    model_config = next(
        item["config"] for item in model_configs if item["name"] == selected_model
    )

    for fname in os.listdir(FEAT_DIR):
        if not fname.endswith(".npz"):
            continue

        player_name = fname.replace("_data.npz", "")
        data = np.load(os.path.join(FEAT_DIR, fname), allow_pickle=True)

        # Extract and convert data to ensure proper numeric types
        X_train = np.array(data["X_train"], dtype=np.float32)
        y_train = np.array(data["y_train"], dtype=np.int32)
        X_val = np.array(data["X_val"], dtype=np.float32)
        y_val = np.array(data["y_val"], dtype=np.int32)
        X_test = np.array(data["X_test"], dtype=np.float32)
        y_test = np.array(data["y_test"], dtype=np.int32)

        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

        # Get the unique pitch classes from all sets to ensure no missing classes
        all_y = np.concatenate([y_train.flatten(), y_val.flatten(), y_test.flatten()])
        unique_classes = np.sort(np.unique(all_y))
        train_classes = np.sort(np.unique(y_train))

        # Check for classes in validation/test but not in training
        missing_in_train = set(unique_classes) - set(train_classes)
        if missing_in_train:
            print(
                f"WARNING: Classes {missing_in_train} appear in val/test but not in training"
            )

        num_classes = len(unique_classes)
        print(
            f"Player {player_name} has {num_classes} unique pitch classes: {unique_classes}"
        )

        # Create a mapping from original class indices to consecutive integers
        class_mapping = {original: i for i, original in enumerate(unique_classes)}
        inverse_mapping = {i: original for original, i in class_mapping.items()}

        # Remap the class labels - flatten first to handle any shape
        y_train_remapped = np.array([class_mapping[y] for y in y_train.flatten()])
        y_val_remapped = np.array([class_mapping[y] for y in y_val.flatten()])
        y_test_remapped = np.array([class_mapping[y] for y in y_test.flatten()])

        # Now our classes will be consecutive integers from 0 to num_classes-1
        print(f"Remapped classes: {np.unique(y_train_remapped)}")

        # One-hot encode targets with remapped classes
        y_train_oh = tf.keras.utils.to_categorical(y_train_remapped, num_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val_remapped, num_classes)
        y_test_oh = tf.keras.utils.to_categorical(y_test_remapped, num_classes)

        # Compute class-weights using only train set classes
        # It's reasonable to only compute weights for classes we actually have in training
        train_class_indices = [class_mapping[cls] for cls in train_classes]
        cw = np.ones(num_classes)  # Default weight 1 for all classes

        if len(train_class_indices) > 1:  # Only compute if we have multiple classes
            train_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train_remapped),
                y=y_train_remapped,
            )
            for i, weight in zip(np.unique(y_train_remapped), train_weights):
                cw[i] = weight

        class_weight = dict(enumerate(cw))

        # Build & train model with selected config
        seq_len, num_features = X_train.shape[1], X_train.shape[2]
        model = build_model(seq_len, num_features, num_classes, model_config)

        # Add more callbacks for better training
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f"logs/{player_name}_{selected_model}", histogram_freq=1
            ),
        ]

        history = model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=50,  # Increase epochs since we have early stopping
            batch_size=32,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=1)
        print(f"{player_name}: Test Acc = {test_acc:.3f}, Test Loss = {test_loss:.3f}")

        # Get per-class performance metrics
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test_oh, axis=1)

        print(f"\nDetailed classification report for {player_name}:")
        print(classification_report(y_test_classes, y_pred_classes))

        # Save model with architecture identifier
        model.save(
            os.path.join(MODEL_DIR, f"{player_name}_{selected_model}_lstm_model.keras")
        )
        np.savez(
            os.path.join(MODEL_DIR, f"{player_name}_{selected_model}_lstm_history.npz"),
            loss=np.array(history.history["loss"]),
            val_loss=np.array(history.history["val_loss"]),
            accuracy=np.array(history.history["accuracy"]),
            val_acc=np.array(history.history["val_accuracy"]),
            classes=unique_classes,
            class_mapping=class_mapping,
            inverse_mapping=inverse_mapping,
            model_config=model_config,  # Save the model configuration
        )
        print(
            f"Saved model + history for {player_name} with {selected_model} architecture\n"
        )
