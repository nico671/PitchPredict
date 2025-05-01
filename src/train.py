import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

FEAT_DIR = "data/featurized"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        # Clip to prevent numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma) * y_true

        return tf.reduce_sum(alpha * weight * cross_entropy, axis=-1)

    return focal_loss_fn


def build_simple_model(seq_len, num_features, num_classes):
    """Build a simple LSTM model for pitch prediction."""
    inputs = Input(shape=(seq_len, num_features))
    x = LSTM(512, return_sequences=False)(inputs)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=focal_loss(gamma=2.0, alpha=0.25),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    for fname in os.listdir(FEAT_DIR):
        if not fname.endswith(".npz"):
            continue

        player_name = fname.replace("_data.npz", "")
        data = np.load(os.path.join(FEAT_DIR, fname), allow_pickle=True)

        # Extract data
        X_train = np.array(data["X_train"], dtype=np.float32)
        y_train = np.array(data["y_train"], dtype=np.int32)
        X_val = np.array(data["X_val"], dtype=np.float32)
        y_val = np.array(data["y_val"], dtype=np.int32)
        X_test = np.array(data["X_test"], dtype=np.float32)
        y_test = np.array(data["y_test"], dtype=np.int32)

        # Basic scaling
        scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, X_train.shape[2])
        X_train_2d = scaler.fit_transform(X_train_2d)
        X_train = X_train_2d.reshape(X_train.shape)

        X_val_2d = X_val.reshape(-1, X_val.shape[2])
        X_val = scaler.transform(X_val_2d).reshape(X_val.shape)
        X_test_2d = X_test.reshape(-1, X_test.shape[2])
        X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

        # Get unique classes and create mapping
        all_y = np.concatenate([y_train.flatten(), y_val.flatten(), y_test.flatten()])
        unique_classes = np.sort(np.unique(all_y))
        num_classes = len(unique_classes)

        class_mapping = {original: i for i, original in enumerate(unique_classes)}
        inverse_mapping = {i: original for original, i in class_mapping.items()}

        # Remap classes
        y_train_remapped = np.array([class_mapping[y] for y in y_train.flatten()])
        y_val_remapped = np.array([class_mapping[y] for y in y_val.flatten()])
        y_test_remapped = np.array([class_mapping[y] for y in y_test.flatten()])

        # One-hot encode
        y_train_oh = tf.keras.utils.to_categorical(y_train_remapped, num_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val_remapped, num_classes)
        y_test_oh = tf.keras.utils.to_categorical(y_test_remapped, num_classes)

        # Build simple model
        seq_len, num_features = X_train.shape[1], X_train.shape[2]
        model = build_simple_model(seq_len, num_features, num_classes)

        # Callbacks with increased patience
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        ]
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_remapped),
            y=y_train_remapped,
        )

        # Convert to dictionary for Keras
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Apply weights during training
        history = model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )
        # Train model
        history = model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=1)
        print(f"{player_name}: Test Acc = {test_acc:.3f}, Test Loss = {test_loss:.3f}")

        # Save model
        model.save(os.path.join(MODEL_DIR, f"{player_name}_lstm_model.keras"))
        np.savez(
            os.path.join(MODEL_DIR, f"{player_name}_lstm_history.npz"),
            loss=np.array(history.history["loss"]),
            val_loss=np.array(history.history["val_loss"]),
            accuracy=np.array(history.history["accuracy"]),
            val_accuracy=np.array(history.history["val_accuracy"]),
            classes=unique_classes,
            class_mapping=class_mapping,
            inverse_mapping=inverse_mapping,
        )
        print(f"Saved improved model for {player_name}\n")
