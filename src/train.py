import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

FEAT_DIR = "data/featurized"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(seq_len: int, num_features: int, num_classes: int) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=(seq_len, num_features)),
            LSTM(16),
            Dense(num_classes, activation="softmax"),
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


if __name__ == "__main__":
    for fname in os.listdir(FEAT_DIR):
        if not fname.endswith(".npz"):
            continue

        pitcher = fname.replace("_data.npz", "")
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

        # One-hot encode targets
        num_classes = len(np.unique(y_train))
        y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)
        y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

        # Compute class-weights
        y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train
        cw = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train_flat), y=y_train_flat
        )
        class_weight = dict(enumerate(cw))

        # Build & train model
        seq_len, num_features = X_train.shape[1], X_train.shape[2]
        model = build_model(seq_len, num_features, num_classes)

        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = model.fit(
            X_train,
            y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=30,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[es],
            verbose=1,
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=1)
        print(f"{pitcher}: Test Acc = {test_acc:.3f}, Test Loss = {test_loss:.3f}")

        # Save model & history
        model.save(os.path.join(MODEL_DIR, f"{pitcher}_lstm_model"))
        np.savez(
            os.path.join(MODEL_DIR, f"{pitcher}_lstm_history.npz"),
            loss=np.array(history.history["loss"]),
            val_loss=np.array(history.history["val_loss"]),
            accuracy=np.array(history.history["accuracy"]),
            val_acc=np.array(history.history["val_accuracy"]),
        )
        print(f"Saved model + history for {pitcher}\n")
