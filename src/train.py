import logging
import os
import pickle
import sys
from pathlib import Path
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
import numpy as np
from lstm_model import (
    calculate_class_weights,
    compile_and_fit,
    create_model,
)

logger = logging.getLogger("choo choo")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def create_training_data(df, pitcher, features):
    # split data by pitcher
    pitcher_df = df[df["pitcher"] == pitcher].reset_index(drop=True)
    # sort again to ensure sequential order
    pitcher_df = pitcher_df.sort_values(["game_date", "at_bat_number", "pitch_number"])

    # split into x, y
    X = pitcher_df[features].values
    y = pitcher_df["next_pitch"].values

    # encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y = y_encoded

    # declare time steps
    # TODO: make this a parameter
    time_steps = 25

    # sequence creation helper function
    def create_sequences(X, y, time_steps):
        sequences = []
        targets = []
        for i in range(len(X) - time_steps):
            seq = X[
                i : i + time_steps
            ]  # Get the sequence of features (input pitches), size = time_steps
            label = y[i + time_steps]  # Target for the sequence
            sequences.append(seq)
            targets.append(label)
        return np.array(sequences), np.array(targets)

    X_sequences, y_sequences = create_sequences(X, y, time_steps)
    # convert to correct datatypes for model
    X_sequences = X_sequences.astype("float32")
    y_sequences = y_sequences.astype("int32")  # Ensure labels are integers

    # split into train, test, val
    train_size = int(len(X_sequences) * 0.7)
    val_size = int(len(X_sequences) * 0.15)
    X_train, X_val, X_test = np.split(X_sequences, [train_size, train_size + val_size])
    y_train, y_val, y_test = np.split(y_sequences, [train_size, train_size + val_size])
    class_weight = calculate_class_weights(y_train)

    # scale features, no reason for minmax tbh, just read a stackoverflow that recommended it
    scaler = MinMaxScaler()
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(
        X_test.shape
    )

    # TODO: fix that not all labels show up in the splits, this is affecting test accuracy for some pitchers with weird splits
    # logger.info("Unique labels in y_train:", list(np.unique(y_train)))
    # logger.info("Unique labels in y_val:", list(np.unique(y_val)))
    # logger.info("Unique labels in y_test:", list(np.unique(y_test)))

    return (
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        label_encoder,
        class_weight,
    )


def training_loop(df):
    pitcher_data = {}
    count = 0
    logger.info(f"Number of pitchers: {len(df["pitcher"].unique())}")
    features = []
    with open("data/features.txt", "r") as f:
        features = f.read().splitlines()

    for pitcher in df["pitcher"].unique():
        logger.info(
            f"Training model for pitcher: {df[df['pitcher'] == pitcher]['player_name'].iloc[0]}"
        )
        pitcher_df = df[df["pitcher"] == pitcher]
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, class_weight = (
            create_training_data(pitcher_df, pitcher, features)
        )

        # Ensure labels are integers
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)

        lstm_model = create_model(X_train.shape[1:], len(label_encoder.classes_))

        history = compile_and_fit(
            lstm_model, X_train, y_train, X_val, y_val, class_weight
        )

        test_loss, test_accuracy = lstm_model.evaluate(X_test, y_test)
        pitcher_data[pitcher] = {
            "model": lstm_model,
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "total_pitches": len(y_test) + len(y_train) + len(y_val),
            "unique_classes": len(np.unique(y_train)),
            "player_name": df[df["pitcher"] == pitcher]["player_name"].iloc[0],
            "X_test": X_test,
            "y_test": y_test,
            "most_common_pitch_rate": pitcher_df["next_pitch"]
            .value_counts()
            .sort_values(ascending=False)
            .iloc[0]
            / len(pitcher_df),
        }
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        logger.info(
            f"Average Test Accuracy: {mean([pitcher_data[pitcher]['test_accuracy'] for pitcher in pitcher_data])}"
        )
        count += 1
        logger.info(
            f'{count} of {len(df["pitcher"].unique())}, {count/len(df["pitcher"].unique()) * 100:.2f}% done!'
        )
    return pitcher_data


def main():

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)
    file_name = sys.argv[1]

    df = pd.read_parquet(Path(file_name))
    pitcher_data = training_loop(df)

    output_dir = Path("data/evaluate/")
    if os.path.isfile(output_dir / "pitcher_data.pickle"):
        os.remove(output_dir / "pitcher_data.pickle")  # Opt.: os.system("rm "+strFile)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "pitcher_data.pickle", "wb") as f:
        pickle.dump(pitcher_data, f)
    logger.info("all done")


if __name__ == "__main__":
    main()
