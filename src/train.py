import logging
import os
import pickle
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import polars as pl
import yaml
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from lstm_model import compile_and_fit, create_model

params_path = Path("params.yaml")
with open(params_path, "r") as file:
    params = yaml.safe_load(file)

logger = logging.getLogger("choo choo")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i : (i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def create_training_data(pitcher_df, features):
    # sort the data by game_date, game_pk, at_bat_number, pitch_number
    pitcher_df = pitcher_df.sort(
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ],
        descending=False,
    )

    # select features and target variable for initial split of X and y
    X = pitcher_df.select(pl.col(features)).to_numpy()
    y = pitcher_df.select(pl.col("next_pitch")).to_numpy().ravel()

    # encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y = y_encoded

    # declare time steps
    time_steps = params["train"]["time_steps"]

    X_seq, y_seq = create_sequences(X, y, time_steps)

    # convert to correct datatypes for model
    X_seq = X_seq.astype("float32")
    y_seq = y_seq.astype("int32")  # Ensure labels are integers

    # split into train, test, val
    train_split = params["train"]["train_split"]
    train_size = int(len(X_seq) * train_split)
    val_size = int(len(X_seq) * (1 - train_split) / 2)
    X_train, X_val, X_test = np.split(X_seq, [train_size, train_size + val_size])
    y_train, y_val, y_test = np.split(y_seq, [train_size, train_size + val_size])

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

    # hidden until i figure out how to handle this problem
    # logger.info(f"Unique classes: {len(np.unique(y))}")
    # logger.info(f"Classes missing from test: {len(np.setdiff1d(y, y_test))}")
    # logger.info(f"Classes missing from train: {len(np.setdiff1d(y, y_train))}")
    # logger.info(f"Classes missing from val: {len(np.setdiff1d(y, y_val))}")

    return (
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        label_encoder,
    )


def training_loop(df, params):
    pitcher_data = {}
    count = 0
    features = df.columns
    for feature in [
        "next_pitch",
        "pitcher",
        "player_name",
        "game_date",
        "game_pk",
        "pitch_type",
        "type",
    ]:
        features.remove(feature)
    start_time = time.time()
    for pitcher_df in df.group_by("pitcher"):
        pitcher_code = pitcher_df[0]
        pitcher_df = pitcher_df[1]

        pitcher_name = pitcher_df.select(pl.first("player_name")).item()
        num_pitches = len(pitcher_df)
        logger.info(
            f"Training model for pitcher: {pitcher_name} - {num_pitches} pitches"
        )
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = (
            create_training_data(pitcher_df, features)
        )

        lstm_model = create_model(X_train.shape[1:], len(label_encoder.classes_))
        history = compile_and_fit(
            lstm_model,
            X_train,
            y_train,
            X_val,
            y_val,
            pitcher_name,
        )

        most_common_pitch_rate = pitcher_df.select(
            pl.col("pitch_type").value_counts().sort(descending=False).head(1)
        ).item()["count"] / len(pitcher_df)

        print(f"Most common pitch rate for {pitcher_name}: {most_common_pitch_rate}")
        test_loss, test_accuracy = lstm_model.evaluate(X_test, y_test)
        pitcher_data[pitcher_code] = {
            "model": lstm_model,
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "total_pitches": len(y_test) + len(y_train) + len(y_val),
            "unique_classes": len(np.unique(y_train)),
            "player_name": pitcher_name,
            "X_test": X_test,
            "y_test": y_test,
            "label_encoder": label_encoder,
            "most_common_pitch_rate": most_common_pitch_rate,
            "performance_gain": (test_accuracy - most_common_pitch_rate) * 100,
        }
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")
        logger.info(
            f" Accuracy Gained over guessing most common pitch for {pitcher_name}: {pitcher_data[pitcher_code]["performance_gain"]:.2f}%",
        )
        logger.info(
            f"Average Test Accuracy: {mean([pitcher_data[pitcher]['test_accuracy'] for pitcher in pitcher_data])*100:.2f}%"
        )

        logger.info(
            f"Average Performance Gained over guessing most common pitch: {mean([pitcher_data[pitcher]['performance_gain'] for pitcher in pitcher_data]):.2f}%"
        )

        count += 1
        logger.info(
            f'{count} of {len(df.select(pl.col("pitcher")).unique())}, {(count/len(df.select(pl.col("pitcher")).unique())) * 100:.2f}% done!'
        )
    end_time = time.time()
    logger.info(f"Training took {end_time - start_time} seconds")
    return pitcher_data


def main():
    if len(sys.argv) != 1:
        logger.error("Arguments error. Usage:\n")
        logger.error("not enough inputs, expected input structure is: *.py")
        sys.exit(1)

    df = pl.read_parquet(Path(params["train"]["input_data_path"]))
    pitcher_data = training_loop(df, params)

    output_dir = Path("data/evaluate/")
    if os.path.isfile(output_dir / "pitcher_data.pickle"):
        os.remove(output_dir / "pitcher_data.pickle")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "pitcher_data.pickle", "wb") as f:
        pickle.dump(pitcher_data, f)
    logger.info("all done")


if __name__ == "__main__":
    main()
