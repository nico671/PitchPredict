from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.preprocessing import MinMaxScaler

from utils.featurize_utils import sort_by_time

params_path = Path("params.yaml")
with open(params_path, "r") as file:
    params = yaml.safe_load(file)


def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i : (i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def create_training_data(pitcher_df, features):
    pitcher_df = sort_by_time(pitcher_df)

    # select features and target variable for initial split of X and y
    X = pitcher_df.select(pl.col(features)).to_numpy()
    y = pitcher_df.select(pl.col("next_pitch")).to_numpy().ravel()

    # declare time steps
    time_steps = params["train"]["time_steps"]

    X_seq, y_seq = create_sequences(X, y, time_steps)
    # convert to correct datatypes for model
    X_seq = X_seq.astype("float32")
    y_seq = y_seq.astype("int32")  # Ensure labels are integers
    num_classes = np.max(y_seq) + 1
    # split into train, test, val
    train_split = params["train"]["train_split"]
    train_size = int(len(X_seq) * train_split)
    val_size = int(len(X_seq) * (1 - train_split) / 2)
    X_train, X_val, X_test = np.split(X_seq, [train_size, train_size + val_size])
    y_train, y_val, y_test = np.split(y_seq, [train_size, train_size + val_size])

    scaler = MinMaxScaler()
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(
        X_test.shape
    )

    return (
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        X_test_scaled,
        y_test,
        num_classes,
    )
