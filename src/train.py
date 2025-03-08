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

from utils.featurize_utils import sort_by_time
from utils.lstm_model import compile_and_fit, create_model
from utils.train_utils import create_training_data

# load parameters
params_path = Path("params.yaml")
with open(params_path, "r") as file:
    params = yaml.safe_load(file)

# set up logging (not really necessary as this won't go to prod but good practice)
logger = logging.getLogger("choo choo")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def training_loop(df, params):
    pitcher_data = {}
    count = 0

    # initialize features to be a list of all the columns in the dataframe
    features = df.columns

    # remove columns that are not features, either target or they have no value for prediction, they were just used in feature engineering
    for feature in [
        "next_pitch",
        "pitcher",
        "pitcher_name",
        # "pitch_type",
        "game_date",
        # "game_pk",
        # "at_bat_number",
        "type",
        # "pitch_number",
    ]:
        features.remove(feature)
    logger.info(f"{len(features)} features: {features}")

    # loop through each pitcher and train a model, uses the polars groupby function to group by pitcher and then iterate through each pitcher
    for pitcher_df in df.group_by("pitcher"):
        # get the pitcher code (statcast id)
        pitcher_code, pitcher_df = pitcher_df

        # sort the dataframe by game date, game pk, inning, at bat number, and pitch number, ensuring the data is in time series order
        pitcher_df = sort_by_time(pitcher_df)

        # get the pitcher name and number of pitches
        pitcher_name = pitcher_df.select(pl.first("pitcher_name")).item()
        num_pitches = pitcher_df.height

        logger.info(
            f"Training model for pitcher: {pitcher_name} - {num_pitches} pitches"
        )

        # split the data into training, validation, and test sets using the create_training_data function
        X_train, y_train, X_val, y_val, X_test, y_test, num_classes = (
            create_training_data(pitcher_df, features)
        )

        # logger.info("\nLabel Distribution:")
        # for i in range(y_train.shape[1]):
        #     logger.info(f"Class {i}: {y_train[:,i].sum() / len(y_train):.2%}")

        # create and train the model using utility functions
        lstm_model = create_model(X_train.shape[1:], num_classes)
        history = compile_and_fit(
            lstm_model,
            X_train,
            y_train,
            X_val,
            y_val,
            pitcher_name,
        )

        # get the most common pitch rate for the pitcher
        most_common_pitch_rate = pitcher_df.select(
            pl.col("pitch_type").value_counts(normalize=True, sort=True).head(1)
        ).item()["proportion"]

        logger.info(
            f"Most common pitch rate for {pitcher_name}: {most_common_pitch_rate}"
        )

        # evaluate the model on the test set
        test_loss, test_accuracy, *_ = lstm_model.evaluate(X_test, y_test)

        # store the model and metrics in the pitcher_data dictionary
        pitcher_data[pitcher_code] = {
            "model": lstm_model,
            "history": history,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "total_pitches": num_pitches,
            "unique_classes": len(
                np.unique(
                    np.concatenate(
                        [
                            y_train,
                            y_val,
                            y_test,
                        ]
                    )
                )
            ),
            "player_name": pitcher_name,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "features": features,
            "most_common_pitch_rate": most_common_pitch_rate,
            "performance_gain": (test_accuracy - most_common_pitch_rate) * 100,
        }

        # print a buncha stats so that i have something to look at while it runs
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")
        logger.info(
            f" Accuracy Gained over guessing most common pitch for {pitcher_name}: {pitcher_data[pitcher_code]['performance_gain']:.2f}%",
        )
        logger.info(
            f"Average Test Accuracy: {mean([pitcher_data[pitcher]['test_accuracy'] for pitcher in pitcher_data]) * 100:.2f}%"
        )

        logger.info(
            f"Average Performance Gained over guessing most common pitch: {mean([pitcher_data[pitcher]['performance_gain'] for pitcher in pitcher_data]):.2f}%"
        )
        count += 1
        logger.info(
            f"{count} of {len(df.select(pl.col('pitcher')).unique())}, {(count / len(df.select(pl.col('pitcher')).unique())) * 100:.2f}% done!"
        )

    return pitcher_data


def main():
    start_time = time.time()
    # input error checking
    if len(sys.argv) != 1:
        logger.error("Arguments error. Usage:\n")
        logger.error("not enough inputs, expected input structure is: *.py")
        sys.exit(1)

    # read in data
    df = pl.read_parquet(Path(params["train"]["input_data_path"]))
    # call training loop function to train models, pitcher and model metrics are stored in pitcher_data (see definition above)
    pitcher_data = training_loop(df, params)

    output_dir = Path("data/evaluate/")
    if os.path.isfile(output_dir / "pitcher_data.pickle"):
        os.remove(output_dir / "pitcher_data.pickle")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "pitcher_data.pickle", "wb") as f:
        pickle.dump(pitcher_data, f)
    end_time = time.time()
    logger.info(f"Training took {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
