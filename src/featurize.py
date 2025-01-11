import logging
import sys
import time
from pathlib import Path

import polars as pl
import yaml

from utils.featurize_utils import (
    add_batting_stats,
    create_base_state_feature,
    create_count_feature,
    create_run_diff_feature,
    create_target,
    encode_categorical_features,
    handle_missing_values,
)

logger = logging.getLogger("feats of epic proportions")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def main():
    start_time = time.time()
    # check for correct input length
    if len(sys.argv) != 1:
        logger.error("Arguments error. Usage:\n")
        logger.error("not enough inputs, expected input structure is: *.py")
        sys.exit(1)
    # check for correct input file types
    elif ".py" not in sys.argv[0]:
        logger.error(
            "Please enter a valid python source file as the first input for this stage"
        )
        sys.exit(1)

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)
    input_file_path = params["featurize"]["input_data_path"]

    df = pl.read_parquet(Path(input_file_path))

    df = df.sort(
        [
            "game_date",
            "game_pk",
            "inning",
            "at_bat_number",
            "pitch_number",
        ],
        descending=False,
    )

    df = encode_categorical_features(df)

    # create target variable
    df = create_target(df)

    # create count feature
    df = create_count_feature(df)

    # create base state feature
    df = create_base_state_feature(df)

    # create run_diff feature
    df = create_run_diff_feature(df)

    df = df.drop(
        df.select(pl.all().is_null().sum() / df.height)
        .unpivot()
        .filter(pl.col("value") > 0.05)
        .select("variable")
        .to_series()
        .to_list()
    )

    df = add_batting_stats(df, params["clean"]["start_year"])
    # Handle missing values
    df = handle_missing_values(df)

    df_list = []
    for pitcher_df in df.group_by("pitcher"):
        # get the pitcher dataframe
        pitcher_df = pitcher_df[1]

        # sort the dataframe by game date, game pk, inning, at bat number, and pitch number, ensuring the data is in time series order
        pitcher_df = pitcher_df.sort(
            [
                "game_date",
                "game_pk",
                "inning",
                "at_bat_number",
                "pitch_number",
            ],
            descending=False,
        )
        pitch_types = pitcher_df.select("pitch_type").unique()
        pitcher_df = pitcher_df.with_columns(
            pl.col("game_date").dt.year().alias("year").cast(pl.Int32)
        )
        for pitch_type in pitch_types["pitch_type"].to_list():
            # Assuming pitcher_df is your DataFrame and pitch_type is the variable you are comparing
            filtered = pitcher_df.filter(pl.col("pitch_type") == pitch_type)
            years = filtered["year"].unique().to_list()

            if len(years) != (2024 - params["clean"]["start_year"] + 1):
                pitcher_df = pitcher_df.filter(~(pl.col("pitch_type") == pitch_type))
        df_list.append(pitcher_df)
    df = pl.concat(df_list)
    # Create the output DataFrame
    output_df = df
    output_df = output_df.sort(
        [
            "game_date",
            "game_pk",
            "inning",
            "at_bat_number",
            "pitch_number",
        ],
        descending=False,
    )
    # Ensure output directory exists
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write to Parquet file
    output_df.write_parquet(output_dir / "2015_2024_statcast_train.parquet")
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info("done")


if __name__ == "__main__":
    main()
