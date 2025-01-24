import logging
import sys
import time
from pathlib import Path

import polars as pl
import yaml

from utils.featurize_utils import (
    add_batting_stats,
    create_base_state_feature,
    create_consistency_feature,
    create_count_feature,
    create_lookback_features,
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
    # create base state feature
    df_list = []
    for pitcher_df in df.group_by("pitcher"):
        # get the pitcher dataframe
        _, pitcher_df = pitcher_df
        logger.info(f"Pitcher: {pitcher_df.select(pl.first('player_name')).item()}")
        logger.info(f"Starting with {pitcher_df.height} pitches")
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
        pitcher_df = create_run_diff_feature(pitcher_df)
        pitcher_df = create_lookback_features(pitcher_df)
        pitcher_df = create_count_feature(pitcher_df)
        pitcher_df = create_base_state_feature(pitcher_df)

        pitcher_df, passed = add_batting_stats(
            pitcher_df, params["clean"]["start_year"]
        )
        if not passed:
            logger.error(
                f"Pitcher {pitcher_df.select(pl.first('player_name')).item()} did not pass the batting stats check"
            )
            continue
        pitcher_df = encode_categorical_features(pitcher_df)

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
        pitcher_df = pitcher_df.drop(["year"])
        pitcher_df = create_consistency_feature(pitcher_df)
        # create target variable

        pitcher_df = create_target(pitcher_df)

        logger.info(f"Ending with {pitcher_df.height} pitches")
        if "BA" in pitcher_df.columns:
            df_list.append(pitcher_df)

    df = pl.concat(df_list)
    logger.info(f"{len(df_list)} pitchers made it through the featurization")
    output_df = handle_missing_values(df)
    # Ensure output directory exists
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write to Parquet file
    output_df.write_parquet(output_dir / "2015_2024_statcast_train.parquet")
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
