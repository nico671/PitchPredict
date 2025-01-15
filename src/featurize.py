import logging
import sys
import time
from pathlib import Path

import polars as pl
import yaml

from utils.featurize_utils import *  # noqa: F403

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
        class_counts = pitcher_df.select(
            pl.col("pitch_type").value_counts(normalize=True)
        ).unnest("pitch_type")
        rare_pitches = class_counts.filter(pl.col("proportion") < 0.03)
        rare_pitches = rare_pitches.select(pl.col("pitch_type").unique())[
            "pitch_type"
        ].to_list()
        pitcher_df = pitcher_df.with_columns(
            pl.col("pitch_type").replace(rare_pitches, "Other").alias("pitch_type")
        )
        pitcher_df = encode_categorical_features(pitcher_df)

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

        pitcher_df = add_batting_stats(pitcher_df, params["clean"]["start_year"])
        pitcher_df = create_consistency_feature(pitcher_df)
        pitcher_df = create_target(pitcher_df)

        # create target variable

        logger.info(f"Ending with {pitcher_df.height} pitches")

        df_list.append(pitcher_df)

    df = pl.concat(df_list)

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
