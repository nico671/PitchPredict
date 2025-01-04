import logging
import sys
import time
from pathlib import Path

import polars as pl
import yaml

from src.utils.featurize_utils import (
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

    # Handle missing values
    df = handle_missing_values(df)

    df = add_batting_stats(df, params["clean"]["start_year"])

    # Create the output DataFrame
    output_df = df
    output_df = output_df.sort(
        [
            "game_date",
            "game_pk",
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
