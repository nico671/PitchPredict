import logging
import sys
import time
from pathlib import Path

# import pandas as pd
import polars as pl
import pybaseball as pb
import yaml

logger = logging.getLogger("featurize")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
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
        descending=[False, False, False, True],
    )

    df.head()

    df = df.with_columns(
        df.select(
            pl.col(pl.String)
            .exclude(["player_name"])
            .cast(pl.Categorical)
            .to_physical()
        ),
    )
    df = df.fill_null(-1)
    df = df.sort(
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ],
        descending=True,
    )

    # create target variable
    df = df.with_columns(
        df.select(pl.col("pitch_type").shift(-1).alias("next_pitch")),
    ).drop_nulls("next_pitch")

    # create count feature
    df = df.with_columns(
        (pl.col("balls").cast(pl.String) + " - " + pl.col("strikes").cast(pl.String))
        .alias("count")
        .cast(pl.Categorical)
        .to_physical()
    )
    df = df.drop(["balls", "strikes"])

    # create base state feature
    df = df.with_columns(
        pl.col("on_1b")
        .map_elements(lambda s: 0 if s == -1.0 else 1, return_dtype=pl.Int32)
        .alias("on_1b"),
        pl.col("on_2b")
        .map_elements(lambda s: 0 if s == -1.0 else 1, return_dtype=pl.Int32)
        .alias("on_2b"),
        pl.col("on_3b")
        .map_elements(lambda s: 0 if s == -1.0 else 1, return_dtype=pl.Int32)
        .alias("on_3b"),
    )
    df = df.with_columns(
        (pl.col("on_1b") * 3 + pl.col("on_2b") * 5 + pl.col("on_3b") * 7).alias(
            "base_state"
        )
    )
    df = df.drop(["on_1b", "on_2b", "on_3b"])

    # create run_diff feature
    df = df.with_columns(
        (pl.col("fld_score") - pl.col("bat_score")).alias("run_diff").cast(pl.Int32),
    )
    df = df.drop(["fld_score", "bat_score"])

    # calculate current game pitch count
    df = df.with_columns(
        pl.col("pitch_type")
        .cum_count(reverse=False)
        .over(["player_name", "game_date", "game_pk"])
        .alias("pitches_thrown_curr_game")
    )
    df = df.sort(
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ],
        descending=True,
    )

    batting_df = pb.batting_stats_bref(params["clean"]["start_year"])
    print(batting_df.columns)
    player_ids = list(df.select("batter").unique().to_pandas()["batter"])
    batting_df = pl.DataFrame(batting_df[batting_df["mlbID"].isin(player_ids)])
    batting_df = batting_df.drop(
        [
            "Name",
            "Age",
            "#days",
            "Lev",
            "Tm",
            "G",
            "PA",
            "AB",
            "R",
            "H",
            "2B",
            "3B",
            "HR",
            "RBI",
            "BB",
            "IBB",
            "SO",
            "HBP",
            "SH",
            "SF",
            "GDP",
            "SB",
            "CS",
        ]
    )
    df = df.join(batting_df, left_on="batter", right_on="mlbID", how="left")

    if "next_pitch" not in df.columns:
        logger.error("next_pitch not in columns")
        sys.exit(1)

    features = df.columns
    features_path = Path(params["train"]["features_path"])
    with open(features_path, "r") as f:
        for item in f.readlines():
            if item not in ["next_pitch", "pitcher", "player_name", "game_date"]:
                features.append(item.strip())
    features = list(set(features))
    logger.info(f"Features: {features}")
    # Create the output DataFrame
    output_df = df
    output_df = output_df.sort(
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ],
        descending=True,
    )
    # Ensure output directory exists
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(output_dir / "2015_2024_statcast_train.parquet")
    # Write to Parquet file
    output_df.write_parquet(output_dir / "2015_2024_statcast_train.parquet")
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info("done")


if __name__ == "__main__":
    main()
