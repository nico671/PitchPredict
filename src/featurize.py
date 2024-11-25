import logging
import sys
from pathlib import Path
import pandas as pd
import yaml

logger = logging.getLogger("featurize")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def next_pitch(player):
    player = player.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )
    logger.info(player)
    player["next_pitch"] = player["pitch_type"].shift(-1)
    return player


def main():
    if len(sys.argv) != 2:
        logger.error("Arguments error. Usage:\n")
        logger.error("not enough inputs, expected input structure is: *.py *.parquet")
        sys.exit(1)

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    input_file_path = params["featurize"]["input_data_path"]
    df = pd.read_parquet(Path(input_file_path))
    logger.info(f"Shape is initially: {df.shape[0]} rows and {df.shape[1]} columns")

    logger.info("Creating target column")

    df = df.groupby(["player_name"]).apply(
        lambda gdf: gdf.assign(
            next_pitch=lambda df: df["pitch_type"].shift(-1), include_groups=False
        )
    )

    # create target
    if "next_pitch" not in df.columns:
        logger.error("next_pitch column not created")
        sys.exit(1)
    else:
        logger.info("next_pitch column created")
        df = df.reset_index(drop=True)

    logger.info(f"Null count: {df.isnull().sum()}")

    logger.info("Converting game_date to datetime")
    df["game_date"] = pd.to_datetime(df["game_date"])

    logger.info("Converting needing columns to categorical to work with lstm")
    df["stand"] = df["stand"].astype("category").cat.codes
    df["count"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    df["count"] = df["count"].astype("category").cat.codes
    df["on_1b"] = df["on_1b"].notnull().astype(int)
    df["on_2b"] = df["on_2b"].notnull().astype(int)
    df["on_3b"] = df["on_3b"].notnull().astype(int)
    df["inning_topbot"] = df["inning_topbot"].astype("category").cat.codes
    df["if_fielding_alignment"] = (
        df["if_fielding_alignment"].astype("category").cat.codes
    )
    df["of_fielding_alignment"] = (
        df["of_fielding_alignment"].astype("category").cat.codes
    )
    df["outs_when_up"] = df["outs_when_up"].astype(int)
    df["inning"] = df["inning"].astype(int)
    del df[
        "balls"
    ]  # remove balls and strikes which are useless since we already have count
    del df["strikes"]

    df["run_diff"] = df["bat_score"] - df["fld_score"]  # run difference
    del df["bat_score"]  # remove bat score
    del df["fld_score"]  # remove field score
    df["base_state"] = (
        df["on_1b"].astype(int)
        + 2 * df["on_2b"].astype(int)
        + 4 * df["on_3b"].astype(int)
    )

    # Pitcher tendencies
    df["cumulative_pitch_count"] = df.groupby(["game_date", "pitcher"]).cumcount() + 1
    df["is_high_pressure"] = (
        ((df["inning"] >= 7) & (abs(df["run_diff"]) <= 3))
    ).astype(int)

    features = []
    features_path = Path(params["featurize"]["features_path"])
    with open(features_path, "r") as f:
        for item in f.readlines():
            features.append(item.strip())

    df["is_tied"] = (df["run_diff"] == 0).astype(int)
    df["is_leading"] = (df["run_diff"] > 0).astype(int)
    df["is_trailing"] = (df["run_diff"] < 0).astype(int)
    df["pitcher_game_pitch_count"] = df.groupby(["game_date", "pitcher"]).cumcount() + 1
    df["spin_rate"] = df["release_spin_rate"].astype(float).fillna(-1)
    df["next_pitch"] = df["next_pitch"].astype("str")

    logger.info(f"Features: {features}")
    logger.info(f"Shape is now: {df.shape[0]} rows and {df.shape[1]} columns")
    df = df.fillna(-1)

    # Fill NaN values if necessary
    df = df.fillna(-1).infer_objects(copy=False)
    # Create the output DataFrame
    output_df = df[features + ["next_pitch", "pitcher", "player_name", "game_date"]]
    # Drop duplicated columns
    output_df = output_df.loc[:, ~output_df.columns.duplicated()]
    # Ensure output directory exists
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to Parquet file
    output_df.to_parquet(output_dir / "2015_2024_statcast_train.parquet")

    logger.info("done")


if __name__ == "__main__":
    main()
