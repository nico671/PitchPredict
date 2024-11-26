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
    input_file_path = params["clean"]["input_data_path"]

    df = pd.read_parquet(Path(input_file_path))
    logger.info(f"Shape is initially: {df.shape[0]} rows and {df.shape[1]} columns")

    logger.info("Creating target column")

    df = (
        df.set_index("Unnamed: 0")
        .groupby("player_name")
        .apply(
            lambda x: x.assign(next_pitch=x["pitch_type"].shift(-1)),
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
    # Create necessary features

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
    del df["balls"]
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
    df["is_high_pressure"] = ((df["inning"] >= 7) & (abs(df["run_diff"]) <= 3)).astype(
        int
    )
    df["is_tied"] = (df["run_diff"] == 0).astype(int)
    df["is_leading"] = (df["run_diff"] > 0).astype(int)
    df["is_trailing"] = (df["run_diff"] < 0).astype(int)
    df["pitcher_game_pitch_count"] = df.groupby(["game_date", "pitcher"]).cumcount() + 1
    df["spin_rate"] = df["release_spin_rate"].astype(float).fillna(-1)

    # Define features list
    features = [
        "stand",
        "is_high_pressure",
        "zone",
        "cumulative_pitch_count",
        "count",
        "inning_topbot",
        "if_fielding_alignment",
        "of_fielding_alignment",
        "at_bat_number",
        "pitch_number",
        "run_diff",
        "base_state",
        "release_speed",
        "release_pos_x",
        "release_pos_z",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "outs_when_up",
        "inning",
        "hc_x",
        "hc_y",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "hit_distance_sc",
        "launch_speed",
        "launch_angle",
        "effective_speed",
        "release_spin_rate",
        "release_extension",
        "release_pos_y",
        "estimated_woba_using_speedangle",
        "woba_value",
        "woba_denom",
        "babip_value",
        "iso_value",
        "launch_speed_angle",
        "spin_axis",
        "delta_run_exp",
        "is_tied",
        "is_leading",
        "is_trailing",
        "pitcher_game_pitch_count",
        "spin_rate",
    ]

    logger.info(f"Features: {features}")
    logger.info(f"Shape is now: {df.shape[0]} rows and {df.shape[1]} columns")

    # Fill NaN values if necessary
    df = df.fillna(-1).infer_objects(copy=False)
    # Create the output DataFrame
    output_df = df[features + ["next_pitch", "pitcher", "player_name", "game_date"]]
    # Ensure output directory exists
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(output_dir / "2015_2024_statcast_train.parquet")
    # Write to Parquet file
    output_df.to_parquet(output_dir / "2015_2024_statcast_train.parquet")

    logger.info("done")


if __name__ == "__main__":
    main()
