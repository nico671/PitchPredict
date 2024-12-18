import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("mr. cleannnnn")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


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

    # read in the complete data frame
    df = pd.read_parquet(Path(input_file_path))
    # check that dataframe is not empty
    if df.empty:
        logger.error("Dataframe is empty, please check the input file")
        sys.exit(1)

    logger.info(f"Shape is initially: {df.shape[0]} rows and {df.shape[1]} columns")

    # conversion of game_date to datetime
    logger.info("Converting game_date to datetime")
    df["game_date"] = pd.to_datetime(df["game_date"])

    # filter out games before start_year
    start_year = params["clean"]["start_year"]
    df = df[df["game_date"].dt.year >= start_year]

    num_pitchers = params["clean"]["num_pitchers"]
    # get the n highest appearing pitchers
    top_pitchers = df["pitcher"].value_counts().nlargest(num_pitchers).index
    df = df[df["pitcher"].isin(top_pitchers)]

    min_pitches = params["clean"]["min_pitches"]
    # filter out pitchers with less than min_pitches
    pitcher_counts = df["pitcher"].value_counts()
    df = df[df["pitcher"].isin(pitcher_counts[pitcher_counts > min_pitches].index)]

    # drop duplicate rows
    df = df.drop_duplicates(keep="first")
    logger.info(f"Shape is now: {df.shape[0]} rows and {df.shape[1]} columns")

    # output the to featurization pq file
    output_dir = Path("data/cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_dir / "2015_2024_statcast_clean.parquet")
    logger.info("done")


if __name__ == "__main__":
    main()
