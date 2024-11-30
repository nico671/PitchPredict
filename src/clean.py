import logging
import sys
import time
from pathlib import Path

import polars as pl
import yaml

logger = logging.getLogger("mr. cleannnnn")
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
    input_file_path = params["clean"]["input_data_path"]

    # read in the complete data frame
    df = pl.read_parquet(input_file_path)

    df = df.with_columns(pl.col("game_date").str.to_datetime())
    df = df.filter(pl.col("game_date").dt.year() >= params["clean"]["start_year"])

    df.filter(pl.col("pitcher").is_not_null())

    # get top k pitchers (decided by number of pitches and num_pitchers from params.yaml)
    pitcher_counts = df.group_by("pitcher").len().sort("len", descending=True)
    top_k_pitchers = pitcher_counts.head(params["clean"]["num_pitchers"])["pitcher"]
    df = df.filter(pl.col("pitcher").is_in(top_k_pitchers))

    # check that dataframe is not empty
    if df.is_empty():
        logger.error("Dataframe is empty")
        sys.exit(1)

    logger.info(f"Num rows is {df.shape[0]}")

    # drop duplicate rows
    df = df.unique(keep="first")
    logger.info(f"Num rows is {df.shape[0]}")

    # output the to featurization pq file
    output_dir = Path("data/cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_dir / "2015_2024_statcast_clean.parquet")
    logger.info("done")
    end_time = time.time()
    logger.info(f"Cleaning took {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
