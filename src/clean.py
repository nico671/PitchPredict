import logging
import sys
import time

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
    df = pl.scan_parquet(input_file_path)

    # drop columns that will never be used
    df = df.drop(
        [
            "Unnamed: 0",
            "spin_dir",
            "spin_rate_deprecated",
            "break_angle_deprecated",
            "break_length_deprecated",
            "game_type",
            "home_team",
            "away_team",
            "des",
            "description",
            "game_year",
            "tfs_deprecated",
            "tfs_zulu_deprecated",
            "home_score",
            "away_score",
            "fielder_2",
            "umpire",
            "sv_id",
            "pitcher.1",
            "fielder_2.1",
            "fielder_3",
            "fielder_4",
            "fielder_5",
            "fielder_6",
            "fielder_7",
            "fielder_8",
            "fielder_9",
            "pitch_name",
            "post_away_score",
            "post_home_score",
            "post_bat_score",
            "post_fld_score",
            "delta_home_win_exp",
            "delta_run_exp",
        ]
    )

    # filter to correct years
    df = df.with_columns(pl.col("game_date").str.to_datetime())
    df = df.filter(pl.col("game_date").dt.year() >= params["clean"]["start_year"])

    # get top k pitchers (decided by number of pitches and num_pitchers from params.yaml)
    pitcher_counts = df.group_by("pitcher").len().sort("len", descending=True)
    top_k_pitchers = pitcher_counts.head(params["clean"]["num_pitchers"]).select(
        "pitcher"
    )

    # Collect the 'pitcher' column into a list and filter the dataframe to only include the top k pitchers
    top_k_pitchers_list = top_k_pitchers.collect().get_column("pitcher").to_list()
    df = df.filter(pl.col("pitcher").is_in(top_k_pitchers_list))

    # drop rows with null values in the columns 'release_speed' and 'release_pos_x', as they cant be used for training
    df = df.fill_nan(-1)
    df = df.fill_null(-1)

    df.sink_parquet(params["featurize"]["input_data_path"])
    end_time = time.time()
    logger.info(f"Cleaning took {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
