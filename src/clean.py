import logging
import time

import polars as pl
import yaml

logger = logging.getLogger("mr. cleannnnn")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)


def main():
    start_time = time.time()
    # read in the complete data frame
    df = pl.scan_parquet(params["clean"]["input_data_path"])

    # drop columns that will never be used
    df = df.drop(
        [
            # "events",
            # "spin_dir",
            "delta_home_win_exp",
            # "ax",
            # "ay",
            # "post_bat_score",
            # "post_fld_score",
            # "az",
            "spin_rate_deprecated",
            "break_angle_deprecated",
            "break_length_deprecated",
            "game_type",
            "home_team",
            "away_team",
            "des",
            "hc_x",
            "hc_y",
            "description",
            "game_year",
            "tfs_deprecated",
            "tfs_zulu_deprecated",
            "home_score",
            "away_score",
            "fielder_2",
            "umpire",
            "sv_id",
            "fielder_3",
            "fielder_4",
            "fielder_5",
            "fielder_6",
            "fielder_7",
            "fielder_8",
            "fielder_9",
            "pitch_name",
            # "p_throws",
            # "zone",
            # "hit_distance_sc",
            # "launch_angle",
            # "inning_topbot",
            "post_home_score",
            "post_away_score",
            # "sz_top",
            # "sz_bot",
            # "bat_speed",
            # "launch_speed",
            # "swing_length",
            # "hit_location",
            # "events",
            # "description",
            # "spin_dir",
            # "estimated_ba_using_speedangle",
            # "estimated_woba_using_speedangle",
            # "woba_value",
            # "woba_denom",
            # "babip_value",
            # "iso_value",
            # "launch_speed_angle",
        ]
    )
    df = df.with_columns(pl.col("game_date").str.to_datetime())  # convert to datetime
    # filter to correct years
    df = df.filter(
        pl.col("game_date").dt.year() >= params["clean"]["start_year"]
    )  # filter to start year

    # drop rows with null values in the columns 'pitch_type' and 'pitcher', as they cant be used for training
    df = df.drop_nulls(subset=["pitch_type", "pitcher"])
    
    # get top k pitchers (decided by number of pitches and num_pitchers from params.yaml)
    pitcher_counts = df.group_by("pitcher").len().sort("len", descending=True)
    top_k_pitchers = pitcher_counts.head(params["clean"]["num_pitchers"]).select(
        "pitcher"
    )

    top_k_pitchers_list = (
        top_k_pitchers.collect().get_column("pitcher").to_list()
    )  # Collect the 'pitcher' column into a list

    df = df.filter(
        pl.col("pitcher").is_in(top_k_pitchers_list)
    )  # filter the dataframe to only include the top k pitchers
    df = df.rename({"player_name": "pitcher_name"})
    df.sink_parquet(params["featurize"]["input_data_path"])
    end_time = time.time()
    logger.info(f"Cleaning took {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
