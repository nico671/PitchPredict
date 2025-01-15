import numpy as np
import polars as pl
import pybaseball as pb


def create_consistency_feature(df):
    return df.with_columns(
        (pl.col("release_pos_x").diff().pow(2) + pl.col("release_pos_z").diff().pow(2))
        .sqrt()
        .alias("release_point_consistency"),
        (pl.col("pfx_x") * (pl.col("plate_x") - pl.col("release_pos_x"))).alias(
            "late_break_x"
        ),
        (pl.col("pfx_z") * (pl.col("plate_z") - pl.col("release_pos_z"))).alias(
            "late_break_z"
        ),
        (pl.arctan2(pl.col("vz0"), pl.col("vy0")) * 180 / np.pi).alias(
            "approach_angle"
        ),
    )


def create_run_diff_feature(df):
    # create a new column for run differential
    df = df.with_columns(
        (pl.col("fld_score") - pl.col("bat_score")).alias("run_diff").cast(pl.Int32),
    ).drop(["fld_score", "bat_score"])
    return df.with_columns(
        (pl.col("run_diff").abs() >= 4).alias("large_lead"),
        (pl.col("run_diff").abs() <= 2).alias("close_game"),
        (pl.col("inning") >= 10).alias("extra_innings"),
    )


def create_base_state_feature(df):
    # fill null values with -1, since null means base is empty
    df = df.with_columns(df.select(["on_1b", "on_2b", "on_3b"]).fill_null(-1))

    # map the base state to 0 if there isn't a runner on that base (i.e. was previously null), 1 otherwise meaning there is a runner on that base
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
    return df.with_columns(
        ((pl.col("on_2b") == 1) | (pl.col("on_3b") == 1))
        .alias("runners_in_scoring_position")
        .cast(pl.Int32),
    )


def add_batting_stats(df, start_year):
    # get the batting stats for all players since the start year
    batting_df = pb.batting_stats_bref(start_year)
    # get the unique batter ids from the dataframe (there is probably a more "Polars" way to do this sorry)
    player_ids = list(df.select("batter").unique().to_pandas()["batter"])
    # filter the batting stats dataframe to only include the players in the unique batter ids (aka in our dataframe)
    batting_df = pl.DataFrame(batting_df[batting_df["mlbID"].isin(player_ids)]).drop(
        [
            "Name",
            "Age",
            "#days",
            "Lev",
            "Tm",
            "G",
            "PA",
            "AB",
            "SO",
            "HBP",
            "SH",
            "SF",
            "SB",
            "R",
            "RBI",
            "IBB",
            "2B",
            "3B",
            "GDP",
            "CS",
        ]
    )

    # join the batting stats dataframe with our dataframe on the batter column
    df = df.join(batting_df, left_on="batter", right_on="mlbID", how="left")

    return df.with_columns(
        pl.col("SLG").mean().over(["batter", "zone"]).alias("batter_zone_slug"),
        pl.col("BA").mean().over(["batter", "zone"]).alias("batter_zone_ba"),
        pl.col("BA")
        .mean()
        .over(["batter", "pitch_type"])
        .alias("batter_pitch_type_avg"),
        (pl.col("stand") != pl.col("p_throws")).alias("platoon_advantage"),
    )


def create_lookback_features(df):
    return df.with_columns(
        pl.col("pitch_type")
        .shift(-1)
        .over(["balls", "strikes", "inning", "run_diff"])
        .alias("last_pitch_in_situation"),
        pl.col("pitch_type")
        .shift(-1)
        .over(["balls", "strikes"])
        .alias("last_pitch_in_count"),
        pl.col("zone")
        .shift(-1)
        .over(["balls", "strikes", "inning", "run_diff"])
        .alias("last_zone_in_situation"),
        pl.col("zone").shift(-1).over(["balls", "strikes"]).alias("last_zone_in_count"),
    )


def create_count_feature(df):
    # create a new column for the count, concatenating the balls and strikes columns,
    # then casting to a categorical variable, then to a physical type which means integer,
    # so now we have encoded all the possible base states in one column as numbers
    df = df.with_columns(
        (pl.col("balls").cast(pl.String) + " - " + pl.col("strikes").cast(pl.String))
        .alias("count")
        .cast(pl.Categorical)
        .to_physical()
    )

    df = df.with_columns(
        pl.when(
            (pl.col("balls") == 3) & (pl.col("strikes") == 2)
            | (pl.col("balls") == 3) & (pl.col("strikes") == 1)
            | (pl.col("balls") == 2) & (pl.col("strikes") == 1)
            | (pl.col("balls") == 2) & (pl.col("strikes") == 2)
        )
        .then(1)
        .otherwise(0)
        .alias("high_leverage_count"),
        pl.when((pl.col("balls") == 0) & (pl.col("strikes") == 0))
        .then(1)
        .otherwise(0)
        .alias("first_pitch"),
        pl.when(pl.col("balls") > pl.col("strikes"))
        .then(1)
        .otherwise(0)
        .alias("pitcher_ahead"),
        pl.when(pl.col("balls") < pl.col("strikes"))
        .then(1)
        .otherwise(0)
        .alias("batter_ahead"),
        pl.when(pl.col("strikes") == 2).then(1).otherwise(0).alias("two_strikes"),
        pl.when(pl.col("balls") == 3).then(1).otherwise(0).alias("three_balls"),
    )
    # drop the original columns
    return df


def create_target(df):
    # Add a `next_pitch` column by shifting `pitch_type`
    df = df.with_columns(pl.col("pitch_type").shift(-1).alias("next_pitch"))

    # Ensure the `next_pitch` of the last pitch in a game is null (no valid next pitch)
    df = df.with_columns(
        pl.when(pl.col("game_pk").shift(-1) != pl.col("game_pk"))
        .then(None)
        .otherwise(pl.col("next_pitch"))
        .alias("next_pitch")
    )

    # Drop rows where `next_pitch` is null
    df = df.drop_nulls("next_pitch")

    return df


# fill null values with -1, makes sure LSTM can handle the data (maybe there is a better way to do this but its working for me)
def handle_missing_values(df):
    df = df.fill_null(0)
    return df


# encode all the categorical features as integers
def encode_categorical_features(df):
    return df.with_columns(
        df.select(
            pl.col(pl.String)
            .exclude(["player_name"])
            .cast(pl.Categorical)
            .to_physical()
        ),
    )


def add_shift_features(df):
    df = df.sort(
        [
            "game_date",
            "game_pk",
            "inning",
            "at_bat_number",
            "pitch_number",
        ],
        descending=False,
    )

    # Previous N pitches
    for n in [1, 2, 3]:
        df = df.with_columns(
            pl.col("pitch_type")
            .shift(n)
            .over(["game_pk", "inning", "at_bat_number"])
            .alias(f"pitch_type_minus_{n}"),
            pl.col("plate_x")
            .shift(n)
            .over(["game_pk", "inning", "at_bat_number"])
            .alias(f"pitch_location_x_minus_{n}"),
            pl.col("plate_z")
            .shift(n)
            .over(["game_pk", "inning", "at_bat_number"])
            .alias(f"pitch_location_z_minus_{n}"),
            pl.col("release_speed")
            .shift(n)
            .over(["game_pk", "inning", "at_bat_number"])
            .alias(f"release_speed_minus_{n}"),
        )

    # Pitch type transitions
    # df["back_to_back_same_pitch"] = df["pitch_type"] == df["pitch_type_minus_1"]
    df = df.with_columns(
        (pl.col("pitch_type") == pl.col("pitch_type_minus_1")).alias(
            "back_to_back_same_pitch"
        ),
        (pl.col("plate_x") - pl.col("pitch_location_x_minus_1")).alias("plate_x_diff"),
        (pl.col("plate_z") - pl.col("pitch_location_z_minus_1")).alias("plate_z_diff"),
        (pl.col("release_speed") - pl.col("release_speed_minus_1")).alias(
            "release_speed_diff"
        ),
    )
    return df.with_columns(
        ((pl.col("plate_x_diff") < 0.5) & (pl.col("plate_z_diff") < 0.5)).alias(
            "same_location"
        ),
    )
