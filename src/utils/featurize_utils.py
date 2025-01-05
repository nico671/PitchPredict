import polars as pl
import pybaseball as pb


def create_run_diff_feature(df):
    # create a new column for run differential
    df = df.with_columns(
        (pl.col("fld_score") - pl.col("bat_score")).alias("run_diff").cast(pl.Int32),
    )
    # drop the original columns
    df = df.drop(["fld_score", "bat_score"])
    return df


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
    # create a new column for the base state
    df = df.with_columns(
        (pl.col("on_1b") * 3 + pl.col("on_2b") * 5 + pl.col("on_3b") * 7).alias(
            "base_state"
        )
    )
    # drop the original columns
    return df.drop(["on_1b", "on_2b", "on_3b"])


def add_batting_stats(df, start_year):
    # get the batting stats for all players since the start year
    batting_df = pb.batting_stats_bref(start_year)
    # get the unique batter ids from the dataframe (there is probably a more "Polars" way to do this sorry)
    player_ids = list(df.select("batter").unique().to_pandas()["batter"])
    # filter the batting stats dataframe to only include the players in the unique batter ids (aka in our dataframe)
    batting_df = pl.DataFrame(batting_df[batting_df["mlbID"].isin(player_ids)])
    # drop the columns that we don't need
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
    return df.join(batting_df, left_on="batter", right_on="mlbID", how="left")


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
    # drop the original columns
    return df.drop(["balls", "strikes"])


def create_target(df):
    df = df.sort(
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ],
        descending=False,
    )
    # create target variable, shifting one row down to make it the next pitch since our data is sorted to descend as time passes
    # in the future i want to create cutoffs, like at the end of an at bat, inning, etc. but for now this is fine
    return df.with_columns(
        df.select(pl.col("pitch_type").shift().alias("next_pitch")),
    ).drop_nulls("next_pitch")


# fill null values with -1, makes sure LSTM can handle the data (maybe there is a better way to do this but its working for me)
def handle_missing_values(df):
    df = df.fill_null(-1)
    df = df.fill_nan(-1)
    return df


# encode all the categorical features as integers
def encode_categorical_features(df):
    df = df.with_columns(
        df.select(
            pl.col(pl.String)
            .exclude(["player_name"])
            .cast(pl.Categorical)
            .to_physical()
        ),
    )
    return df
