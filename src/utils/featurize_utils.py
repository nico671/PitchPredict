import polars as pl
import pybaseball as pb


def create_run_diff_feature(df):
    df = df.with_columns(
        (pl.col("fld_score") - pl.col("bat_score")).alias("run_diff").cast(pl.Int32),
    )
    df = df.drop(["fld_score", "bat_score"])
    df = df.sort(
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ],
        descending=False,
    )
    return df


def create_base_state_feature(df):
    df = df.with_columns(df.select(["on_1b", "on_2b", "on_3b"]).fill_null(-1))
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
    return df.drop(["on_1b", "on_2b", "on_3b"])


def add_batting_stats(df, start_year):
    batting_df = pb.batting_stats_bref(start_year)
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
    return df.join(batting_df, left_on="batter", right_on="mlbID", how="left")


def create_count_feature(df):
    df = df.with_columns(
        (pl.col("balls").cast(pl.String) + " - " + pl.col("strikes").cast(pl.String))
        .alias("count")
        .cast(pl.Categorical)
        .to_physical()
    )
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
    # create target variable
    return df.with_columns(
        df.select(pl.col("pitch_type").shift().alias("next_pitch")),
    ).drop_nulls("next_pitch")


def handle_missing_values(df):
    df = df.fill_null(-1)
    df = df.fill_nan(-1)
    return df


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
