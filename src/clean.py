import polars as pl
import yaml

keep_cols = [
    "pitch_type",
    "pitcher",
    "game_date",
    "game_pk",
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "zone",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "on_3b",
    "on_2b",
    "on_1b",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "sz_top",
    "sz_bot",
    "release_spin_rate",
    "release_extension",
    "release_pos_y",
    "at_bat_number",
    "pitch_number",
    "bat_score",
    "fld_score",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "spin_axis",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
]

with open("params.yaml") as f:
    TOP_K_PITCHERS = yaml.safe_load(f)["clean"]["top_k_pitchers"]


def encode_base_states(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("on_3b").fill_null(0).cast(pl.Int32),
        pl.col("on_2b").fill_null(0).cast(pl.Int32),
        pl.col("on_1b").fill_null(0).cast(pl.Int32),
        pl.col("if_fielding_alignment").fill_null(""),
        pl.col("of_fielding_alignment").fill_null(""),
    )
    df = df.with_columns(
        pl.when(pl.col("on_3b") == 0).then(0).otherwise(1).alias("on_3b"),
        pl.when(pl.col("on_2b") == 0).then(0).otherwise(1).alias("on_2b"),
        pl.when(pl.col("on_1b") == 0).then(0).otherwise(1).alias("on_1b"),
    )
    return df


def clean():
    df = pl.read_parquet("data/raw_statcast_data.parquet")
    # remove rows with either pitch_type or release_speed null, they are useless
    df = df.drop_nulls(
        [
            "pitch_type",
            "release_speed",
        ]
    )

    top_k_pitcher_codes = (
        df.group_by("pitcher")
        .agg(pl.col("pitch_type").count().alias("pitch_count"))
        .sort("pitch_count", descending=True)
        .head(5)
        .select(pl.col("pitcher"))
        .to_series()
        .to_list()
    )
    df = df.filter(pl.col("pitcher").is_in(top_k_pitcher_codes))
    # only keep the columns we need
    df = df.select(keep_cols)

    df = encode_base_states(df)
    df = df.fill_null(-1)

    df = df.fill_nan(-1)
    cont_cols = ["release_spin_rate", "release_extension", "spin_axis"]

    df = df.with_columns(
        [pl.col(col).fill_null(0).cast(pl.Float32) for col in cont_cols]
    )
    df = df.with_columns(
        [
            (
                pl.when(pl.col(col).is_null())  # if value is null
                .then(pl.col(col).median().over("pitch_type"))  # use group median
                .otherwise(pl.col(col))  # else keep original
            ).alias(col)
            for col in cont_cols
        ]
    )
    # for col in df.columns:
    #     print(f"{col}: {df[col].dtype}")
    cat_cols = [
        "stand",
        "p_throws",
        "inning_topbot",
        "if_fielding_alignment",
        "of_fielding_alignment",
    ]
    for col in cat_cols:
        df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical())
    df = df.with_columns(pl.col("game_date").str.strptime(pl.Date, format="%Y-%m-%d"))
    df.write_parquet("data/clean_statcast_data.parquet")
    print("Cleaned data saved to data/clean_statcast_data.parquet")


if __name__ == "__main__":
    clean()
