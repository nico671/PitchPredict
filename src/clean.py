import polars as pl
import yaml

KEEP_COLS = [
    "pitch_type",
    "game_date",
    "release_speed",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "player_name",
    "events",
    "zone",
    "stand",
    "p_throws",
    "type",
    "hit_location",
    "bb_type",
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
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "sz_top",
    "sz_bot",
    "hit_distance_sc",
    "launch_speed",
    "launch_angle",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "at_bat_number",
    "pitch_number",
    "bat_score",
    "fld_score",
    "spin_axis",
    "n_thruorder_pitcher",
    "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
]

with open("params.yaml") as f:
    yaml_dict = yaml.safe_load(f)
    TOP_K_PITCHERS = yaml_dict["clean"]["top_k_pitchers"]
    NUM_YEARS = yaml_dict["clean"]["num_years"]


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

    # filter to desired number of years
    df = df.with_columns(pl.col("game_date").str.to_date(format="%Y-%m-%d"))
    df = df.with_columns(pl.col("game_date").dt.year().alias("year"))
    max_year = df.select(pl.col("year").max()).item()
    df = df.filter(pl.col("year") >= max_year - NUM_YEARS)

    # remove rows with either pitch_type or release_speed null, they are useless
    df = df.drop_nulls(
        [
            "pitch_type",
            "release_speed",
        ]
    )

    top_k_player_codes = (
        df.group_by("player_name")
        .agg(pl.col("pitch_type").count().alias("pitch_count"))
        .sort("pitch_count", descending=True)
        .head(TOP_K_PITCHERS)
        .select(pl.col("player_name"))
        .to_series()
        .to_list()
    )
    df = df.filter(pl.col("player_name").is_in(top_k_player_codes))
    # only keep the columns we need
    full_df = df.select(KEEP_COLS)
    df = df.with_columns(
        pl.col("events").fill_null("N/A").cast(pl.Categorical).to_physical()
    )
    hand_to_int = {"R": 0, "L": 1}
    df = df.with_columns(
        [
            pl.col("stand").replace_strict(hand_to_int).alias("stand"),
            pl.col("p_throws").replace_strict(hand_to_int).alias("p_throws"),
        ]
    )
    df = df.rename({"type": "pitch_result"})
    df = df.with_columns(
        [
            pl.col("spin_axis").fill_null(-1).cast(pl.Int16).alias("spin_axis"),
            pl.col("bb_type").fill_null("N/A").cast(pl.Categorical).to_physical(),
            pl.col("hit_location").fill_null(0).cast(pl.Int8).alias("hit_location"),
            pl.col("pitch_result").fill_null("N/A").cast(pl.Categorical).to_physical(),
        ]
    )
    df = df.with_columns(
        pl.col(
            [
                "hit_distance_sc",
                "launch_angle",
                "release_spin_rate",
                "pitcher_days_since_prev_game",
            ]
        ).fill_null(-1),
        pl.col(["launch_speed", "effective_speed", "release_extension"]).fill_null(
            -1.0
        ),
    )

    df = df.with_columns(
        (pl.col("balls").cast(pl.String) + "-" + pl.col("strikes").cast(pl.String))
        .cast(pl.Categorical)
        .to_physical()
        .alias("balls_strikes"),
    ).drop(["balls", "strikes"])
    df = df.with_columns(
        pl.when(pl.col("on_3b").is_null()).then(0).otherwise(1).alias("on_3b"),
        pl.when(pl.col("on_2b").is_null()).then(0).otherwise(1).alias("on_2b"),
        pl.when(pl.col("on_1b").is_null()).then(0).otherwise(1).alias("on_1b"),
    )
    df_list = []
    for p, df in full_df.group_by("player_name"):
        rare_pitch_types = (
            df.select(pl.col("pitch_type").value_counts(normalize=True))
            .unnest(columns=["pitch_type"])
            .filter(pl.col("proportion") < 0.03)
            .select(pl.col("pitch_type"))
            .to_series()
            .to_list()
        )
        df = df.with_columns(
            pl.when(pl.col("pitch_type").is_in(rare_pitch_types))
            .then(pl.lit("other"))
            .otherwise(pl.col("pitch_type"))
            .alias("pitch_type")
        )
        # Get unique pitch types and create a mapping dictionary
        unique_pitches = df.select(pl.col("pitch_type")).unique().to_series().sort()
        pitch_to_int = {pitch: i for i, pitch in enumerate(unique_pitches)}
        int_to_pitch = {i: pitch for pitch, i in pitch_to_int.items()}

        # Print the mapping for reference
        print("Pitch type mapping:")
        for pitch, code in pitch_to_int.items():
            print(f"{pitch}: {code}")

        # Create a new column with integer codes
        df = df.with_columns(
            pl.col("pitch_type").replace_strict(pitch_to_int).alias("pitch_type_code")
        )

        # Show the result
        df.select(["pitch_type", "pitch_type_code"]).head()
        df = df.drop("pitch_type")
        df_list.append(df)
    full_df = pl.concat(df_list)
    full_df = full_df.sort(by=["game_date", "at_bat_number", "pitch_number"])
    full_df.write_parquet("data/clean_statcast_data.parquet")
    print("Cleaned data saved to data/clean_statcast_data.parquet")


if __name__ == "__main__":
    clean()
