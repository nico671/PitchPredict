import os

import polars as pl
from sklearn.preprocessing import MinMaxScaler

OUTPUT_DIR = "data/featurized"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEQ_LEN = 25


def featurize(df):
    df = df.sort(by=["game_date", "at_bat_number", "pitch_number"])
    df = df.with_columns(
        pl.col("pitch_type_code")
        .shift(-1)
        .over("game_date")
        .alias("next_pitch_type_code"),
    )

    df = df.drop_nulls(subset=["next_pitch_type_code", "pitch_type_code"])
    rolling_cols = [
        "release_speed",
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "plate_x",
        "plate_z",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "spin_axis",
        "bb_type",
    ]

    rolling_window = 5
    df = df.with_columns(
        [
            pl.col(col)
            .rolling_mean(window_size=rolling_window, min_samples=1)
            .alias(f"{col}_rolling_mean")
            for col in rolling_cols
        ]
    )
    # last result in this current count

    df = df.with_columns(
        pl.col("pitch_result")
        .backward_fill()
        .over(["balls_strikes", "on_3b", "on_2b", "on_1b"])
        .alias("last_pitch_result"),
    )
    mms = MinMaxScaler()

    def scale_columns(df, columns):
        df[columns] = mms.fit_transform(df[columns])
        return df

    cols_to_scale = list(df.columns)
    cols_to_scale.remove("game_date")
    cols_to_scale.remove("player_name")
    cols_to_scale.remove("pitch_type_code")
    cols_to_scale.remove("next_pitch_type_code")
    cols_to_scale.remove("events")
    scale_columns(df, cols_to_scale)
    df.write_parquet(
        os.path.join(OUTPUT_DIR, "featurized_statcast_data.parquet"),
        compression="gzip",
    )
    print(f"Featurized data saved to {OUTPUT_DIR}/featurized_statcast_data.parquet")


if __name__ == "__main__":
    df = pl.read_parquet("data/clean_statcast_data.parquet")
    featurize(df)
