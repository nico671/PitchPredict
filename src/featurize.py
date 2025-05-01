import os

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "data/featurized"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEQ_LEN = 10
feature_cols = [
    "pitch_type_code",
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


def featurize_for_single_player(
    df: pl.DataFrame,
    player_name: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    player_df = df.filter(pl.col("player_name") == player_name)
    player_df = player_df.sort(
        ["game_pk", "game_date", "at_bat_number", "pitch_number"], descending=False
    )
    player_df = player_df.with_columns(
        [
            pl.col("pitch_type")
            .cast(pl.Categorical)
            .to_physical()
            .alias("pitch_type_code"),
        ]
    )

    # 2) Create next‐pitch target (still as string), then encode it too:
    player_df = player_df.with_columns(
        pl.col("pitch_type_code")
        .shift(-1)
        .over("game_pk")
        .alias("next_pitch_type_code")
    ).drop_nulls("next_pitch_type_code")

    # Context features
    player_df = player_df.with_columns(
        [
            (pl.col("balls") * 3 + pl.col("strikes")).alias("count_state"),
            (pl.col("bat_score") - pl.col("fld_score")).alias("score_diff"),
            (pl.col("stand") == pl.col("p_throws")).cast(pl.Int8).alias("same_hand"),
        ]
    )

    # Get game_pk for each row
    game_pks = player_df.select("game_pk").to_numpy().flatten()

    X_raw = player_df.select(feature_cols).to_numpy()
    y_raw = player_df.select("next_pitch_type_code").to_numpy()

    Xs, ys = [], []
    for i in range(SEQ_LEN, len(X_raw)):
        # Check if all pitches in the sequence belong to the same game
        if len(set(game_pks[i - SEQ_LEN : i])) == 1:
            Xs.append(X_raw[i - SEQ_LEN : i])
            ys.append(y_raw[i])

    X = np.stack(Xs)
    y = np.array(ys)

    # Use time-series split to avoid data leakage
    # Earlier games should be in training, later games in validation/test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, shuffle=False
    )

    return X_tr, y_tr, X_val, y_val, X_te, y_te


if __name__ == "__main__":
    full_df_clean = pl.read_parquet("data/clean_statcast_data.parquet")
    players = full_df_clean.select(pl.col("player_name").unique()).to_series().to_list()
    all_data = {}
    for player_id in players:
        X_tr, y_tr, X_val, y_val, X_te, y_te = featurize_for_single_player(
            full_df_clean, player_id
        )
        all_data[player_id] = {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_te,
            "y_test": y_te,
        }
    # Save the data
    for player_id, splits in all_data.items():
        out_path = os.path.join(OUTPUT_DIR, f"{player_id}_data.npz")
        np.savez_compressed(
            out_path,
            X_train=splits["X_train"],
            y_train=splits["y_train"],
            X_val=splits["X_val"],
            y_val=splits["y_val"],
            X_test=splits["X_test"],
            y_test=splits["y_test"],
        )
        print(f"Saved {player_id} → {out_path}")
