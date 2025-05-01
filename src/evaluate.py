import json
import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
FEAT_DIR = "data/featurized"
MODEL_DIR = "models"
EVAL_ROOT = "models/eval"
METRICS_DIR = "metrics"
os.makedirs(EVAL_ROOT, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

if __name__ == "__main__":
    df_clean = pl.read_parquet("data/clean_statcast_data.parquet")

    # Create a mapping of unique pitch types
    # First create the categorical mapping (same as in featurize.py)
    df_with_code = df_clean.with_columns(
        pl.col("pitch_type").cast(pl.Categorical).to_physical().alias("pitch_type_code")
    )

    # Create the mapping for pitch codes to names
    # Sort the unique values numerically instead of trying to sort by column name
    pitch_codes = (
        df_with_code.select("pitch_type_code").unique().to_series().sort().to_list()
    )
    pitch_types = [
        df_with_code.filter(pl.col("pitch_type_code") == code)
        .select("pitch_type")
        .item(0, 0)
        for code in pitch_codes
    ]
    pitch_type_mapping = dict(zip(pitch_codes, pitch_types))

    # Dictionary to collect metrics from all players for aggregation
    all_players_metrics = {}

    for fname in os.listdir(FEAT_DIR):
        if not fname.endswith("_data.npz"):
            continue
        player_name = fname.replace("_data.npz", "")

        # 1) Load data
        data = np.load(os.path.join(FEAT_DIR, fname))
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Load the player-specific class mapping
        history_file = os.path.join(MODEL_DIR, f"{player_name}_lstm_history.npz")
        history_data = np.load(history_file, allow_pickle=True)
        player_classes = history_data["classes"]

        # 2) Load model & predict
        model = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, f"{player_name}_lstm_model.keras")
        )
        y_prob = model.predict(X_test, verbose=0)  # shape (N, C_player)
        y_pred = np.argmax(y_prob, axis=1)

        # 3) Use player-specific class mapping
        num_classes_player = len(player_classes)
        labels = list(range(num_classes_player))

        # Map numeric classes to actual pitch types
        class_names_player = [
            pitch_type_mapping.get(int(c), f"Unknown-{c}") for c in player_classes
        ]

        # 4) Compute metrics
        acc = accuracy_score(y_test, y_pred)
        top2 = np.argsort(y_prob, axis=1)[:, -2:]
        top2_acc = np.mean([y_test[i] in top2[i] for i in range(len(y_test))])
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=class_names_player,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Store metrics for this player in the aggregated dictionary
        all_players_metrics[player_name] = {
            "accuracy": float(acc),
            "top2_accuracy": float(top2_acc),
            "num_classes": num_classes_player,
            "class_distribution": {
                class_name: int(np.sum(y_test.flatten() == i))
                for i, class_name in enumerate(class_names_player)
            },
        }

        # 5) Save under player‐specific folder
        player_dir = os.path.join(EVAL_ROOT, player_name)
        os.makedirs(player_dir, exist_ok=True)
        with open(os.path.join(player_dir, "metrics.json"), "w") as fp:
            json.dump({"accuracy": acc, "top2_accuracy": top2_acc}, fp, indent=2)
        with open(os.path.join(player_dir, "classification_report.json"), "w") as fp:
            json.dump(report_dict, fp, indent=2)

        # 6) Confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(class_names_player, rotation=45, ha="right")
        ax.set_yticklabels(class_names_player)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{player_name} Confusion Matrix")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(player_dir, "confusion.png"))
        plt.close(fig)

        print(
            f"✓ Evaluated {player_name}: {num_classes_player} classes, acc={acc:.3f}, top2={top2_acc:.3f}"
        )

    # Add aggregate stats
    all_players_metrics["summary"] = {
        "avg_accuracy": np.mean(
            [m["accuracy"] for m in all_players_metrics.values() if "accuracy" in m]
        ),
        "avg_top2_accuracy": np.mean(
            [
                m["top2_accuracy"]
                for m in all_players_metrics.values()
                if "top2_accuracy" in m
            ]
        ),
        "total_players": len([k for k in all_players_metrics.keys() if k != "summary"]),
    }

    # Save the aggregated metrics file that DVC expects
    with open(os.path.join(METRICS_DIR, "evaluation.json"), "w") as f:
        json.dump(all_players_metrics, f, indent=2)

    print(f"\nSaved aggregated metrics to {METRICS_DIR}/evaluation.json")
