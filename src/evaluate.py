import logging
import os
from pathlib import Path
import pickle
from statistics import mean
import sys
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


logger = logging.getLogger("evaluate")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)
    file_name = sys.argv[1]
    with open(file_name, "rb") as f:
        pitcher_data = pickle.load(f)
    label_encoder = LabelEncoder()
    all_labels = [pitcher_data[pitcher]["y_test"] for pitcher in pitcher_data]
    label_encoder.fit(np.concatenate(all_labels))
    accuracy = []
    accuracy_diff = []
    for pitcher in pitcher_data:
        logger.info(f'Pitcher: {pitcher_data[pitcher]["player_name"]}')
        logger.info(
            f'Test Loss: {pitcher_data[pitcher]["test_loss"]:.4f}, Test Accuracy: {pitcher_data[pitcher]["test_accuracy"]:.4f}'
        )
        logger.info(
            f'Total Pitches: {pitcher_data[pitcher]["total_pitches"]}, Unique Classes: {pitcher_data[pitcher]["unique_classes"]}'
        )
        logger.info(
            f'Most common pitch rate: {pitcher_data[pitcher]["most_common_pitch_rate"]:.4f}'
        )
        logger.info(
            f"Average Performance Gained over just guessing the most common pitch: {pitcher_data[pitcher]['performance_gain']:.2f}"
        )
        accuracy_diff.append(pitcher_data[pitcher]["performance_gain"])
        accuracy.append(pitcher_data[pitcher]["test_accuracy"])
        plt.plot(pitcher_data[pitcher]["history"].history["loss"])
        plt.plot(pitcher_data[pitcher]["history"].history["val_loss"])
        plt.title(
            "model train vs validation loss for " + pitcher_data[pitcher]["player_name"]
        )
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper right")
        # Ensure the directory exists
        output_dir = Path("data/outputs/loss_plots/")
        if os.path.isfile(
            os.path.join(output_dir, f"{pitcher_data[pitcher]['player_name']}_loss.png")
        ):
            os.remove(
                os.path.join(
                    output_dir, f"{pitcher_data[pitcher]['player_name']}_loss.png"
                )
            )  # Opt.: os.system("rm "+strFile)
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(
            os.path.join(output_dir, f"{pitcher_data[pitcher]['player_name']}_loss.png")
        )
        plt.close()

        plt.plot(
            pitcher_data[pitcher]["history"].history["sparse_categorical_accuracy"]
        )
        plt.plot(
            pitcher_data[pitcher]["history"].history["val_sparse_categorical_accuracy"]
        )
        plt.title(
            "model train vs validation accuracy for "
            + pitcher_data[pitcher]["player_name"]
        )
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper right")
        # Ensure the directory exists
        output_dir = Path("data/outputs/accuracy_plots/")
        if os.path.isfile(
            os.path.join(
                output_dir, f"{pitcher_data[pitcher]['player_name']}_accuracy.png"
            )
        ):
            os.remove(
                os.path.join(
                    output_dir, f"{pitcher_data[pitcher]['player_name']}_accuracy.png"
                )
            )  # Opt.: os.system("rm "+strFile)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir, f"{pitcher_data[pitcher]['player_name']}_accuracy.png"
            )
        )
        plt.close()

        confusion_matrix = tf.math.confusion_matrix(
            pitcher_data[pitcher]["y_test"],
            np.argmax(
                pitcher_data[pitcher]["model"].predict(pitcher_data[pitcher]["X_test"]),
                axis=1,
            ),
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f'Confusion Matrix for {pitcher_data[pitcher]["player_name"]}')
        output_dir = Path("data/outputs/confusion_mat/")
        if os.path.isfile(
            os.path.join(
                output_dir, f"{pitcher_data[pitcher]['player_name']}_confusion.png"
            )
        ):
            os.remove(
                os.path.join(
                    output_dir, f"{pitcher_data[pitcher]['player_name']}_confusion.png"
                )
            )  # Opt.: os.system("rm "+strFile)
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(
            os.path.join(
                output_dir, f"{pitcher_data[pitcher]['player_name']}_confusion.png"
            )
        )
        plt.close()
    if len(accuracy_diff) > 1:
        logger.info(
            f"Average Performance Gained over just guessing the most common pitch across all pitchers: {mean(accuracy_diff):.4f}",
        )
    logger.info(f"Average Test Accuracy across all pitchers: {mean(accuracy):.4f}")


if __name__ == "__main__":
    main()
