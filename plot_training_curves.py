import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(gru_log_path, transformer_log_path, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if gru_log_path and os.path.exists(gru_log_path):
        gru_df = pd.read_csv(gru_log_path)
        axes[0].plot(gru_df["epoch"], gru_df["train_loss"], label="GRU Train", color="#267e8c")
        axes[0].plot(gru_df["epoch"], gru_df["validation_loss"], label="GRU Validation", color="#bf4d35", linestyle="--")

    if transformer_log_path and os.path.exists(transformer_log_path):
        tf_df = pd.read_csv(transformer_log_path)
        axes[0].plot(tf_df["epoch"], tf_df["train_loss"], label="Transformer Train", color="#3a923a")
        axes[0].plot(tf_df["epoch"], tf_df["validation_loss"], label="Transformer Validation", color="#923a92", linestyle="--")

    axes[0].set_title("Training Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if gru_log_path and os.path.exists(gru_log_path) and transformer_log_path and os.path.exists(transformer_log_path):
        min_epochs = min(len(gru_df), len(tf_df))
        axes[1].plot(gru_df["epoch"][:min_epochs], gru_df["validation_loss"][:min_epochs], label="GRU", color="#bf4d35")
        axes[1].plot(tf_df["epoch"][:min_epochs], tf_df["validation_loss"][:min_epochs], label="Transformer", color="#923a92")
        axes[1].set_title("Validation Loss Comparison")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved training curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves for Seq2Seq models")
    parser.add_argument("--gru-log", default="outputs/zinc_gru_bidir/training_log.csv", help="GRU training log path")
    parser.add_argument("--transformer-log", default="outputs/zinc_transformer/training_log.csv", help="Transformer training log path")
    parser.add_argument("--output", default="outputs/training_curves.png", help="Output plot path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plot_curves(args.gru_log, args.transformer_log, args.output)


if __name__ == "__main__":
    main()