"""Compare learned encoder embeddings with RDKit descriptor baselines."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seq2seq_mol.data_utils import enrich_with_descriptors, load_dataframe


DEFAULT_DESCRIPTORS = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]


def fit_and_score(features, target, train_indices, test_indices, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features[train_indices], target[train_indices])
    predicted = model.predict(features[test_indices])
    truth = target[test_indices]
    return {
        "r2": float(r2_score(truth, predicted)),
        "rmse": float(mean_squared_error(truth, predicted) ** 0.5),
        "mae": float(mean_absolute_error(truth, predicted)),
    }, truth, predicted


def plot_projection(features, target, title, path, sample_indices):
    projected = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(features[sample_indices]))
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(projected[:, 0], projected[:, 1], c=target[sample_indices], cmap="viridis", s=10, alpha=0.75)
    figure.colorbar(scatter, ax=axis, label="Property value")
    axis.set(title=title, xlabel="PC1", ylabel="PC2")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_predictions(results, path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (name, (truth, prediction, metrics)) in zip(axes, results.items()):
        lower, upper = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        axis.scatter(truth, prediction, s=14, alpha=0.7, color="#267e8c")
        axis.plot([lower, upper], [lower, upper], color="#bf4d35", linewidth=1)
        axis.set(title=f"{name}: R2={metrics['r2']:.3f}", xlabel="Observed", ylabel="Predicted")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def compare_representations(data_path, embeddings_path, output_dir, property_name="LogP", test_fraction=0.2, seed=42, alpha=1.0):
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    os.makedirs(output_dir, exist_ok=True)
    frame = load_dataframe(data_path)
    if property_name not in frame.columns or frame[property_name].isna().any():
        frame = enrich_with_descriptors(frame)
    required = DEFAULT_DESCRIPTORS + [property_name]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Unable to obtain requested property/descriptor columns: {sorted(missing)}")
    frame = frame.dropna(subset=required).reset_index(drop=True)
    embeddings = np.load(embeddings_path)
    if len(embeddings) != len(frame):
        raise ValueError(
            f"Embedding/data row mismatch: {len(embeddings)} embeddings for {len(frame)} usable molecules. "
            "Use the processed dataset that was supplied to training."
        )
    descriptor_names = [name for name in DEFAULT_DESCRIPTORS if name != property_name]
    descriptors = frame[descriptor_names].to_numpy(dtype=np.float64)
    target = frame[property_name].to_numpy(dtype=np.float64)
    train_indices, test_indices = train_test_split(
        np.arange(len(frame)), test_size=test_fraction, random_state=seed
    )
    descriptor_metrics, desc_truth, desc_prediction = fit_and_score(descriptors, target, train_indices, test_indices, alpha)
    embedding_metrics, emb_truth, emb_prediction = fit_and_score(embeddings, target, train_indices, test_indices, alpha)
    summary = pd.DataFrame([
        {"representation": "RDKit descriptors", **descriptor_metrics},
        {"representation": "Seq2Seq encoder embedding", **embedding_metrics},
    ])
    summary.to_csv(os.path.join(output_dir, "property_prediction_summary.csv"), index=False)
    with open(os.path.join(output_dir, "property_prediction_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, indent=2)

    rng = np.random.default_rng(seed)
    visual_indices = rng.choice(len(frame), size=min(len(frame), 5000), replace=False)
    plot_projection(descriptors, target, "RDKit descriptor space", os.path.join(output_dir, "descriptor_pca.png"), visual_indices)
    plot_projection(embeddings, target, "Seq2Seq encoder embedding space", os.path.join(output_dir, "encoder_embedding_pca.png"), visual_indices)
    plot_predictions({
        "RDKit descriptors": (desc_truth, desc_prediction, descriptor_metrics),
        "Encoder embeddings": (emb_truth, emb_prediction, embedding_metrics),
    }, os.path.join(output_dir, "predicted_vs_observed.png"))
    print(summary.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate learned molecular representations")
    parser.add_argument("--data", required=True, help="Processed CSV used by the Seq2Seq trainer")
    parser.add_argument("--embeddings", required=True, help="encoder_embeddings.npy from the trainer")
    parser.add_argument("--output", default="eval_outputs")
    parser.add_argument("--property", default="LogP")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_representations(
        args.data, args.embeddings, args.output, args.property,
        args.test_fraction, args.seed, args.ridge_alpha,
    )
