"""Compare learned encoder embeddings with RDKit descriptor and Morgan fingerprint baselines."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from seq2seq_mol.data_utils import IUPAC_COLUMN, SMILES_COLUMN, enrich_with_descriptors, load_dataframe


DEFAULT_DESCRIPTORS = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"]
MORGAN_RADIUS = 2
MORGAN_BITS = 1024

PROPERTY_ALIASES = {
    "logp": "logP",
    "qed": "qed",
    "sas": "SAS",
}


def resolve_property_name(frame, property_name):
    """Resolve a property name case-insensitively against the dataframe columns."""
    columns = {c.lower(): c for c in frame.columns}
    # Direct match
    if property_name in frame.columns:
        return property_name
    # Case-insensitive match
    if property_name.lower() in columns:
        return columns[property_name.lower()]
    # Alias match
    alias = PROPERTY_ALIASES.get(property_name.lower())
    if alias and alias in frame.columns:
        return alias
    raise ValueError(
        f"Property '{property_name}' not found in columns: {list(frame.columns)}"
    )


def deduplicate_by_iupac(frame, embeddings):
    """Deduplicate the dataframe and embeddings by IUPAC name.

    The augmented dataset contains multiple SMILES variants of the same molecule
    (same IUPAC name). Random train/test splits leak the same molecule into both
    sets, inflating downstream metrics. Since the encoder receives the IUPAC name
    as input, all variants of the same molecule share the same embedding, so we
    keep only the first occurrence of each unique IUPAC name.
    """
    if IUPAC_COLUMN not in frame.columns:
        return frame, embeddings
    unique_mask = ~frame[IUPAC_COLUMN].astype(str).duplicated(keep="first")
    n_before = len(frame)
    frame = frame.loc[unique_mask].reset_index(drop=True)
    embeddings = embeddings[unique_mask.to_numpy()]
    n_after = len(frame)
    if n_before != n_after:
        print(f"Deduplicated by IUPAC: {n_before} -> {n_after} unique molecules")
    return frame, embeddings


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except ImportError as error:
        raise RuntimeError("RDKit is required for Morgan fingerprint computation.") from error


def compute_morgan_fingerprints(smiles_list, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    """Compute Morgan fingerprints (ECFP) for a list of SMILES strings."""
    Chem, AllChem = _require_rdkit()
    fingerprints = np.zeros((len(smiles_list), n_bits), dtype=np.float64)
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprints[idx] = np.array(bit_vector)
    return fingerprints


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


def kfold_eval(features, target, alpha, n_splits=5, seed=42):
    """Run k-fold cross-validation and return mean/std metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    metrics_list = []
    for train_idx, test_idx in kf.split(features):
        metrics, _, _ = fit_and_score(features, target, train_idx, test_idx, alpha)
        metrics_list.append(metrics)
    mean_metrics = {
        key: float(np.mean([m[key] for m in metrics_list]))
        for key in metrics_list[0]
    }
    std_metrics = {
        key: float(np.std([m[key] for m in metrics_list]))
        for key in metrics_list[0]
    }
    return mean_metrics, std_metrics, metrics_list


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
    n_panels = len(results)
    figure, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    for axis, (name, (truth, prediction, metrics)) in zip(axes, results.items()):
        lower, upper = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        axis.scatter(truth, prediction, s=14, alpha=0.7, color="#267e8c")
        axis.plot([lower, upper], [lower, upper], color="#bf4d35", linewidth=1)
        axis.set(title=f"{name}: R2={metrics['r2']:.3f}", xlabel="Observed", ylabel="Predicted")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def compare_representations(data_path, embeddings_path, output_dir, property_name="LogP",
                            test_fraction=0.2, seed=42, alpha=1.0, n_folds=5, deduplicate=True):
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    os.makedirs(output_dir, exist_ok=True)
    frame = load_dataframe(data_path)
    property_name = resolve_property_name(frame, property_name)
    if property_name not in frame.columns or frame[property_name].isna().any():
        frame = enrich_with_descriptors(frame)
        property_name = resolve_property_name(frame, property_name)
    required = DEFAULT_DESCRIPTORS + [property_name, SMILES_COLUMN]
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
    if deduplicate:
        frame, embeddings = deduplicate_by_iupac(frame, embeddings)

    descriptor_names = [name for name in DEFAULT_DESCRIPTORS if name != property_name]
    descriptors = frame[descriptor_names].to_numpy(dtype=np.float64)
    smiles_list = frame[SMILES_COLUMN].astype(str).tolist()
    morgan_fps = compute_morgan_fingerprints(smiles_list)
    target = frame[property_name].to_numpy(dtype=np.float64)

    # Single split evaluation (for visualization)
    train_indices, test_indices = train_test_split(
        np.arange(len(frame)), test_size=test_fraction, random_state=seed
    )
    desc_metrics, desc_truth, desc_prediction = fit_and_score(descriptors, target, train_indices, test_indices, alpha)
    morgan_metrics, morgan_truth, morgan_prediction = fit_and_score(morgan_fps, target, train_indices, test_indices, alpha)
    emb_metrics, emb_truth, emb_prediction = fit_and_score(embeddings, target, train_indices, test_indices, alpha)

    # k-fold cross-validation
    print(f"\n=== {n_folds}-Fold Cross-Validation for {property_name} ===")
    desc_mean, desc_std, _ = kfold_eval(descriptors, target, alpha, n_folds, seed)
    morgan_mean, morgan_std, _ = kfold_eval(morgan_fps, target, alpha, n_folds, seed)
    emb_mean, emb_std, _ = kfold_eval(embeddings, target, alpha, n_folds, seed)

    summary = pd.DataFrame([
        {"representation": "RDKit descriptors", **desc_metrics,
         "kfold_r2_mean": desc_mean["r2"], "kfold_r2_std": desc_std["r2"],
         "kfold_rmse_mean": desc_mean["rmse"], "kfold_rmse_std": desc_std["rmse"]},
        {"representation": "Morgan fingerprint (ECFP4)", **morgan_metrics,
         "kfold_r2_mean": morgan_mean["r2"], "kfold_r2_std": morgan_std["r2"],
         "kfold_rmse_mean": morgan_mean["rmse"], "kfold_rmse_std": morgan_std["rmse"]},
        {"representation": "Seq2Seq encoder embedding", **emb_metrics,
         "kfold_r2_mean": emb_mean["r2"], "kfold_r2_std": emb_std["r2"],
         "kfold_rmse_mean": emb_mean["rmse"], "kfold_rmse_std": emb_std["rmse"]},
    ])
    summary.to_csv(os.path.join(output_dir, "property_prediction_summary.csv"), index=False)
    with open(os.path.join(output_dir, "property_prediction_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, indent=2)

    print("\n=== Single Split Results ===")
    print(summary[["representation", "r2", "rmse", "mae"]].to_string(index=False))
    print("\n=== K-Fold Cross-Validation (mean ± std) ===")
    for _, row in summary.iterrows():
        print(f"{row['representation']}: R2={row['kfold_r2_mean']:.4f}±{row['kfold_r2_std']:.4f}, "
              f"RMSE={row['kfold_rmse_mean']:.4f}±{row['kfold_rmse_std']:.4f}")

    rng = np.random.default_rng(seed)
    visual_indices = rng.choice(len(frame), size=min(len(frame), 5000), replace=False)
    plot_projection(descriptors, target, "RDKit descriptor space", os.path.join(output_dir, "descriptor_pca.png"), visual_indices)
    plot_projection(morgan_fps, target, "Morgan fingerprint space", os.path.join(output_dir, "morgan_pca.png"), visual_indices)
    plot_projection(embeddings, target, "Seq2Seq encoder embedding space", os.path.join(output_dir, "encoder_embedding_pca.png"), visual_indices)
    plot_predictions({
        "RDKit descriptors": (desc_truth, desc_prediction, desc_metrics),
        "Morgan fingerprint": (morgan_truth, morgan_prediction, morgan_metrics),
        "Encoder embeddings": (emb_truth, emb_prediction, emb_metrics),
    }, os.path.join(output_dir, "predicted_vs_observed.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate learned molecular representations")
    parser.add_argument("--data", required=True, help="Processed CSV used by the Seq2Seq trainer")
    parser.add_argument("--embeddings", required=True, help="encoder_embeddings.npy from the trainer")
    parser.add_argument("--output", default="eval_outputs")
    parser.add_argument("--property", default="LogP")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--no-deduplicate", action="store_true",
                        help="Disable deduplication by IUPAC name (leaves augmented data as-is)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_representations(
        args.data, args.embeddings, args.output, args.property,
        args.test_fraction, args.seed, args.ridge_alpha, args.n_folds,
        deduplicate=not args.no_deduplicate,
    )