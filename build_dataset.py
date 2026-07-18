import argparse
import os
from typing import Optional

import pandas as pd

from seq2seq_mol.data_utils import build_translation_dataset, enrich_with_descriptors, load_translation_source, save_dataframe


def build_dataset(
    input_path: str,
    output_path: str,
    max_samples: Optional[int] = None,
    with_descriptors: bool = False,
    smiles_column: str = "smiles",
    iupac_column: str = "iupac",
):
    source = load_translation_source(input_path, smiles_column, iupac_column, n_samples=max_samples)
    # Keep any existing ZINC property labels (for example logP/QED/SAS) while
    # adding SELFIES only for successfully converted paired molecules.
    raw_source = pd.read_csv(input_path, sep="\t" if input_path.lower().endswith(".tsv") else ",")
    raw_source = raw_source.rename(columns={smiles_column: "smiles", iupac_column: "iupac"}).dropna(subset=["smiles", "iupac"])
    raw_source["smiles"] = raw_source["smiles"].astype(str).str.strip()
    raw_source["iupac"] = raw_source["iupac"].astype(str).str.strip()
    if max_samples is not None:
        raw_source = raw_source.head(max_samples)
    translated = build_translation_dataset(source.itertuples(index=False, name=None), max_samples=max_samples)
    frame = raw_source.merge(translated[["smiles", "iupac", "selfies"]], on=["smiles", "iupac"], how="inner")
    if with_descriptors:
        frame = enrich_with_descriptors(frame)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_dataframe(frame, output_path)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Build IUPAC -> SELFIES translation dataset")
    parser.add_argument("--input", type=str, required=True, help="CSV/TSV containing paired SMILES and IUPAC columns")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--with-descriptors", action="store_true", help="Append RDKit descriptor columns")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--iupac-column", default="iupac")
    return parser.parse_args()


def main():
    args = parse_args()
    frame = build_dataset(
        args.input, args.output, args.max_samples, args.with_descriptors,
        args.smiles_column, args.iupac_column,
    )
    print(f"Saved {len(frame)} molecules to {args.output}")


if __name__ == "__main__":
    main()
