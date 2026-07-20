"""Augment the IUPAC->SELFIES dataset using SMILES randomization.

For each molecule, generate multiple random SMILES variants, convert each to
SELFIES, and pair with the original IUPAC name. This expands the training
data without requiring additional IUPAC labels.
"""

import argparse
import os

import pandas as pd

from seq2seq_mol.data_utils import (
    IUPAC_COLUMN,
    SELFIES_COLUMN,
    SMILES_COLUMN,
    smiles_to_selfies,
)


def generate_random_smiles(smiles, n_variants, seed=None):
    """Generate n_variants random SMILES for the same molecule."""
    try:
        from rdkit import Chem
    except ImportError as error:
        raise RuntimeError("RDKit is required for SMILES augmentation.") from error

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    variants = set()
    if seed is not None:
        import random
        random.seed(seed)
    max_attempts = n_variants * 5
    for _ in range(max_attempts):
        if len(variants) >= n_variants:
            break
        random_smiles = Chem.MolToSmiles(mol, doRandom=True)
        if random_smiles and random_smiles not in variants:
            variants.add(random_smiles)
    return list(variants)


def augment_dataset(input_path, output_path, n_variants=5, seed=42):
    """Augment dataset by generating random SMILES variants."""
    frame = pd.read_csv(input_path)
    augmented_records = []
    for _, row in frame.iterrows():
        smiles = str(row[SMILES_COLUMN]).strip()
        iupac = str(row[IUPAC_COLUMN]).strip()
        # Keep original
        original_selfies = smiles_to_selfies(smiles)
        if original_selfies:
            record = row.to_dict()
            record[SELFIES_COLUMN] = original_selfies
            augmented_records.append(record)
        # Generate variants
        variants = generate_random_smiles(smiles, n_variants, seed)
        for variant_smiles in variants:
            if variant_smiles == smiles:
                continue
            variant_selfies = smiles_to_selfies(variant_smiles)
            if variant_selfies:
                record = row.to_dict()
                record[SMILES_COLUMN] = variant_smiles
                record[SELFIES_COLUMN] = variant_selfies
                augmented_records.append(record)

    augmented_frame = pd.DataFrame(augmented_records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    augmented_frame.to_csv(output_path, index=False)
    print(f"Augmented {len(frame)} molecules to {len(augmented_frame)} records")
    print(f"Saved to {output_path}")
    return augmented_frame


def main():
    parser = argparse.ArgumentParser(description="Augment IUPAC->SELFIES dataset with SMILES randomization")
    parser.add_argument("--input", required=True, help="Input CSV with SMILES, IUPAC columns")
    parser.add_argument("--output", required=True, help="Output augmented CSV path")
    parser.add_argument("--n-variants", type=int, default=5, help="Number of random SMILES variants per molecule")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    augment_dataset(args.input, args.output, args.n_variants, args.seed)


if __name__ == "__main__":
    main()