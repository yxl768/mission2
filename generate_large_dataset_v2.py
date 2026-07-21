"""Generate large-scale IUPAC→SELFIES dataset using chemical-converters."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import selfies
from chemicalconverters import NamesConverter
from seq2seq_mol.data_utils import get_mol_from_smiles


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def smiles_to_iupac_batch(smiles_list, converter, batch_size=100):
    """Convert SMILES to IUPAC names in batches."""
    iupac_names = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Converting SMILES→IUPAC"):
        batch = smiles_list[i:i + batch_size]
        try:
            results = converter.smiles_to_iupac(batch, process_in_batch=True, batch_size=batch_size)
            iupac_names.extend(results)
        except Exception as e:
            logger.warning("Batch conversion failed: %s", e)
            for s in batch:
                try:
                    iupac_names.append(converter.smiles_to_iupac(s))
                except:
                    iupac_names.append(None)
    return iupac_names


def smiles_to_canonical_selfies(smiles):
    """Convert SMILES to canonical SELFIES."""
    mol = get_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return selfies.encoder(smiles)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate large-scale IUPAC→SELFIES dataset")
    parser.add_argument("--input", required=True, help="Input SMILES CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum molecules to process")
    args = parser.parse_args()

    logger.info("Loading input data from %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("Loaded %d molecules", len(df))

    df = df.head(args.limit)
    logger.info("Processing first %d molecules", len(df))

    logger.info("Initializing chemical-converters...")
    converter = NamesConverter(model_name="knowledgator/SMILES2IUPAC-canonical-base")

    smiles_list = df["smiles"].astype(str).str.strip().tolist()
    
    logger.info("Converting SMILES to IUPAC names...")
    iupac_names = smiles_to_iupac_batch(smiles_list, converter)

    logger.info("Converting SMILES to SELFIES...")
    selfies_list = []
    for smiles in tqdm(smiles_list, desc="Converting SMILES→SELFIES"):
        selfies_list.append(smiles_to_canonical_selfies(smiles))

    results = []
    for idx, (smiles, iupac, selfies_seq) in enumerate(zip(smiles_list, iupac_names, selfies_list)):
        if iupac and selfies_seq and smiles:
            results.append({
                "smiles": smiles,
                "iupac": iupac,
                "selfies": selfies_seq,
                "logP": df.iloc[idx].get("logP", df.iloc[idx].get("LogP", "")),
                "qed": df.iloc[idx].get("qed", df.iloc[idx].get("QED", "")),
                "SAS": df.iloc[idx].get("SAS", df.iloc[idx].get("sas", "")),
            })

    logger.info("Successfully processed %d/%d molecules", len(results), len(df))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    logger.info("Saved dataset to %s", output_path)
    logger.info("Dataset statistics:")
    logger.info("  Total molecules: %d", len(result_df))
    logger.info("  Average IUPAC length: %.1f", result_df["iupac"].str.len().mean())
    logger.info("  Average SELFIES length: %.1f", result_df["selfies"].str.len().mean())


if __name__ == "__main__":
    main()
