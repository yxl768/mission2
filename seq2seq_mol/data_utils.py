import os
from typing import Dict, Iterable, List, Optional

import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    import selfies
except ImportError as error:  # Keep training usable when only preprocessing dependencies are absent.
    selfies = None
    SELFIES_IMPORT_ERROR = error
else:
    SELFIES_IMPORT_ERROR = None

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError as error:
    Chem = None
    Descriptors = None
    RDKIT_IMPORT_ERROR = error
else:
    RDKIT_IMPORT_ERROR = None


SELFIES_COLUMN = "selfies"
IUPAC_COLUMN = "iupac"
SMILES_COLUMN = "smiles"


def _require_selfies() -> None:
    if SELFIES_IMPORT_ERROR is not None:
        raise RuntimeError("selfies is required for dataset construction.") from SELFIES_IMPORT_ERROR


def _require_rdkit() -> None:
    if RDKIT_IMPORT_ERROR is not None:
        raise RuntimeError("RDKit is required for dataset construction and descriptors.") from RDKIT_IMPORT_ERROR


def smiles_to_selfies(smiles: str) -> Optional[str]:
    _require_selfies()
    try:
        selfies_seq = selfies.encoder(smiles)
        return selfies_seq
    except Exception:
        return None


def get_mol_from_smiles(smiles: str):
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    return mol


def compute_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    _require_rdkit()
    mol = get_mol_from_smiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
    }


def load_smiles_source(
    input_path: str,
    smiles_column: str = SMILES_COLUMN,
    n_samples: Optional[int] = None,
) -> List[str]:
    ext = os.path.splitext(input_path)[1].lower()
    smiles: List[str] = []

    if ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        frame = pd.read_csv(input_path, sep=sep, usecols=[smiles_column])
        smiles = frame[smiles_column].dropna().astype(str).tolist()
    else:
        with open(input_path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip().split()
                if not row:
                    continue
                smiles.append(row[0])
    if n_samples is not None:
        smiles = smiles[: n_samples]
    return smiles


def load_translation_source(
    input_path: str,
    smiles_column: str = SMILES_COLUMN,
    iupac_column: str = IUPAC_COLUMN,
    n_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Load paired SMILES/IUPAC data.

    RDKit deliberately does not provide SMILES-to-IUPAC naming.  A paired source
    (for example a PubChem/ZINC export augmented with names) is therefore
    required rather than silently producing an empty training set.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in {".csv", ".tsv"}:
        raise ValueError("IUPAC translation data must be CSV/TSV with SMILES and IUPAC columns.")
    sep = "," if ext == ".csv" else "\t"
    frame = pd.read_csv(input_path, sep=sep)
    missing = {smiles_column, iupac_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(sorted(missing))}")
    frame = frame[[smiles_column, iupac_column]].rename(
        columns={smiles_column: SMILES_COLUMN, iupac_column: IUPAC_COLUMN}
    )
    frame = frame.dropna().astype(str)
    if n_samples is not None:
        frame = frame.head(n_samples)
    return frame.reset_index(drop=True)


def build_translation_dataset(
    molecules: Iterable,
    save_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    progress: bool = True,
) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    iterator = list(molecules)
    if max_samples is not None:
        iterator = iterator[:max_samples]

    for item in tqdm(iterator, desc="Converting molecules", disable=not progress):
        if isinstance(item, dict):
            smiles, iupac = item.get(SMILES_COLUMN), item.get(IUPAC_COLUMN)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            smiles, iupac = item[0], item[1]
        else:
            raise ValueError("Each molecule must provide both SMILES and an IUPAC name.")
        if not isinstance(smiles, str) or not isinstance(iupac, str):
            continue
        smiles, iupac = smiles.strip(), iupac.strip()
        selfies_seq = smiles_to_selfies(smiles)
        if not smiles or not iupac or selfies_seq is None:
            continue
        records.append({SMILES_COLUMN: smiles, IUPAC_COLUMN: iupac, SELFIES_COLUMN: selfies_seq})

    frame = pd.DataFrame(records)
    if save_path is not None:
        frame.to_csv(save_path, index=False)
    return frame


def enrich_with_descriptors(frame: pd.DataFrame, smiles_column: str = SMILES_COLUMN) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for smiles in tqdm(frame[smiles_column].tolist(), desc="Computing descriptors"):
        desc = compute_descriptors(smiles)
        if desc is None:
            desc = {"MolWt": float("nan"), "LogP": float("nan"), "TPSA": float("nan"), "NumHDonors": float("nan"), "NumHAcceptors": float("nan"), "NumRotatableBonds": float("nan")}
        records.append(desc)
    desc_frame = pd.DataFrame(records)
    result = pd.concat([frame.reset_index(drop=True), desc_frame.reset_index(drop=True)], axis=1)
    return result


def save_dataframe(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path, index=False)


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
