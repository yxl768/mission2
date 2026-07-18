"""Build a small, real ZINC IUPAC/SMILES subset through PubChem PUG REST.

The script is deliberately rate-limited and resumable.  It does not invent
names: every retained IUPAC name is returned by PubChem for the input SMILES.
"""

import argparse
import time
from pathlib import Path
import pandas as pd
import requests


PUG_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/IUPACName/JSON"


def lookup_iupac(session, smiles, retries=2):
    for attempt in range(retries + 1):
        try:
            # POST avoids URL parsing failures for stereochemistry and brackets.
            response = session.post(PUG_URL, data={"smiles": smiles}, timeout=20)
            if response.status_code == 200:
                properties = response.json().get("PropertyTable", {}).get("Properties", [])
                if properties and properties[0].get("IUPACName"):
                    return properties[0]["IUPACName"]
            if response.status_code not in {429, 503}:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Fetch authentic IUPAC names for a small ZINC subset")
    parser.add_argument("--input", default="data/zinc250k.csv")
    parser.add_argument("--output", default="data/zinc_iupac_subset.csv")
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between PubChem requests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    if "smiles" not in source.columns:
        raise ValueError("Input must contain a smiles column.")
    output_path = Path(args.output)
    if output_path.exists():
        try:
            records = pd.read_csv(output_path).to_dict("records")
        except pd.errors.EmptyDataError:
            records = []
    else:
        records = []
    known_smiles = {record["smiles"] for record in records}
    candidates = source.loc[~source["smiles"].isin(known_smiles)].sample(frac=1, random_state=args.seed)
    max_attempts = args.max_attempts or args.target_size * 4
    session = requests.Session()

    attempts = 0
    for _, row in candidates.iterrows():
        if len(records) >= args.target_size or attempts >= max_attempts:
            break
        smiles = row["smiles"]
        iupac = lookup_iupac(session, smiles)
        attempts += 1
        if iupac:
            record = row.to_dict()
            record["iupac"] = iupac
            records.append(record)
        if attempts % 25 == 0:
            pd.DataFrame(records, columns=[*source.columns, "iupac"]).to_csv(output_path, index=False)
            print(f"attempts={attempts}, matched={len(records)}")
        time.sleep(args.delay)

    pd.DataFrame(records, columns=[*source.columns, "iupac"]).to_csv(output_path, index=False)
    print(f"Saved {len(records)} real ZINC/PubChem pairs to {output_path} after {attempts} requests.")


if __name__ == "__main__":
    main()
