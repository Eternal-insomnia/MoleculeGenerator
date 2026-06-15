from rdkit import Chem
from pathlib import Path

# CONFIG
DATAPATH = Path.cwd() / 'data'
SMILES_FILE = "rnn_fine_tuned10k.smi"
SMILES_COUNT = 10000


def main():
    valid = 0
    for smiles in open(DATAPATH / SMILES_FILE):
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
        if molecule is not None:
            valid += 1
        else:
            continue

    print(f"From {SMILES_COUNT} there are {valid} valid SMILES")


if __name__ == "__main__":
    main()
