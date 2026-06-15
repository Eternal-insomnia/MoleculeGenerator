from rdkit import Chem
from pathlib import Path
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# CONFIG
DATAPATH = Path.cwd() / 'output'
SMILES_FILES = [
    "rnn_test10k.smi",
    "rnn_fine_tuned10k.smi",
    "rnn_trained10k.smi"
]


def main():
    for filename in SMILES_FILES:
        print(f"Processing {filename}")
        valid = 0
        unique_smiles = set()
        for smiles in open(DATAPATH / filename):
            molecule = Chem.MolFromSmiles(smiles, sanitize=True)
            if molecule is not None:
                valid += 1
                can_smi = Chem.MolToSmiles(molecule, canonical=True)
                unique_smiles.add(str(can_smi))
            else:
                continue
        
        print(f"There are {valid} valid SMILES and {len(unique_smiles)} unique SMILES")


if __name__ == "__main__":
    main()
