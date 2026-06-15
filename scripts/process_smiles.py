import pandas as pd
from rdkit import Chem

def main():
	smiles = set()
	df = pd.read_csv("data/smiles_list.csv")
	for smi in df["SMILES"]:
		mol = Chem.MolFromSmiles(smi)
		if mol is None:
			print(f"Error with {smi}. Mol not found")
			smiles.add(str(smi))
			continue
		can_smi = Chem.MolToSmiles(mol, canonical=True)
		smiles.add(str(can_smi))
	with open("output/fine_tuned_smiles.smi", "w", encoding="utf-8") as file:
		for smi in smiles:
			file.write(smi + "\n")


if __name__ == "__main__":
    main()
