import pandas as pd
from rdkit import Chem

def main():
	smiles = set()
	df = pd.read_csv("data/smiles_list.csv")

	df_ic50 = df[df["Activity_type"] == "IC50"]

	for smi in df_ic50["SMILES"]:
		mol = Chem.MolFromSmiles(smi)
		if mol is None:
			print(f"Error with {smi}. Mol not found")
			smiles.add(str(smi))
			continue
		can_smi = Chem.MolToSmiles(mol, canonical=True)
		smiles.add(str(can_smi))
	with open("output/processed_smiles.smi", "w", encoding="utf-8") as file:
		for smi in smiles:
			file.write(smi + "\n")


if __name__ == "__main__":
    main()
