from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import RDLogger
from tqdm import tqdm
from typing import List, Tuple, Optional

# Отключаем все логи RDKit (включая ошибки парсинга SMILES)
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)


def compute_properties_for_smiles_list(smiles_list: List[str], reference_fps: Optional[List] = None) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Computes MW, QED, LogP, and FCD (avg Tanimoto similarity to reference set or self).
    
    - FCD: if reference_fps is None → self-similarity (avg similarity to other molecules in list)
           if reference_fps is provided → similarity to that fixed set (more stable for comparisons)
    """
    morg_gen = GetMorganGenerator(radius=2)
    valid_mols = []
    valid_fps = []

    # Parse all SMILES, keep only valid ones
    for smi in tqdm(smiles_list, desc="Parsing molecules"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue   # логи уже отключены, просто пропускаем
        valid_mols.append(mol)
        valid_fps.append(morg_gen.GetFingerprint(mol))

    if not valid_mols:
        return [], [], [], []

    # Compute FCD (similarity scores)
    fcds = []
    if reference_fps is None:
        # Self‑similarity: average over all other molecules in the list
        for i, fp in enumerate(tqdm(valid_fps, desc="Computing self-FCD")):
            similarities = []
            for j, other_fp in enumerate(valid_fps):
                if i != j:
                    similarities.append(FingerprintSimilarity(fp, other_fp))
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            fcds.append(avg_sim)
    else:
        # Similarity to a fixed reference set
        for fp in tqdm(valid_fps, desc="Computing FCD vs reference"):
            similarities = [FingerprintSimilarity(fp, ref_fp) for ref_fp in reference_fps]
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            fcds.append(avg_sim)

    # Compute molecular properties for all valid molecules
    mws = [Descriptors.MolWt(mol) for mol in valid_mols]
    qeds = [QED.qed(mol) for mol in valid_mols]
    logs = [Descriptors.MolLogP(mol) for mol in valid_mols]

    return mws, qeds, logs, fcds


def read_smiles_from_file(filepath: str) -> List[str]:
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main(target_file: str, reference_file: Optional[str] = None):
    target_smiles = read_smiles_from_file(target_file)

    # Prepare reference fingerprints and compute reference properties if provided
    reference_fps = None
    ref_mws = ref_qeds = ref_logs = None
    if reference_file:
        ref_smiles = read_smiles_from_file(reference_file)
        morg_gen = GetMorganGenerator(radius=2)
        reference_fps = []
        for smi in tqdm(ref_smiles, desc="Building reference fingerprints"):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                reference_fps.append(morg_gen.GetFingerprint(mol))
            # else: пропускаем без логов

        # Compute reference properties (MW, QED, LogP) – FCD is ignored
        ref_mws, ref_qeds, ref_logs, _ = compute_properties_for_smiles_list(ref_smiles, reference_fps=None)
        if len(ref_mws) == 0:
            ref_mws = ref_qeds = ref_logs = None

    # Compute target properties (using reference_fps if available)
    target_mws, target_qeds, target_logs, target_fcds = compute_properties_for_smiles_list(
        target_smiles, reference_fps=reference_fps
    )

    n_valid = len(target_mws)
    if n_valid == 0:
        raise ValueError("No valid molecules in target file!")

    # Print target averages
    avg_target_mw = sum(target_mws) / n_valid
    avg_target_qed = sum(target_qeds) / n_valid
    avg_target_logp = sum(target_logs) / n_valid
    avg_target_fcd = sum(target_fcds) / n_valid

    print("\n✅ Target properties (averaged over {} molecules):".format(n_valid))
    print(f"  MW  : {avg_target_mw:.3f}")
    print(f"  QED : {avg_target_qed:.3f}")
    print(f"  LogP: {avg_target_logp:.3f}")
    print(f"  FCD : {avg_target_fcd:.3f} (avg Tanimoto to {'self' if reference_fps is None else 'reference set'})")

    # If reference properties are available, print comparison
    if ref_mws is not None:
        n_ref = len(ref_mws)
        avg_ref_mw = sum(ref_mws) / n_ref
        avg_ref_qed = sum(ref_qeds) / n_ref
        avg_ref_logp = sum(ref_logs) / n_ref

        print("\n📊 Reference properties (averaged over {} molecules):".format(n_ref))
        print(f"  MW  : {avg_ref_mw:.3f}")
        print(f"  QED : {avg_ref_qed:.3f}")
        print(f"  LogP: {avg_ref_logp:.3f}")

        print("\n📈 Comparison (Target - Reference):")
        print(f"  MW  : {avg_target_mw - avg_ref_mw:+.3f}  ({100*(avg_target_mw/avg_ref_mw - 1):+.1f}%)")
        print(f"  QED : {avg_target_qed - avg_ref_qed:+.3f}  ({100*(avg_target_qed/avg_ref_qed - 1):+.1f}%)")
        print(f"  LogP: {avg_target_logp - avg_ref_logp:+.3f}  ({100*(avg_target_logp/avg_ref_logp - 1):+.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute average molecular properties with deprecation-free Morgan fingerprints.")
    parser.add_argument("target", help="Path to target SMILES file")
    parser.add_argument("--ref", help="Path to reference SMILES (optional, for FCD baseline and property comparison)")
    args = parser.parse_args()
    main(args.target, args.ref)