from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import FingerprintSimilarity
from rdkit import RDLogger
from tqdm import tqdm
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

# Turn off all RDKit logs (including parsing errors)
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
            continue   # logs already disabled, skip
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


def main(target_files: List[str], reference_file: Optional[str] = None, save_dir: Optional[str] = None):
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
        if not reference_fps:
            print("Warning: no valid molecules in reference file.")
            reference_fps = None
        else:
            # Compute reference properties (MW, QED, LogP) – FCD is ignored
            ref_mws, ref_qeds, ref_logs, _ = compute_properties_for_smiles_list(ref_smiles, reference_fps=None)
            if len(ref_mws) == 0:
                ref_mws = ref_qeds = ref_logs = None

    # Process each target file
    targets_avg_mw = []
    targets_avg_qed = []
    targets_avg_logp = []
    targets_avg_fcd = []
    targets_labels = []
    targets_valid_counts = []

    for target_file in target_files:
        target_file = Path.cwd() / target_file
        label = target_file.stem  # short name for plots
        targets_labels.append(label)
        target_smiles = read_smiles_from_file(target_file)
        mws, qeds, logs, fcds = compute_properties_for_smiles_list(target_smiles, reference_fps=reference_fps)
        n = len(mws)
        targets_valid_counts.append(n)
        if n == 0:
            print(f"Warning: No valid molecules in target file '{target_file}'. Skipping from averages and plots.")
            targets_avg_mw.append(None)
            targets_avg_qed.append(None)
            targets_avg_logp.append(None)
            targets_avg_fcd.append(None)
        else:
            targets_avg_mw.append(sum(mws) / n)
            targets_avg_qed.append(sum(qeds) / n)
            targets_avg_logp.append(sum(logs) / n)
            targets_avg_fcd.append(sum(fcds) / n)

    # Filter out None entries if any
    valid_indices = [i for i, n in enumerate(targets_valid_counts) if n > 0]
    if not valid_indices:
        raise ValueError("No valid molecules found in any target file!")

    # Print results for each target
    print("\n" + "=" * 60)
    print("Average properties per target:")
    for i in valid_indices:
        print(f"\n  Target: {targets_labels[i]} ({targets_valid_counts[i]} molecules)")
        print(f"    MW  : {targets_avg_mw[i]:.3f}")
        print(f"    QED : {targets_avg_qed[i]:.3f}")
        print(f"    LogP: {targets_avg_logp[i]:.3f}")
        fcd_type = "reference set" if reference_fps is not None else "self"
        print(f"    FCD : {targets_avg_fcd[i]:.3f} (avg Tanimoto to {fcd_type})")

    # If reference properties are available, print comparison for each target
    if ref_mws is not None:
        n_ref = len(ref_mws)
        avg_ref_mw = sum(ref_mws) / n_ref
        avg_ref_qed = sum(ref_qeds) / n_ref
        avg_ref_logp = sum(ref_logs) / n_ref
        print("\n" + "=" * 60)
        print(f"Reference properties (averaged over {n_ref} molecules):")
        print(f"  MW  : {avg_ref_mw:.3f}")
        print(f"  QED : {avg_ref_qed:.3f}")
        print(f"  LogP: {avg_ref_logp:.3f}")
        print("\nComparison (Target - Reference):")
        for i in valid_indices:
            print(f"\n  Target: {targets_labels[i]}")
            print(f"    MW  : {abs(targets_avg_mw[i] - avg_ref_mw):.3f}  ({100*(targets_avg_mw[i]/avg_ref_mw - 1):+.1f}%)")
            print(f"    QED : {abs(targets_avg_qed[i] - avg_ref_qed):.3f}  ({100*(targets_avg_qed[i]/avg_ref_qed - 1):+.1f}%)")
            print(f"    LogP: {abs(targets_avg_logp[i] - avg_ref_logp):.3f}  ({100*(targets_avg_logp[i]/avg_ref_logp - 1):+.1f}%)")

    # Plot only if more than one valid target
    if len(valid_indices) > 1:
        # Prepare data for plotting (only valid targets)
        plot_labels = [targets_labels[i] for i in valid_indices]
        plot_mw = [targets_avg_mw[i] for i in valid_indices]
        plot_qed = [targets_avg_qed[i] for i in valid_indices]
        plot_logp = [targets_avg_logp[i] for i in valid_indices]
        plot_fcd = [targets_avg_fcd[i] for i in valid_indices]

        # First figure: FCD comparison
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        bars = ax1.bar(plot_labels, plot_fcd, color='skyblue', edgecolor='black')
        ax1.set_ylabel('Average FCD', fontsize=12)
        ax1.set_title('FCD Comparison Across Targets', fontsize=14)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        for bar, val in zip(bars, plot_fcd):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}',
                     ha='center', va='bottom', fontsize=10)
        fig1.tight_layout()

        # Second figure: comparative bar charts for MW, QED, LogP
        if ref_mws is not None:
            plot_labels_with_ref = plot_labels + ['Reference']
            plot_mw_with_ref = plot_mw + [avg_ref_mw]
            plot_qed_with_ref = plot_qed + [avg_ref_qed]
            plot_logp_with_ref = plot_logp + [avg_ref_logp]
            ref_color = 'lightgray'
        else:
            plot_labels_with_ref = plot_labels
            plot_mw_with_ref = plot_mw
            plot_qed_with_ref = plot_qed
            plot_logp_with_ref = plot_logp

        fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
        properties = [('MW', plot_mw_with_ref, 'Molecular Weight'),
                      ('QED', plot_qed_with_ref, 'QED'),
                      ('LogP', plot_logp_with_ref, 'LogP')]
        target_colors = ['salmon', 'lightgreen', 'plum']

        for ax, (prop, values, ylabel), t_color in zip(axes, properties, target_colors):
            if ref_mws is not None:
                colors_list = [t_color] * len(plot_labels) + [ref_color]
            else:
                colors_list = [t_color] * len(values)
            bars = ax.bar(plot_labels_with_ref, values, color=colors_list, edgecolor='black')
            ax.set_title(prop, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(values)*0.01),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        fig2.suptitle('Comparison of MW, QED, LogP Across Targets', fontsize=16)
        fig2.tight_layout()

        # Save or show plots
        if save_dir:
            SAVE_DIR = Path.cwd() / save_dir
            fcd_path = SAVE_DIR / 'fcd_comparison.png'
            props_path = SAVE_DIR / 'properties_comparison.png'
            fig1.savefig(fcd_path, dpi=150, bbox_inches='tight')
            fig2.savefig(props_path, dpi=150, bbox_inches='tight')
            print(f"\nGraphs saved to:\n  {fcd_path}\n  {props_path}")
            plt.close('all')
        else:
            plt.show()
    else:
        print("\n(Graphs are not displayed because there is only one valid target.)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute average molecular properties and compare multiple target sets.")
    parser.add_argument("target", nargs='+', help="Path(s) to target SMILES file(s)")
    parser.add_argument("--ref", help="Path to reference SMILES file (optional, for FCD baseline and property comparison)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save plots (instead of showing them). "
                                                                   "If not provided, plots are displayed interactively.")
    args = parser.parse_args()
    main(args.target, args.ref, save_dir=args.save_dir)
