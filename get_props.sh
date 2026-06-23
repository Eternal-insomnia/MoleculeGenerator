#!/bin/bash
#SBATCH --partition=aichem
#SBATCH --job-name=smls_rnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /mnt/tank/scratch/asuvorov/miniconda3/etc/profile.d/conda.sh
conda activate smiles_rnn

# python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

python utility_scripts/get_avg_props.py output/rnn_base10k.smi output/rnn_trained10k.smi output/rnn_ftuned10k.smi --ref output/processed_smiles.smi --save-dir output/graphs