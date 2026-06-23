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

python scripts/train_prior.py -i output/processed_smiles.smi -o output/trained_model -s ckpt --n_epochs 10 RNN