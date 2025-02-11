#!/bin/bash
#SBATCH --partition=leia           # Choose the appropriate partition
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --gres=gpu:1               # Request 1 GPU (remove if not needed)
#SBATCH --time=03:00:00            # Maximum time allowed

# Activate Anaconda work environment
source /home/csi24301/miniconda3/etc/profile.d/conda.sh
conda activate /home/csi24301/projects/Greek_Article_Classifier/conda_env

# Run your code
python3 /home/csi24301/projects/Greek_Article_Classifier/bert.py

