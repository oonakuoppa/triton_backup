#!/bin/bash

# SLURM settings for the job:
#SBATCH --output=test_online_%j.out    # Save terminal output to file named trajectron_eval_<jobID>.out
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --mem=20G                          # Reserve 20 GB of memory
#SBATCH --time=00:30:00                    # Time limit: 4 hours
#SBATCH --cpus-per-task=6                  # Number of CPUs to use



# Creating the environment
module purge
module load triton/2024.1-gcc
module load mamba/2025.1
source activate trajectron++

# Running 
python test_online.py --log_dir=../experiments/nuScenes/models --data_dir=../experiments/processed --conf=config.json --eval_data_dict=nuScenes_test_mini_full.pkl
