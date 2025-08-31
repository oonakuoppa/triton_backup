#!/bin/bash



# SLURM settings for the job:

#SBATCH --output=trajectron_eval_%j.out    # Save terminal output to file named trajectron_eval_<jobID>.out

#SBATCH --gres=gpu:1                       # Request 1 GPU

#SBATCH --mem=20G                          # Reserve 20 GB of memory

#SBATCH --time=02:00:00                    # Time limit: 4 hours

#SBATCH --cpus-per-task=6                  # Number of CPUs to use







# Creating the environment

module purge

module load triton/2024.1-gcc

module load mamba/2025.1

source activate trajectron++







#  Make sure the following parameters are set correctly for your configuration: --model, --data, --output_tag

# This command is the same as in the Trajectron++ documentation

python evaluate_oma2.py --model /scratch/work/kuoppao1/Trajectron-plus-plus_project/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3 --checkpoint 100 --data ../processed/eth_test.pkl --output_path results --output_tag eth_zmode --node_type PEDESTRIAN
