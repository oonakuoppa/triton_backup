#!/bin/bash
#SBATCH --output=/scratch/work/kuoppao1/Trajectron-plus-plus_project/Trajectron-plus-plus/trainoutputs/traj_eval_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=6

# ympäristö
module restore adaptive-module
source activate trajectron++

# siirrytään oikeaan sijaintiin
cd /scratch/work/kuoppao1/Trajectron-plus-plus_project/Trajectron-plus-plus/experiments/pedestrians/

# evaluointi
python evaluate.py --model models/eth_vel --checkpoint 100 --data ../processed/eth_test.pkl --output_path results --output_tag eth_vel_12 --node_type PEDESTRIAN

