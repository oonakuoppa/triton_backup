#!/bin/bash
#SBATCH --output=/scratch/work/kuoppao1/Trajectron-plus-plus_project/Trajectron-plus-plus/trainoutputs/traj_train_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6

# ympäristö
module restore adaptive-module
source activate trajectron++

# siirrytään oikeaan sijaintiin
cd /scratch/work/kuoppao1/Trajectron-plus-plus_project/Trajectron-plus-plus/trajectron/

# koulutus
python train.py \
  --eval_every 10 \
  --vis_every 1 \
  --train_data_dict eth_train.pkl \
  --eval_data_dict eth_val.pkl \
  --offline_scene_graph yes \
  --preprocess_workers 5 \
  --log_dir ../experiments/pedestrians/models \
  --log_tag _eth_vel_ar3 \
  --train_epochs 100 \
  --augment \
  --conf ../experiments/pedestrians/models/eth_vel/config.json
