import sys

import os

import dill

import json

import argparse

import torch

import numpy as np

import pickle



sys.path.append("../../trajectron")

from tqdm import tqdm

from model.model_registrar import ModelRegistrar

from model.trajectron import Trajectron



seed = 0

np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(seed)



parser = argparse.ArgumentParser()

parser.add_argument("--model", help="model full path", type=str)

parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)

parser.add_argument("--data", help="full path to data file", type=str)

parser.add_argument("--output_path", help="path to output directory", type=str)

parser.add_argument("--output_tag", help="name tag for output file", type=str)

parser.add_argument("--node_type", help="node type to evaluate", type=str)

args = parser.parse_args()





def load_model(model_dir, env, ts=100):

    model_registrar = ModelRegistrar(model_dir, 'cpu')

    model_registrar.load_models(ts)

    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:

        hyperparams = json.load(config_json)



    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)

    trajectron.set_annealing_params()

    return trajectron, hyperparams





if __name__ == "__main__":

    with open(args.data, 'rb') as f:

        env = dill.load(f, encoding='latin1')



    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)



    if 'override_attention_radius' in hyperparams:

        for attention_radius_override in hyperparams['override_attention_radius']:

            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')

            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)



    scenes = env.scenes

    print("-- Preparing Node Graph")

    for scene in tqdm(scenes):

        scene.calculate_scene_graph(env.attention_radius,

                                    hyperparams['edge_addition_filter'],

                                    hyperparams['edge_removal_filter'])



    ph = hyperparams['prediction_horizon']



    all_data = []  # Tallennetaan listana dict, jossa ennusteet, ground truth, agent_id, frame_id



    with torch.no_grad():

        print("-- Evaluating Full predictions only")

        for i, scene in enumerate(scenes):

            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")

            for t in tqdm(range(0, scene.timesteps, 10)):

                timesteps = np.arange(t, t + 10)



                predictions = eval_stg.predict(scene,

                                              timesteps,

                                              ph,

                                              num_samples=2000,

                                              min_history_timesteps=7,

                                              min_future_timesteps=12,

                                              z_mode=False,

                                              gmm_mode=False,

                                              full_dist=False)



                if not predictions:

                    continue



                # Kerätään dataa ennusteista, ground truthista ja tunnisteista

                # predictions: dict[node_id] -> tensor (sampled trajectories)

                # Ground truth löytyy scene.nodes[node_id].future_traj

                # agent_id = node_id, frame_id = timestep t



                for node_id, pred in predictions.items():

                    node = scene.nodes[node_id]

                    if node.type.name != args.node_type:

                        continue



                    ground_truth = node.future_traj  # numpy array, shape (ph, 2)

                    # Frame id (alkutimestep): t

                    data_entry = {

                        'scene_id': i,

                        'frame_id': t,

                        'agent_id': node_id,

                        'predictions': pred.cpu().numpy(),  # shape: (num_samples, prediction_horizon, 2)

                        'ground_truth': ground_truth,

                    }

                    all_data.append(data_entry)



    os.makedirs(args.output_path, exist_ok=True)

    output_file = os.path.join(args.output_path, args.output_tag + '_full_predictions.pkl')

    with open(output_file, 'wb') as f:

        pickle.dump(all_data, f)



    print(f"Saved predictions and ground truth to {output_file}")
