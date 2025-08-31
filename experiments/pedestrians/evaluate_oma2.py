import sys

import os

import dill

import json

import argparse

import torch

import numpy as np

import pandas as pd

import pickle



sys.path.append("../../trajectron")

from tqdm import tqdm

from model.model_registrar import ModelRegistrar

from model.trajectron import Trajectron

import evaluation



seed = 0

np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(seed)



parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="model full path")

parser.add_argument("--checkpoint", type=int, help="model checkpoint to evaluate")

parser.add_argument("--data", type=str, help="full path to data file")

parser.add_argument("--output_path", type=str, help="path to output csv and pickle files")

parser.add_argument("--output_tag", type=str, help="name tag for output files")

parser.add_argument("--node_type", type=str, help="node type to evaluate")

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





def save_predictions_for_visualization(predictions, scene, scene_id, output_path, output_tag, mode_type):

    output_data = []

    for node, node_predictions in predictions.items():

        for t, samples in node_predictions.items():

            try:

                data_entry = {

                    'scene_id': scene_id,

                    'scene_name': scene.name,

                    'timestep': int(t),

                    'node_id': node.id,

                    'node_type': node.type,

                    'history': node.history[t].copy(),        # (history_length, 2)

                    'ground_truth': node.future[t].copy(),    # (prediction_horizon, 2)

                    'predictions': samples                    # shape: (num_samples, prediction_horizon, 2)

                }

                output_data.append(data_entry)

            except KeyError:

                continue  # skip if data is not available

    filename = f"{output_tag}_scene_{scene_id}_{mode_type}_predictions.pkl"

    with open(os.path.join(output_path, filename), 'wb') as f:

        pickle.dump(output_data, f)





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

    max_hl = hyperparams['maximum_history_length']



    with torch.no_grad():



        ########## MODE Z (tallennetaan tämä vain esimerkkinä) ##########

        eval_ade_batch_errors = np.array([])

        eval_fde_batch_errors = np.array([])

        eval_kde_nll = np.array([])



        print("-- Evaluating Mode Z")

        for i, scene in enumerate(scenes):

            print(f"---- Evaluating Scene {i+1}/{len(scenes)}")

            all_predictions = {}

            for t in tqdm(range(0, scene.timesteps, 10)):

                timesteps = np.arange(t, t + 10)

                predictions = eval_stg.predict(scene,

                                               timesteps,

                                               ph,

                                               num_samples=2000,

                                               min_history_timesteps=7,

                                               min_future_timesteps=12,

                                               z_mode=True,

                                               full_dist=False)



                if not predictions:

                    continue



                # Tallenna kaikki ennusteet tätä sceneä varten

                for k, v in predictions.items():

                    if k not in all_predictions:

                        all_predictions[k] = {}

                    all_predictions[k].update(v)



                batch_error_dict = evaluation.compute_batch_statistics(predictions,

                                                                       scene.dt,

                                                                       max_hl=max_hl,

                                                                       ph=ph,

                                                                       node_type_enum=env.NodeType,

                                                                       map=None,

                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))

                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))

                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))



            # Tallennetaan visuaalisointia varten

            save_predictions_for_visualization(all_predictions, scene, i, args.output_path, args.output_tag, mode_type="z_mode")



        # Tallennetaan metriikat (voit poistaa nämä jos et tarvitse)

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'z_mode'}

                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_z_mode.csv'))

        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'z_mode'}

                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_z_mode.csv'))

        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'z_mode'}

                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_z_mode.csv'))
