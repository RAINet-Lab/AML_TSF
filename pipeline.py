import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as mcolors
from torch.utils.data import TensorDataset, DataLoader
import h5py
import os
import copy
import sys
import yaml
import time
from easydict import EasyDict as edict
from collections import defaultdict
from models import LSTM, DLinear, PatchTST,NeuroFlexMLP,Linear,TSMixer
from utils import *
from attack import Attack, flatten_dataset, attack_contrib_elect_points,highest_contrib_elect_points,attack_distrib_elect_points,attack_rolling_var_dynamic_elect_points,attack_rolling_var_elect_points
# from backtime_utils.run import backtime_comparison
attack_dict={'attack_contrib':Attack.attack_contrib,'highest_contrib':Attack.highest_contrib,'attack_distrib':Attack.attack_distrib,'attack_rolling_var':Attack.attack_rolling_var,'attack_rolling_var_dynamic':Attack.attack_rolling_var_dynamic,}
elect_points_dict={'attack_contrib':attack_contrib_elect_points,'highest_contrib':highest_contrib_elect_points,'attack_distrib':attack_distrib_elect_points,'attack_rolling_var':attack_rolling_var_elect_points,'attack_rolling_var_dynamic':attack_rolling_var_dynamic_elect_points,}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parser_args():
    
    # load training config
    run_config = yaml.load(open('configs/config.yaml'), Loader=yaml.FullLoader)

    attack_list = run_config.get('attacks', [])
    print('attack_list : ',attack_list)
    dataset_list = run_config.get('datasets', [])
    print('dataset_list : ',dataset_list)
    model_list = run_config.get('models', [])
    print('model_list : ',model_list)
    config = {}
    config = defaultdict(lambda: defaultdict(dict))
    
    config['Model'] = {}
    for model in model_list:
        config['Model'][model]={}
    for dataset in dataset_list:
        lookback = run_config["Datasets"][dataset]['lookback']
        horizon = run_config["Datasets"][dataset]['horizon']
        if len(lookback) == len(horizon) :
            lookback = np.array(lookback)
            horizon = np.array(horizon)
        elif len(lookback)==1:
            horizon = np.array(horizon)
            lookback = np.array(lookback*len(horizon))
        elif len(horizon)==1:
            lookback = np.array(lookback)
            horizon = np.array(horizon*len(lookback))
        else:
            print('Error : lookback and horizon should have same dimension or at least one of these should only have one value')
            sys.exit(1)

        for attack in attack_list:

            config['Attack'][attack] = run_config['Attacks'][attack]
            config['Attack'][attack]["poison_number"] = np.array(config['Attack'][attack]["poison_number"])
            config['Attack'][attack]["poison_number"].sort()
            config['Attack'][attack]["epsilon"] = np.array(config['Attack'][attack]["epsilon"])
        print("after adding the attacks : ", config.keys())

        for idx,l in enumerate(lookback):
            h = horizon[idx]
            # print('lookback and horizon values : ', idx,l,h)
            X_train,X_test,y_train,y_test = dataset_dict[dataset](l,h)
            X_train_flatten = flatten_dataset(X_train).numpy()
            X_test_flatten = flatten_dataset(X_test).numpy()
            y_train_flatten = flatten_dataset(y_train).cpu().detach().numpy()
            y_test_flatten = flatten_dataset(y_test).cpu().detach().numpy()

            config['Dataset'][f'{dataset}_{l}_{h}']["X_train"]=X_train
            config['Dataset'][f'{dataset}_{l}_{h}']["X_test"]=X_test
            config['Dataset'][f'{dataset}_{l}_{h}']["y_train"]=y_train
            config['Dataset'][f'{dataset}_{l}_{h}']["y_test"]=y_test
            config['Dataset'][f'{dataset}_{l}_{h}']["X_train_flatten"] = X_train_flatten
            config['Dataset'][f'{dataset}_{l}_{h}']["X_test_flatten"] = X_test_flatten
            config['Dataset'][f'{dataset}_{l}_{h}']["y_train_flatten"] = y_train_flatten
            config['Dataset'][f'{dataset}_{l}_{h}']["y_test_flatten"] = y_test_flatten

            try: 
                config['Dataset'][f'{dataset}_{l}_{h}']['epsilon_for_shap'] = np.array(run_config["Datasets"][dataset]['epsilon_for_shap'])
                config['Dataset'][f'{dataset}_{l}_{h}']['poison_number_for_shap'] = np.array(run_config["Datasets"][dataset]['poison_number_for_shap'])
            except Exception as e:
                config['Dataset'][f'{dataset}_{l}_{h}']['epsilon_for_shap'] = np.array([])
                config['Dataset'][f'{dataset}_{l}_{h}']['poison_number_for_shap'] = np.array([])
                
            config['Dataset'][f'{dataset}_{l}_{h}']['batch_size'] = run_config["Datasets"][dataset]['batch_size']

            for model in model_list:
                config['Dataset'][f'{dataset}_{l}_{h}'][model] = copy.deepcopy(run_config['Models'][model])
                config['Dataset'][f'{dataset}_{l}_{h}'][model]['shap_batch_size'] = run_config['shap_batch_size']
                config['Dataset'][f'{dataset}_{l}_{h}'][model]['batch_size'] = run_config['batch_size']
                config['Dataset'][f'{dataset}_{l}_{h}'][model]['seq_len'] = int(l)
                config['Dataset'][f'{dataset}_{l}_{h}'][model]['pred_len'] = int(h)
        #         print(f'for model {model} it stores lookback {l} and horizon {h}')
        #     print(dataset,l,h,'dlinear',config['Dataset'][f'euma_{l}_{h}']['dlinear']['seq_len'])

        # print('after everything')
        # print(config['Dataset']['euma_70_35']['dlinear']['seq_len'])
        # print(config['Dataset']['euma_60_35']['dlinear']['seq_len'])
        # print(config['Dataset']['euma_50_35']['dlinear']['seq_len'])

    
    config = edict(config)
    return config

def save_config(h5f, config):
    for key, value in config.items():
        if isinstance(value, dict):
            subgroup = h5f.create_group(key)
            save_config(subgroup, value)
        elif isinstance(value, torch.Tensor):
            h5f.create_dataset(key, data=value.detach().cpu().numpy())
        elif isinstance(value, np.ndarray):
            h5f.create_dataset(key, data=value)
        elif isinstance(value, (int, float, bool)):
            h5f.create_dataset(key, data=value)
        elif isinstance(value, str):
            dt = h5py.string_dtype(encoding='utf-8')
            h5f.create_dataset(key, data=value, dtype=dt)
        else:
            print(value)
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

def save_results_dict(h5f, results_dict):
    for dataset_name, models in results_dict.items():
        dataset_grp = h5f.require_group(dataset_name)
        for model_name, model_data in models.items():
            model_grp = dataset_grp.require_group(model_name)

            for attack, attacks in model_data.items():

                if attack == "model_parameters":
                    params_grp = model_grp.create_group("model_parameters")
                    for param_name, value in attacks.items():
                        if isinstance(value, torch.Tensor):
                            params_grp.create_dataset(param_name, data=value.detach().cpu().numpy())
                        elif isinstance(value, np.ndarray):
                            params_grp.create_dataset(param_name, data=value)
                        elif isinstance(value, (float)):
                            params_grp.create_dataset(param_name, data=value)
                        elif isinstance(value, dict):
                            subgroup = params_grp.create_group(param_name)
                            save_config(subgroup, value)
                else:
                    attack_grp = model_grp.create_group(attack)
                    for key,value in attacks.items():
                        if 'attack_rolling_var_dynamic' in attack and key == 'decision_tensor':
                            dyn_tensor_grp = attack_grp.create_group("dyn_decision_tensor")
                            for idx, tensor in enumerate(value):
                                arr = tensor.detach().cpu().numpy()
                                dyn_tensor_grp.create_dataset(f'tensor_{idx}', data=arr)
                        else:
                            if hasattr(value, 'detach'):
                                value = value.detach().cpu().numpy()
                            attack_grp.create_dataset(key, data=value)

def main(config,folder_path):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attack_list = config['Attack'].keys()
    dataset_list = config['Dataset'].keys()
    model_list = config['Model'].keys()
    print(f"Attack list: {attack_list}, dataset list: {dataset_list}, model list: {model_list}")
    results_dict = {}
    results_dict = defaultdict(lambda: defaultdict(dict))
    offset=3

    for dataset in dataset_list:
        X_train = config.Dataset[dataset]["X_train"].to(dev)
        y_train = config.Dataset[dataset]["y_train"].to(dev)
        X_test = config.Dataset[dataset]["X_test"].to(dev)
        y_test = config.Dataset[dataset]["y_test"].to(dev)

        for model_name in model_list:
            # try :
                # if config['backtime_comparison']:
                    # atk_ts_shape, train_metrics = backtime_comparison(dataset)
                    # results_dict[dataset][model_name]['backtime'] = {}
                    # results_dict[dataset][model_name]['backtime']['backtime']["atk_ts_shape"] = atk_ts_shape
                    # results_dict[dataset][model_name]['backtime']['backtime']["cln_mae"] = train_metrics[1]
                    # results_dict[dataset][model_name]['backtime']['backtime']["cln_rmse"] = train_metrics[2]
                    # results_dict[dataset][model_name]['backtime']['backtime']["atk_mae"] = train_metrics[3]
                    # results_dict[dataset][model_name]['backtime']['backtime']["atk_rmse"] = train_metrics[4]
                    # results_dict[dataset][model_name]['backtime']['backtime']["f_norm"] = train_metrics[5]
                    # results_dict[dataset][model_name]['backtime']['backtime']["mse_norm"] = train_metrics[6]
                    # results_dict[dataset][model_name]['backtime']['backtime']["mape"] = train_metrics[7]
                model = model_dict[model_name](config.Dataset[dataset][model_name])
                model.to(dev)
                # model.load_state_dict(torch.load('/home/quentin/TimeSeriesF/results/EUMA_corrected/trained_models/dlinear_euma_70_35.pt'))
                # predictions = torch.tensor(np.loadtxt(f'/home/quentin/TimeSeriesF/results/EUMA_corrected/predictions/dlinear_euma_70_35_predictions.txt')) +offset
                
                ## Training the model
                
                model, predictions, loss_train, val_loss, test_loss = train_model(model, dev, X_train, y_train, X_test, y_test, config.Dataset[dataset][model_name])
                model.to(dev)

                ## Extracting SHAP and CP values form real or surrogate model ???
                # prev_results_dict = load_results_dict(h5py.File('results/save/results.h5', 'r'))
                # shap_values = prev_results_dict['euma']['dlinear']['model_parameters']['shap_values']
    
                shap_values = get_SHAP_values(model, dev, X_train, X_test, config.Dataset[dataset][model_name]).values
                shap_values = torch.tensor(shap_values)

                X_train = X_train.cpu()
                y_train = y_train.cpu()
                X_test = X_test.cpu()
                y_test = y_test.cpu()

                X_test_rs = X_test.reshape(X_test.shape[0],config.Dataset[dataset][model_name]['enc_in']*config.Dataset[dataset][model_name]['seq_len']) + offset
                cp_values, xai_matrix = linear_xai(X_test_rs, shap_values)
                tensor_dict = {'shap':shap_values,'cp':cp_values}

                for attack in attack_list:
                    if attack in ['pgd', 'fgsm', 'bim'] :
                        print(f"Running attack {attack} on {model_name} training with {dataset}")
                        for eps in config['Attack'][attack]['epsilon']:
                            p_number = config['Attack'][attack]["poison_number"]
                            # print(p_number)
                            for idx, p_num in enumerate(p_number):
                                atk = Attack(X_test, y_test, predictions, None, model, dev, attack, p_num, eps)
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}'] = {}
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["input"] = atk.input
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["output"] = atk.output
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["target_points"] = atk.target_points
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["decision_tensor"] = atk.decision_tensor
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["indices"] = atk.indices
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["X_indices"] = atk.X_indices
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["duration"] = atk.duration
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["in_fnorm"] = atk.in_fnorm
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["in_mse"] = atk.in_mse
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["in_rmse"] = atk.in_rmse
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["in_mape"] = atk.in_mape
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_mse_gt"] = atk.out_mse_gt
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_rmse_gt"] = atk.out_rmse_gt
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_mae_gt"] = atk.out_mae_gt
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_mse"] = atk.out_mse
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_rmse"] = atk.out_rmse
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_mae"] = atk.out_mae
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["num_targeted"] = atk.num_targeted
                    else :
                        for tensor in tensor_dict:
                            print(f"Running attack {attack} using {tensor} values on {model_name} training with {dataset}")
                            # if attack != 'backtime':
                            for eps in config['Attack'][attack]['epsilon']:
                                p_number = config['Attack'][attack]["poison_number"]
                                # print(p_number)
                                for idx, p_num in enumerate(p_number):
                                    atk = Attack(X_test, y_test, predictions, tensor_dict[tensor], model, dev, attack, p_num, eps)
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}'] = {}
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["input"] = atk.input
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["output"] = atk.output
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["target_points"] = atk.target_points
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["decision_tensor"] = atk.decision_tensor
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["indices"] = atk.indices
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["X_indices"] = atk.X_indices
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["duration"] = atk.duration
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["in_fnorm"] = atk.in_fnorm
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["in_mse"] = atk.in_mse
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["in_rmse"] = atk.in_rmse
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["in_mape"] = atk.in_mape
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_mse_gt"] = atk.out_mse_gt
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_rmse_gt"] = atk.out_rmse_gt
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_mae_gt"] = atk.out_mae_gt
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_mse"] = atk.out_mse
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_rmse"] = atk.out_rmse
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_mae"] = atk.out_mae
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["num_targeted"] = atk.num_targeted

                                    #### Recompute the SHAP values after attacking
                                    if eps in config['eps_for_shap'] and p_num in config['num_p_for_shap']:

                                        shap_after_poisoning = get_SHAP_values(model, dev, X_train.to(dev), atk.X_indices.to(dev), config.Dataset[dataset][model_name]).values
                                        shap_after_poisoning = torch.tensor(shap_after_poisoning)
                                        X_indices_rs = atk.X_indices.reshape(atk.X_indices.shape[0],config.Dataset[dataset][model_name]['enc_in']*config.Dataset[dataset][model_name]['seq_len']) + offset 
                                        results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["shap_poisoned"] = shap_after_poisoning
                                        tensor_poisoned = shap_after_poisoning

                                        #### Recompute the ChronoProf values if needed
                                        if tensor=='cp':
                                            cp_after_poisoning, xai_matrix = linear_xai(X_indices_rs, shap_after_poisoning)
                                            cp_after_poisoning = torch.tensor(cp_after_poisoning)
                                            results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["cp_poisoned"] = cp_after_poisoning
                                            tensor_poisoned = cp_after_poisoning

                                        #### Get the points that would be poisoned if we attack again the poisoned dataset
                                        if attack=='attack_rolling_var_dynamic' or attack=='attack_rolling_var':
                                            indices_after_pois = elect_points_dict[attack](tensor_poisoned, results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["input"], p_num)
                                            results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["indices_after_pois"] = indices_after_pois
                                        else:
                                            indices_after_pois = elect_points_dict[attack](tensor_poisoned, p_num)
                                            results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["indices_after_pois"] = indices_after_pois

                                    # if config['backtime_comparison']:
                                        # atk_ts_shape, train_metrics = backtime_comparison(dataset, atk.target_points)
                                        # results_dict[dataset][model_name]['backtime'] = {}
                                        # results_dict[dataset][model_name]['backtime']['backtime']["atk_ts_shape"] = atk_ts_shape
                                        # results_dict[dataset][model_name]['backtime'][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["cln_mae"] = train_metrics[1]
                                        # results_dict[dataset][model_name]['backtime'][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["cln_rmse"] = train_metrics[2]
                                        # results_dict[dataset][model_name]['backtime'][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["atk_mae"] = train_metrics[3]
                                        # results_dict[dataset][model_name]['backtime'][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["atk_rmse"] = train_metrics[4]
                                        # results_dict[dataset][model_name]['backtime'][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["f_norm"] = train_metrics[5]
                                        # results_dict[dataset][model_name]['backtime'][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["mse_norm"] = train_metrics[6]
                                        # results_dict[dataset][model_name]['backtime'][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["mape"] = train_metrics[7]

                results_dict[dataset][model_name]['model_parameters'] = {}
                results_dict[dataset][model_name]['model_parameters']["loss_train"] = loss_train
                results_dict[dataset][model_name]['model_parameters']["val_loss"] = val_loss
                results_dict[dataset][model_name]['model_parameters']["test_loss"] = test_loss
                results_dict[dataset][model_name]['model_parameters']["predictions"] = predictions
                results_dict[dataset][model_name]['model_parameters']['X_train'] = config.Dataset[dataset]["X_train"]
                results_dict[dataset][model_name]['model_parameters']['y_train'] = config.Dataset[dataset]["y_train"]
                results_dict[dataset][model_name]['model_parameters']['X_test'] = config.Dataset[dataset]["X_test"]
                results_dict[dataset][model_name]['model_parameters']['y_test'] = config.Dataset[dataset]["y_test"]
                torch.save(model.state_dict(), f'{folder_path}{dataset}_{model_name}.pth')
                results_dict[dataset][model_name]['model_parameters']['shap_values'] = shap_values
                results_dict[dataset][model_name]['model_parameters']['cp_values'] = cp_values


            # except Exception as e:
            #     # print(f"Error at attack {attack}: {e}")
            #     print(f"Error at attack : {e}")

    return results_dict


if __name__ == "__main__":

    folder_path = 'results/tests/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    config = parser_args()
    with h5py.File(f'{folder_path}config.h5', 'w') as h5f:
        save_config(h5f, config)

    results_dict = main(config,folder_path)
    with h5py.File(f'{folder_path}results.h5', 'w') as h5f:
        save_results_dict(h5f, results_dict)

    print("THE END")