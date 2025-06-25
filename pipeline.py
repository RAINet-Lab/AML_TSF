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
    # load configs/default_config.yaml
    default_config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.FullLoader)

    # load training config
    run_config = yaml.load(open('configs/config.yaml'), Loader=yaml.FullLoader)

    attack_list = run_config.get('attacks', [])
    dataset_list = run_config.get('datasets', [])
    model_list = run_config.get('models', [])
    config = {}
    config = defaultdict(lambda: defaultdict(dict))
    try: 
        config['backtime_comparison'] = run_config['backtime_comparison']
    except Exception as e:
        config['backtime_comparison'] = False
        print("Config parameter not found for backtime_comparison and set to False by default", e)
    try: 
        config['compute_poisoned_tensors'] = run_config['compute_poisoned_tensors']
    except Exception as e:
        config['compute_poisoned_tensors'] = False
        print("Config parameter not found for compute_poisoned_tensors and set to False by default", e)
    try: 
        config['compare_targeted_points_before_and_after_poisoning'] = run_config['compare_targeted_points_before_and_after_poisoning']
    except Exception as e:
        config['compare_targeted_points_before_and_after_poisoning'] = False
        print("Config parameter not found for compare_targeted_points_before_and_after_poisoning and set to False by default", e)

    for dataset in dataset_list:
        # config['Dataset'][dataset] = {}
        config['Dataset'][dataset] = default_config['Dataset'][dataset]
        config['Dataset'][dataset]['loopback'] = run_config['loopback']
        config['Dataset'][dataset]['horizon'] = run_config['horizon']
        config['Dataset'][dataset]['batch_size'] = run_config['batch_size']
        
        for attack in attack_list:

            # if attack == 'backtime':

            #     train_mean, train_std, train_data_seq, test_data_seq = load_raw_data(config['Dataset'][dataset], dataset)
            #     config['Dataset_backtime'][dataset]["train_mean"]=train_mean
            #     config['Dataset_backtime'][dataset]["train_std"]=train_std
            #     config['Dataset_backtime'][dataset]["train_data_seq"]=train_data_seq
            #     config['Dataset_backtime'][dataset]["test_data_seq"]=test_data_seq

            #     config['Attack'][attack] = default_config['Attack'][attack]
            #     config['Attack'][attack]['Target_Pattern'] = np.array(default_config['Target_Pattern'][default_config['Attack'][attack]['pattern_type']])
            #     config['Attack'][attack]['Model'] = default_config['Model'][default_config['Attack'][attack]['model_name']]
            #     config['Attack'][attack]['Model']['c_out'] = default_config['Dataset'][dataset]['num_of_vertices']
            #     config['Attack'][attack]['Model']['enc_in'] = default_config['Dataset'][dataset]['num_of_vertices']
            #     config['Attack'][attack]['Model']['dec_in'] = default_config['Dataset'][dataset]['num_of_vertices']


            #     config['Attack'][attack]['loopback'] = run_config['loopback']
            #     config['Attack'][attack]['horizon'] = run_config['horizon']
            #     # config['Attack'][attack]['learning_rate'] = default_config['Attack'][attack]['learning_rate']
            #     # config['Attack'][attack]['batch_size'] = run_config['batch_size']
            #     # config['Attack'][attack]['epochs'] = default_config['Attack'][attack]['epochs']
            #     # config['Attack'][attack]['warmup'] = default_config['Attack'][attack]['warmup']

            #     config['Attack'][attack]['Surrogate'] = default_config['Model'][default_config['Attack'][attack]['surrogate_name']]
            #     config['Attack'][attack]['Surrogate']['c_out'] = default_config['Dataset'][dataset]['num_of_vertices']
            #     config['Attack'][attack]['Surrogate']['enc_in'] = default_config['Dataset'][dataset]['num_of_vertices']
            #     config['Attack'][attack]['Surrogate']['dec_in'] = default_config['Dataset'][dataset]['num_of_vertices']
            
            # elif attack != 'backtime':
            config['Attack'][attack] = default_config['Attack'][attack]
            config['Attack'][attack]["poison_number"] = np.array(config['Attack'][attack]["poison_number"])
            config['Attack'][attack]["poison_number"].sort()
            config['Attack'][attack]["epsilon"] = np.array(config['Attack'][attack]["epsilon"])
            
        X_train,X_test,y_train,y_test = dataset_dict[dataset](config['Dataset'][dataset])
        X_train_flatten = flatten_dataset(X_train).numpy()
        X_test_flatten = flatten_dataset(X_test).numpy()
        y_train_flatten = flatten_dataset(y_train).cpu().detach().numpy()
        y_test_flatten = flatten_dataset(y_test).cpu().detach().numpy()

        config['Dataset'][dataset]["X_train"]=X_train
        config['Dataset'][dataset]["X_test"]=X_test
        config['Dataset'][dataset]["y_train"]=y_train
        config['Dataset'][dataset]["y_test"]=y_test
        config['Dataset'][dataset]["X_train_flatten"] = X_train_flatten
        config['Dataset'][dataset]["X_test_flatten"] = X_test_flatten
        config['Dataset'][dataset]["y_train_flatten"] = y_train_flatten
        config['Dataset'][dataset]["y_test_flatten"] = y_test_flatten

    for model in model_list:
        config['Model'][model] = default_config['Model'][model]
        config['Model'][model]['shap_batch_size'] = run_config['shap_batch_size']
        config['Model'][model]['batch_size'] = run_config['batch_size']

    config = edict(config)
    return config

def save_config(h5f, config):
    for key, value in config.items():
        print(key)
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

                elif attack=='backtime':
                    model_params_grp = model_grp.create_group("backtime")
                    for attack_name, fields in attacks.items():
                        attack_backtime_grp = model_params_grp.create_group(attack_name)
                        for key, value in fields.items():
                            if isinstance(value, torch.Tensor):
                                attack_backtime_grp.create_dataset(key, data=value.detach().cpu().numpy())
                            elif isinstance(value, np.ndarray):
                                attack_backtime_grp.create_dataset(key, data=value)
                            elif isinstance(value, (float)):
                                attack_backtime_grp.create_dataset(key, data=value)
                else:
                    attack_grp = model_grp.create_group(attack)
                    for key,value in attacks.items():
                        print(key)
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

                model = model_dict[model_name](config['Model'][model_name])
                model.to(dev)
                # model.load_state_dict(torch.load('/home/quentin/TimeSeriesF/results/EUMA_corrected/trained_models/dlinear_euma_70_35.pt'))
                # predictions = torch.tensor(np.loadtxt(f'/home/quentin/TimeSeriesF/results/EUMA_corrected/predictions/dlinear_euma_70_35_predictions.txt')) +offset
                
                ## Training the model
                model, predictions, best_state_dict, loss_train, val_loss, test_loss = train_model(model, dev, X_train, y_train, X_test, y_test, config.Model[model_name])
                model.to(dev)
                ## Running the attacks

                ## Extracting SHAP and CP values form real or surrogate model ???
                results_dict = load_results_dict(h5py.File('results/pipeline/shap_values/results.h5', 'r'))
                shap_values = results_dict['euma']['dlinear']['model_parameters']['shap_values']
                # print(shap_values)
                # with open(f'results/EUMA_corrected/shap_values/shap_values_test_euma_dlinear_70_35.pkl', 'rb') as f:
                #     shap_values = pickle.load(f).values

                # shap_values = get_SHAP_values(model, dev, X_train, X_test, config['Model'][model_name]).values
                # # # shap_values = shap_values.reshape(X_test.shape[0],-1,config['Model'][model_name]['pred_len'],config['Model'][model_name]['enc_in'])
                shap_values = torch.tensor(shap_values)
                # print(shap_values)
                X_train = X_train.cpu()
                y_train = y_train.cpu()
                X_test = X_test.cpu()
                y_test = y_test.cpu()
                # shap_values = shap_values.to(dev)
                X_test_rs = X_test.reshape(X_test.shape[0],config['Model'][model_name]['enc_in']*config['Model'][model_name]['seq_len']) + offset
                # X_test_rs = X_test_rs.to(dev)
                # print("shap_values shape ", shap_values.shape, type(shap_values))
                cp_values, xai_matrix = linear_xai(X_test_rs, shap_values)
                
                dim1 = shap_values.shape[1]
                # cp_values = torch.tensor(cp_values).to(dev)
                # predictions = torch.tensor(predictions).to(dev)
                # # run_tests(shap_values, cp_values,X_test,y_test,model,predictions,folder_path)
                tensor_dict = {'shap':shap_values,'cp':cp_values}
                # tensor_dict = {}
                # print(attack_list)
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
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["in_mape"] = atk.in_mape
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_mse_gt"] = atk.out_mse_gt
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_mae_gt"] = atk.out_mae_gt
                                results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}']["out_mse"] = atk.out_mse
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
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["in_mape"] = atk.in_mape
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_mse_gt"] = atk.out_mse_gt
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_mae_gt"] = atk.out_mae_gt
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_mse"] = atk.out_mse
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["out_mae"] = atk.out_mae
                                    results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["num_targeted"] = atk.num_targeted


                                    #### Recompute the SHAP values after attacking
                                    if config['compute_poisoned_tensors']:
                                    
                                        shap_after_poisoning = shap_values.clone()
                                        indices = atk.indices
                                        indices.sort()
                                        points = indices

                                        #### Recompute only the new points or the points within the window
                                        if idx > 0:
                                            points = []
                                            prev_indices = results_dict[dataset][model_name][f'{attack}_{str(p_number[idx-1])}_{str(eps)}_{tensor}']["indices"]
                                            prev_indices.sort()
                                            size = len(indices)
                                            prev_size = len(prev_indices)
                                            i,j=0,0
                                            while i<size and j<prev_size:
                                                if prev_indices[j]==indices[i]:
                                                    i+=1
                                                    j+=1
                                                else:
                                                    close_idxs_up = get_close_idxs(indices[i], prev_size, prev_indices, j, 1)
                                                    close_idxs_down = get_close_idxs(indices[i], prev_size, prev_indices, j-1, -1)
                                                    close_idxs = [indices[i]] + close_idxs_up + close_idxs_down
                                                    points.append(close_idxs)
                                                    i+=1
                                            points = [item for sublist in points for item in sublist]
                                            # print('points to compute : ', points)

                                        for point in points:
                                            # print(point, ', range of points :',max(0,point-dim1),point+1)
                                            temp = get_SHAP_values(model, dev, X_train.to(dev), atk.X_indices[max(0,point-dim1):point+1].to(dev), config['Model'][model_name]).values
                                            temp = torch.tensor(temp)
                                            shap_after_poisoning[max(0,point-dim1):point+1] = temp
                                        X_indices_rs = atk.X_indices.reshape(atk.X_indices.shape[0],config['Model'][model_name]['enc_in']*config['Model'][model_name]['seq_len']) + offset 
                                        results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["shap_poisoned"] = shap_after_poisoning
                                        tensor_poisoned = shap_after_poisoning

                                        #### Recompute the ChronoProf values if needed
                                        if tensor=='cp':
                                            cp_after_poisoning, xai_matrix = linear_xai(X_indices_rs, shap_after_poisoning)
                                            cp_after_poisoning = torch.tensor(cp_after_poisoning)
                                            results_dict[dataset][model_name][f'{attack}_{str(p_num)}_{str(eps)}_{tensor}']["cp_poisoned"] = cp_after_poisoning
                                            tensor_poisoned = cp_after_poisoning

                                        if config['compare_targeted_points_before_and_after_poisoning']:
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
                # results_dict[dataset][model_name]['model_parameters']["best_state_dict"] = best_state_dict
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
    config = parser_args()
    folder_path = 'results/pipeline/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    results_dict = main(config,folder_path)

    with h5py.File(f'{folder_path}results.h5', 'w') as h5f:
        save_results_dict(h5f, results_dict)
    with h5py.File(f'{folder_path}config.h5', 'w') as h5f:
        save_config(h5f, config)
    print("THE END")