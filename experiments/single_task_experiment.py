import os
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

from utils.metrics import metrics
from experiments.base_experiment import BaseExperiment
from data.toy_functions import sample_function
from data.toy_functions import generate_splits
from utils.saver.single_task_saver import single_task_saver
from utils.metrics import get_summary
from utils.plots.single_task_plots import single_task_plots, plot_single_task_overlay
from utils.loader.single_task_loader import single_task_overlay_loader
from utils.general import set_determinism, build_model_dict

class SingleTaskExperiment:
    def __init__(self, model_type=None, seeds=None, model_dict=None, plot_dict=None, seed_xgrid=None, seed_function=None):
        self.model_types = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'MLPDropoutNet', 'DeepEnsembleNet'] if model_type is None else model_type
        # Seed for model init
        self.seeds = seeds if seeds is not None else [random.randint(0, 1000) for _ in range(3)]
        # Seed for toy task x-grid
        self.seed_xgrid = seed_xgrid
        # Seed for toy task function generation
        self.seed_function = seed_function
        self.plot_dict = {"Single": [], "Overlay": []} if plot_dict is None else plot_dict
        # self.hidden_dim = hidden_dim
        # self.hyper_hidden_dim = hyper_hidden_dim
        if model_dict is None or self.model_types != list(model_dict.keys()):
            self.model_dict = {
                'IC_FDNet' :       {'hidden_dim': 23, 'hyper_hidden_dim': 6},      # ≈1004 params
                'LP_FDNet':        {'hidden_dim': 24, 'hyper_hidden_dim': 5},      # ≈1011
                'HyperNet':        {'hidden_dim': 25, 'hyper_hidden_dim': 9},      # ≈1012
                'BayesNet':        {'hidden_dim': 166},                            # ≈998
                'GaussHyperNet':   {'hidden_dim': 24, 'hyper_hidden_dim': 5, 'latent_dim': 9},  # ≈994
                'MLPNet':          {'hidden_dim': 333, 'dropout_rate': 0.1},       # ≈1000
                'MLPDropoutNet':   {'hidden_dim': 333, 'dropout_rate': 0.1},       # ≈1000
                'DeepEnsembleNet': {'hidden_dim': 33, 'dropout_rate': 0.1, 'num_models': 10,
                                    'ensemble_seed_list': [0,1,2,3,4,5,6,7,8,9]}    # ≈1000 total
            }
        else:
            self.model_dict = model_dict

        self.model_dict = {model: self.model_dict[model] for model in self.model_types}

        # self.kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        # self.non_kl_models = {'MLPNet', 'DeepEnsembleNet', 'HyperNet'}
        # self.no_variance_models = {'MLPNet', 'HyperNet'}

        kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        stoch_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'DeepEnsembleNet', 'MLPDropoutNet'}
        mc_model = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'MLPDropoutNet'}
        self.is_stoch = [m in stoch_models for m in self.model_types]
        self.kl_exist = [m in kl_models for m in self.model_types]
        self.training_type = ['MC' if m in mc_model else 'Ensemble' if m == 'DeepEnsembleNet' else 'Deterministic' for m in self.model_types]

    def run_experiments(self, input_data_dict=None,
            epochs=1000, beta_param_dict=None, checkpoint_dicts=None,
            MC_train=1, MC_val=100, MC_test=50, analysis=True, 
            save_path=None, save_switch=False, ensemble_epochs=None, train_val_metrics=True):
        # Parameters
        model_types = self.model_types
        model_dict = self.model_dict
        seeds = self.seeds

        # if 'DeepEnsembleNet' in model_types and num_models is None:
        #     num_models = MC_test

        if checkpoint_dicts is None:
            # Stochastic checkpoint dict
            stoch_checkpoint_dict = {
            'metric_str': 'crps',
            'region_interp': region_interp,
            'min_or_max': 'min',
            'interp_or_extrap': 'interp'
            }
            # Deterministic checkpoint dict
            det_checkpoint_dict = {
            'metric_str': 'mse',
            'region_interp': region_interp,
            'min_or_max': 'min',
            'interp_or_extrap': 'interp'
            }
            checkpoint_dicts = {'stoch': stoch_checkpoint_dict, 'det': det_checkpoint_dict}
        
        # Make dir
        os.makedirs(save_path, exist_ok=True)

        # Load inputs and relevant fields.
        # If input_data_dict is None, we are in toy mode and can control x-grid via seed_xgrid.
        if input_data_dict is None:
            if self.seed_xgrid is not None:
                input_data_dict = generate_splits(seed=self.seed_xgrid)
            else:
                input_data_dict = generate_splits()
        else:
            # Real dataset (npz) already provides x/y splits
            pass

        # Unpack dictionary
        region = input_data_dict['region']
        region_interp = input_data_dict['region_interp']
        x_train = input_data_dict['x_train']
        x_val = input_data_dict['x_val']
        x_test = input_data_dict['x_test']

        # Make sure the checkpoint dictionary has the correct interpolation region
        if len(checkpoint_dicts['stoch']) != 0:
            checkpoint_dicts['stoch']['region_interp'] = region_interp
        if len(checkpoint_dicts['det']) != 0:
            checkpoint_dicts['det']['region_interp'] = region_interp

        for seed_model in tqdm(seeds, leave=False):
            # Model-init seed (and general RNG)
            set_determinism(seed=seed_model)

            # Seed-dependent save path
            seed_save_path = os.path.join(save_path, f"seed_{seed_model}")

            # ------------------------------------------------------------------
            # Targets: either real dataset (provided y_*) or toy function f(x)
            # ------------------------------------------------------------------
            if input_data_dict is not None and "y_train" in input_data_dict:
                # Real dataset: same y’s for all seeds
                y_train = torch.as_tensor(input_data_dict["y_train"], dtype=torch.float64)
                y_val   = torch.as_tensor(input_data_dict["y_val"],   dtype=torch.float64)
                y_test  = torch.as_tensor(input_data_dict["y_test"],  dtype=torch.float64)
                desc    = input_data_dict.get("desc", "real_dataset")
            else:
                # Toy: sample function with potentially separate seed
                seed_f = self.seed_function if self.seed_function is not None else seed_model
                f, desc = sample_function(seed=seed_f)
                y_train = torch.tensor(f(x_train), dtype=torch.float64)
                y_val   = torch.tensor(f(x_val),   dtype=torch.float64)
                y_test  = torch.tensor(f(x_test),  dtype=torch.float64)

            for model_type in tqdm(model_types):
                # Check if the model is stochastic
                is_stoch = self.is_stoch[model_types == model_type]
                # Check if kl divergence exist
                kl_exist = self.kl_exist[model_types == model_type]
                # Pick correct checkpoint condition
                checkpoint_dict = checkpoint_dicts['stoch'] if is_stoch else checkpoint_dicts['det']

                # Base experiment instance
                exp_inst = BaseExperiment(model_type=model_type, seed=seed_model, **model_dict[model_type], save_path=seed_save_path)
                # Choose number of epochs for Deep Ensemble 
                is_de = (model_type == 'DeepEnsembleNet')
                epochs_to_use = ensemble_epochs if (is_de and ensemble_epochs != None) else epochs

                # Run experiment
                info = {'region': np.array(region), 'region_interp': np.array(region_interp)}
                data = (x_train, y_train, x_val, y_val, x_test, y_test, desc)
                preds_dict, data, training_time, metrics_dict, trainer = exp_inst.run_experiments(
                        data=data, epochs=epochs_to_use, beta_param_dict=beta_param_dict,
                        MC_train=MC_train, MC_val=MC_val, MC_test=MC_test, 
                        checkpoint_dict=checkpoint_dict, save_switch=save_switch, train_val_metrics=train_val_metrics, info=info
                        )
                
                # Unpack dictionary if training and validation metrics are calculated
                if isinstance(preds_dict, dict):
                    preds_train = preds_dict['train']; preds_val = preds_dict['val']; preds_test = preds_dict['test']
                    metrics_train = metrics_dict['train']; metrics_val = metrics_dict['val']; metrics_test = metrics_dict['test']
                else:
                    preds_test = preds_dict
                    metrics_test = metrics_dict

                if analysis:  
                    # Plot and save visuals
                    single_plot_save_path = None if save_path is None else os.path.join(seed_save_path, model_type, "plots")
                    # plot_types = self.get_plot_types(model_type)
                    single_task_plots(trainer, preds_test, x_train, y_train, x_val, y_val, x_test, y_test, region_interp, 
                    metrics_train=metrics_train, metrics_val=metrics_val, metrics_test=metrics_test, 
                    kl_exist=kl_exist, is_stoch=is_stoch, block=False, save_dir=single_plot_save_path, plot_types=self.plot_dict['Single'])

            if analysis:
                # Model colors for overlay plots
                model_colors = {
                    'IC_FDNet': '#1f77b4',         # Blue
                    'LP_FDNet': '#ff7f0e',         # Orange
                    'BayesNet': '#2ca02c',         # Green
                    'GaussHyperNet': "#ff0000",    # Red
                    'DeepEnsembleNet': "#653593",  # Purple
                    'HyperNet': '#8c564b',         # Brown
                    'MLPNet': '#e377c2',           # Pink
                    'MLPDropoutNet': "#e2fb55",    # Yellow
                }
                # Stochastic Models
                stoch_models = {"IC_FDNet", "LP_FDNet", "BayesNet", "GaussHyperNet", "DeepEnsembleNet", "MLPDropoutNet"}
                # Stochastic Metrics
                stoch_metrics = {"var", "nlpd_kde", "nlpd_hist", "crps"}
                # Plot and save overlay plots
                plot_single_task_overlay(seed_save_path, region_interp, model_types, stoch_models, stoch_metrics, 
                model_colors, show_figs=False, use_db_scale=True, plot_types=self.plot_dict['Overlay'])
                
            print(f"Completed: {model_type} | seed: {seed_model} | training time: {training_time}s")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Results directory: create a fresh timestamped run if it exists
    # ------------------------------------------------------------------
    base_save_path = os.path.join("results", "single_task_test_run")
    save_path = base_save_path
    if os.path.exists(save_path):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_path = f"{base_save_path}_{timestamp}"
        print(f"Existing results dir found, using new dir: {save_path}")
    else:
        print(f"Saving results to new dir: {save_path}")

    # ------------------------------------------------------------------
    # Models to run (main paper models)
    # ------------------------------------------------------------------
    model_type = [
        "IC_FDNet",
        "LP_FDNet",
        # "HyperNet",
        "BayesNet",
        "GaussHyperNet",
        "MLPDropoutNet",
        "DeepEnsembleNet",
    ]
    # (You can comment some out if you want it to run faster overnight.)

    # ------------------------------------------------------------------
    # Seeds: canonical paper ones + alternate ones to probe
    # ------------------------------------------------------------------
    seeds = [7, 8, 9, 24, 25, 26, 1, 2, 3]
    seeds.extend(range(33, 100))

    # ------------------------------------------------------------------
    # Model configs: ~1k parameters per model (including hypernets)
    # ------------------------------------------------------------------
    # These match the rough 1k-param regime you used in the paper.
    full_model_dict = {
        "IC_FDNet":      {"hidden_dim": 23,  "hyper_hidden_dim": 6},
        "LP_FDNet":      {"hidden_dim": 24,  "hyper_hidden_dim": 5},
        "HyperNet":      {"hidden_dim": 25,  "hyper_hidden_dim": 9},
        "BayesNet":      {"hidden_dim": 166},
        "GaussHyperNet": {"hidden_dim": 24,  "hyper_hidden_dim": 5, "latent_dim": 9},
        # Plain MLP (no Bayesian stuff) – you can include it if you like:
        "MLPNet":        {"hidden_dim": 333, "dropout_rate": 0.1},
        "MLPDropoutNet": {"hidden_dim": 333, "dropout_rate": 0.1},
        "DeepEnsembleNet": {
            "hidden_dim": 33,
            "dropout_rate": 0.1,
            "num_models": 10,
            "ensemble_seed_list": list(range(10)),
        },
    }
    # Restrict to the models we actually asked for
    model_dict = {m: full_model_dict[m] for m in model_type}

    # ------------------------------------------------------------------
    # Training hyperparameters (paper-like)
    # ------------------------------------------------------------------
    epochs = 400
    MC_train = 1
    MC_val = 100
    MC_test = 100

    # Train each ensemble member less, as in your original logic
    ensemble_epochs = 40  # 400 / 10

    beta_scheduler = "cosine"
    warmup_epochs = 200
    beta_max = 0.01
    beta_param_dict = {
        "beta_scheduler": beta_scheduler,
        "warmup_epochs": warmup_epochs,
        "beta_max": beta_max,
    }

    analysis = False
    save_switch = True

    # ------------------------------------------------------------------
    # Data settings (toy 1D regression, paper-like)
    # ------------------------------------------------------------------
    region_interp = (-1.0, 1.0)  # interpolation band
    x_min = -10.0
    x_max = 10.0

    n_train = 1024
    n_test = 2001
    n_val_interp = 256
    n_val_extrap = 256

    # Use the first seed for generating the input splits; the model seeds
    # are handled inside SingleTaskExperiment.
    data_seed = seeds[0]
    input_data_dict = generate_splits(
        x_min=x_min,
        x_max=x_max,
        region_interp=region_interp,
        n_train=n_train,
        n_test=n_test,
        n_val_interp=n_val_interp,
        n_val_extrap=n_val_extrap,
        seed=data_seed,
    )

    # ------------------------------------------------------------------
    # Checkpoint logic (same as before but with updated region)
    # ------------------------------------------------------------------
    do_checkpoint = True
    if do_checkpoint:
        stoch_checkpoint_dict = {
            "metric_str": "mse",
            "region_interp": region_interp,
            "min_or_max": "min",
            "interp_or_extrap": "interp",
        }
        det_checkpoint_dict = {
            "metric_str": "mse",
            "region_interp": region_interp,
            "min_or_max": "min",
            "interp_or_extrap": "interp",
        }
        checkpoint_dicts = {"stoch": stoch_checkpoint_dict, "det": det_checkpoint_dict}
    else:
        checkpoint_dicts = {"stoch": {}, "det": {}}

    # ------------------------------------------------------------------
    # Plots: disable for now to avoid plotting bugs overnight
    # ------------------------------------------------------------------
    plot_dict = {
        "Single": [None],
        "Overlay": [],
    }

    # ------------------------------------------------------------------
    # Run experiment
    # ------------------------------------------------------------------
    exp_inst = SingleTaskExperiment(
        model_type=model_type,
        seeds=seeds,
        plot_dict=plot_dict,
        model_dict=model_dict,
    )

    exp_inst.run_experiments(
        input_data_dict=input_data_dict,
        epochs=epochs,
        beta_param_dict=beta_param_dict,
        checkpoint_dicts=checkpoint_dicts,
        MC_train=MC_train,
        MC_val=MC_val,
        MC_test=MC_test,
        analysis=analysis,
        save_switch=save_switch,
        save_path=save_path,
        ensemble_epochs=ensemble_epochs,
    )

    print("Single Task Experiment Completed")


# if __name__ == "__main__":
#     save_path = 'results\\single_task_test_run'
#     if os.path.exists(save_path):
#         timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#         new_name = f"{save_path}_{timestamp}"
#         save_path =  new_name
#         print(f"Renamed to: {new_name}")
#     else:
#         print("Folder does not exist.")

#     # Model type
#     model_type = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'MLPDropoutNet', 'DeepEnsembleNet'] 
#     model_type = ['LP_FDNet', 'IC_FDNet', 'DeepEnsembleNet', 'MLPDropoutNet'] 
#     model_type = ['IC_FDNet', 'MLPDropoutNet'] 
#     model_type = ['LP_FDNet', 'IC_FDNet', 'MLPDropoutNet'] 
#     model_type = ['LP_FDNet', 'MLPDropoutNet'] 
#     model_type = ['MLPDropoutNet'] 
#     # model_type = ['GaussHyperNet']
#     # Get model parameters
#     # model_dict = {}
#     # for model in model_type:
#     #     model_dict[model] = build_model_dict(
#     #         model_type=model)
#     # model_type = ['LP_FDNet', 'MLPNet'] 
#     # Seeds
#     seeds = [np.random.randint(1,1e4)]
#     # seeds = [2]
#     # seeds = [2]
#     # Model dict
#     model_dict = False
#     if model_dict:
#         model_dict = {
#         'IC_FDNet' :       {'hidden_dim': 20, 'hyper_hidden_dim': 20},     
#         'LP_FDNet':        {'hidden_dim': 18, 'hyper_hidden_dim': 17},      
#         'HyperNet':        {'hidden_dim': 25, 'hyper_hidden_dim': 9},     
#         'BayesNet':        {'hidden_dim': 166},                            
#         'GaussHyperNet':   {'hidden_dim': 24, 'hyper_hidden_dim': 5, 'latent_dim': 9}, 
#         'MLPNet':          {'hidden_dim': 770, 'dropout_rate': 0.1},      
#         'MLPDropoutNet':   {'hidden_dim': 770, 'dropout_rate': 0.1},       
#         'DeepEnsembleNet': {'hidden_dim': 33, 'dropout_rate': 0.1, 'num_models': 10,
#                             'ensemble_seed_list': [0,1,2,3,4,5,6,7,8,9]}    
#         }
#         model_dict = {model: model_dict[model] for model in model_type}
#     else:
#         model_dict = None
#     # Number of epochs
#     epochs = 10
#     # Number of Monte-Carlo trials used for training, validation, and testing
#     MC_train = 1
#     MC_val = 50
#     MC_test = 500
#     # Number of epochs to train each ensemble model
#     ensemble_epochs = round(epochs/10)
#     # Beta scheduler
#     beta_scheduler = "cosine"
#     # Beta parameters
#     if beta_scheduler != "zero":
#         # Warm up epochs
#         warmup_epochs = round(0.75*epochs)
#         # Beta max
#         beta_max = 0.01
#         # Beta parameter dictionary
#         beta_param_dict = {"beta_scheduler": beta_scheduler,
#                            "warmup_epochs": warmup_epochs, "beta_max": beta_max}
#     elif beta_scheduler == "zero":
#         beta_param_dict = {"beta_scheduler": beta_scheduler}
#     # Perform analysis 
#     analysis = True
#     # Save switch
#     save_switch = True
#     # Training region
#     region_interp = (-0.3,0.3)
#     # Create data
#     # input_type = "uniform"
#     # input_seed = random.randint(100,10000)
#     # Min/ Max of region 
#     x_min = -1
#     x_max = 1
#     # Number of training, validation, and test points
#     n_train = 512
#     n_test = 256
#     # Number of validation points in the interpolation and extrapolation region
#     n_val_interp = 128
#     n_val_extrap = 128
#     # Generate input data split
#     input_data_dict = generate_splits(x_min=x_min, x_max=x_max, region_interp=region_interp, 
#                     n_train=n_train, n_test=n_test, n_val_interp=n_val_interp, n_val_extrap=n_val_extrap, seed=seeds[0])
#     # Perform checkpoint
#     do_checkpoint = True
#     if do_checkpoint:
#         # Stochastic checkpoint dict
#         stoch_checkpoint_dict = {
#         'metric_str': 'mse',
#         'region_interp': region_interp,
#         'min_or_max': 'min',
#         'interp_or_extrap': 'interp'
#         }
#         # Deterministic checkpoint dict
#         det_checkpoint_dict = {
#         'metric_str': 'mse',
#         'region_interp': region_interp,
#         'min_or_max': 'min',
#         'interp_or_extrap': 'interp'
#         }
#         checkpoint_dicts = {'stoch': stoch_checkpoint_dict, 'det': det_checkpoint_dict}
#     else:
#         checkpoint_dicts = {'stoch': {}, 'det': {}}
#     # Plots
#     # plot_dict = {
#     #     "Single": ["loss_vs_epoch", "mean_vs_x", "mse_vs_x", "nlpd_kde_vs_x", "pit_two_panel", "mse_db_vs_var_db", "nll_kde_heatmap" ],
#     #     "Overlay": ["mean_vs_x", "mses_vs_epoch", "nlpd_kde_vs_x", "crps_db_vs_nlpd_kde", "crps_db_vs_nlpd_kde_2x2", "mse_db_vs_var_db_2x2"]
#     # }
#     plot_dict = {
#     "Single": [None],
#     "Overlay": []
#     }

#     # Create experiment class instance
#     exp_inst = SingleTaskExperiment(model_type=model_type, seeds=seeds, plot_dict=plot_dict, model_dict=model_dict)
#     # Run experiment
#     exp_inst.run_experiments(input_data_dict=input_data_dict, epochs=epochs, 
#                                     beta_param_dict=beta_param_dict, 
#                                     checkpoint_dicts=checkpoint_dicts,
#                                     MC_train=MC_train, MC_val=MC_val, MC_test=MC_test,
#                                     analysis=analysis, save_switch=save_switch, save_path=save_path, ensemble_epochs=ensemble_epochs
#                                     )
#     print('Single Task Experiment Completed')
