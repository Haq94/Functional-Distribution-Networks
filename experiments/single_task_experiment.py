import os
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

from experiments.base_experiment import BaseExperiment
from data.toy_functions import sample_function
from data.toy_functions import generate_grid
from utils.saver.single_task_saver import single_task_saver
from utils.metrics import get_summary
from utils.plots.single_task_plots import single_task_plots, plot_single_task_overlay
from utils.loader.single_task_loader import single_task_overlay_loader
from utils.general import set_determinism, build_model_dict

class SingleTaskExperiment:
    def __init__(self, model_type=None, seeds=None, model_dict=None, plot_dict=None):
        self.model_types = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'MLPDropoutNet', 'DeepEnsembleNet'] if model_type is None else model_type
        self.seeds = seeds if seeds is not None else [random.randint(0, 1000) for _ in range(3)]
        self.plot_dict = {"Single": [], "Overlay": []} if plot_dict is None else plot_dict
        # self.hidden_dim = hidden_dim
        # self.hyper_hidden_dim = hyper_hidden_dim
        if model_dict is None or self.model_type != list(model_dict.keys()):
            self.model_dict = {}
            for model in self.model_types:
                self.model_dict[model] = build_model_dict(model_type=model)
        else:
            self.model_dict = model_dict

        # self.kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        # self.non_kl_models = {'MLPNet', 'DeepEnsembleNet', 'HyperNet'}
        # self.no_variance_models = {'MLPNet', 'HyperNet'}

        kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        stoch_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'DeepEnsembleNet', 'MLPDropoutNet'}
        mc_model = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'MLPDropoutNet'}
        self.is_stoch = [m in stoch_models for m in self.model_types]
        self.kl_exist = [m in kl_models for m in self.model_types]
        self.training_type = ['MC' if m in mc_model else 'Ensemble' if m == 'DeepEnsembleNet' else 'Deterministic' for m in self.model_types]

    def run_experiments(self, 
            x=np.linspace(start=-10,stop=10,num=500), region_interp=(-1,1), frac_train=0.5, frac_val=0.2, data_dict=None,
            epochs=1000, beta_param_dict=None, checkpoint_dicts=None,
            MC_train=1, MC_val=100, MC_test=50, analysis=True, 
            save_path=None, save_switch=False, ensemble_switch=True):
        # Parameters
        model_types = self.model_types
        model_dict = self.model_dict
        seeds = self.seeds

        # if 'DeepEnsembleNet' in model_types and num_models is None:
        #     num_models = MC_test

        if checkpoint_dicts is None:
            # Stochastic checkpoint dict
            stoch_checkpoint_dict = {
            'metric_str': 'var',
            'region_interp': region_interp,
            'min_or_max': 'max',
            'interp_or_extrap': 'extrap'
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
        # results_dir = "results"
        os.makedirs(save_path, exist_ok=True)
        
        # Date and time stamp for run name
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for seed in tqdm(seeds, leave=False):
            # Set seed
            set_determinism(seed=seed)
            # Seed save path
            seed_save_path = os.path.join(save_path, f'seed_{seed}')
            # Generate function
            f, desc = sample_function(seed=seed)
            # Load inputs and relevant field if it exists
            if data_dict is not None:
                region_interp = data_dict['region_interp']
                ind_interp = data_dict['ind_interp']
                ind_extrap = data_dict['ind_extrap']
                ind_train = data_dict['ind_train']
                ind_val = data_dict['ind_val']
                ind_test = data_dict['ind_test']
                x = data_dict['x']
                x_train = data_dict['x_train']
                x_val = data_dict['x_val']
                x_test = data_dict['x_test']
            else:
                # Define interpolation and extrapolation region
                ind_interp = np.where((x >= region_interp[0]) & (x <= region_interp[1]))[0]
                ind_extrap = np.where((x < region_interp[0]) | (x > region_interp[1]))[0]
                # Seperate validation data set
                ind_val = np.concatenate([np.random.choice(ind_interp, size=int(frac_val*x.shape[0]/2), replace=False),
                    np.random.choice(ind_extrap, size=int(frac_val*x.shape[0]/2), replace=False)])
                x_val = x[ind_val]
                x = np.setdiff1d(x, x_val)
                ind_interp = np.where((x >= region_interp[0]) & (x <= region_interp[1]))[0]
                ind_extrap = np.where((x < region_interp[0]) | (x > region_interp[1]))[0]
                x_test = torch.tensor(x, dtype=torch.float64).unsqueeze(-1)
                # Training data
                ind_train = np.random.choice(ind_interp, size=round(len(ind_interp)*frac_train), replace=False)
                ind_test = np.setdiff1d(range(x.shape[0]), ind_train)
                x_train = torch.tensor(x[ind_train], dtype=torch.float64).unsqueeze(-1)

            # Generate outputs
            y_train = torch.tensor(f(x_train), dtype=torch.float64)
            y_val = torch.tensor(f(x_val), dtype=torch.float64)
            y_test = torch.tensor(f(x_test), dtype=torch.float64)

            for model_type in tqdm(model_types):
                # Check if the model is stochastic
                is_stoch = self.is_stoch[model_types == model_type]
                # Check if kl divergence exist
                kl_exist = self.kl_exist[model_types == model_type]
                # Pick correct checkpoint condition
                checkpoint_dict = checkpoint_dicts['stoch'] if is_stoch else checkpoint_dicts['det']

                # Base experiment instance
                exp_inst = BaseExperiment(model_type=model_type, seed=seed, **model_dict[model_type], save_path=seed_save_path)
                # Choose number of epochs for Deep Ensemble 
                is_de = (model_type == 'DeepEnsembleNet')
                epochs_to_use = epochs if not (is_de and ensemble_switch) else max(1, epochs // model_dict[model_type]['num_models'])

                # Run experiment
                data = (x_train, y_train, x_val, y_val, x_test, y_test, desc)
                preds, data_dict, training_time, metric_outputs, trainer = exp_inst.run_experiments(
                        data=data, epochs=epochs_to_use, beta_param_dict=beta_param_dict,
                        MC_train=MC_train, MC_val = MC_val, MC_test=MC_test, 
                        checkpoint_dict=checkpoint_dict, save_switch=save_switch
                        )

                if analysis:    
                    # Plot and save visuals
                    single_plot_save_path = None if save_path is None else os.path.join(seed_save_path, model_type, "plots")
                    # plot_types = self.get_plot_types(model_type)
                    single_task_plots(trainer, preds, x_train, y_train, x_test, y_test, ind_train, ind_test,
                    ind_interp, ind_extrap, region_interp, metric_outputs=metric_outputs, kl_exist=kl_exist,
                    is_stoch=is_stoch, block=False, save_dir=single_plot_save_path, plot_types=self.plot_dict['Single'])

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
                plot_single_task_overlay(seed_save_path, model_types, stoch_models, stoch_metrics, 
                model_colors, show_figs=True, use_db_scale=True, plot_types=self.plot_dict['Overlay'])
                
            print(f"Completed: {model_type} | seed: {seed} | training time: {training_time}s")

    # def get_plot_types(self, model_type):
    #     kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
    #     stoch_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'DeepEnsembleNet', 'MLPDropoutNet'}
    #     # mc_model = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'MLPDropoutNet'}
    #     plot_types = ['mse_vs_epoch', 'mean_vs_x', 'bias_vs_x']
    #     if model_type in kl_models:
    #         plot_types.extend(['kl_vs_epoch', 'loss_vs_epoch'])
    #     if model_type in stoch_models:
    #         plot_types.extend(['residulas_vs_x', 'mean_vs_x', 'pit_two_panel', 'pdf_heatmap', 'mse_vs_x', 'nlpd_vs_x', 'crps_vs_x', 'mse_db_vs_var_db', 'mse_db_vs_var_db_2x2', 'bias_sq_db_vs_var_db', 'bias_sq_db_vs_var_db_2x2', 'nlpd_kde_vs_var_db', 'nlpd_kde_vs_var_db_2x2', 'nlpd_hist_vs_var_db', 'nlpd_hist_vs_var_db_2x2', 'crps_db_vs_var_db', 'crps_db_vs_var_db_2x2', 'bias_sq_db_vs_mse_db', 'bias_sq_db_vs_mse_db_2x2', 'nlpd_kde_vs_mse_db', 'nlpd_kde_vs_mse_db_2x2', 'nlpd_hist_vs_mse_db', 'nlpd_hist_vs_mse_db_2x2', 'crps_db_vs_mse_db', 'crps_db_vs_mse_db_2x2', 'nlpd_kde_vs_bias_sq_db', 'nlpd_kde_vs_bias_sq_db_2x2', 'nlpd_hist_vs_bias_sq_db', 'nlpd_hist_vs_bias_sq_db_2x2', 'crps_db_vs_bias_sq_db', 'crps_db_vs_bias_sq_db_2x2', 'crps_db_vs_nlpd_kde', 'crps_db_vs_nlpd_kde_2x2', 'crps_db_vs_nlpd_hist', 'crps_db_vs_nlpd_hist_2x2'])
    #     return plot_types

        # capabilities = {"mean", "bias"}  # always
        # if model_type not in self.no_variance_models:
        #     capabilities |= {"residuals", "variance", "nlpd"}
        # return capabilities

if __name__ == "__main__":
    save_path = 'results\\delete'
    # Model type
    model_type = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'MLPDropoutNet', 'DeepEnsembleNet'] 
    model_type = ['LP_FDNet', 'DeepEnsembleNet', 'MLPNet', 'GaussHyperNet'] 
    # model_type = ['GaussHyperNet']
    # Get model parameters
    model_dict = {}
    for model in model_type:
        model_dict[model] = build_model_dict(
            model_type=model)
    # model_type = ['LP_FDNet', 'MLPNet'] 
    # Seeds
    seeds = [random.randint(100,10000) for _ in range(1)]
    # Number of epochs
    epochs = 10
    # Number of Monte-Carlo trials used for training, validation, and testing
    MC_train = 1
    MC_val = 100
    MC_test = 50
    # # Number of models for Deep Ensemble
    # num_models = 5
    # Ensemble switch (if True divide the number of epochs by the number of models)
    ensemble_switch = True
    # Beta scheduler
    beta_scheduler = "linear"
    # Beta parameters
    if beta_scheduler == "linear":
        # Warm up epochs
        warmup_epochs = round(0.5*epochs)
        # Beta max
        beta_max = 1.0
        # Beta parameter dictionary
        beta_param_dict = {"beta_scheduler": beta_scheduler,
                           "warmup_epochs": warmup_epochs, "beta_max": beta_max}
    elif beta_scheduler == "zero":
        beta_param_dict = {"beta_scheduler": beta_scheduler}
    # Perform analysis 
    analysis = True
    # Save switch
    save_switch = True
    # Training region
    region_interp = (-1,1)
    # Create data
    input_type = "uniform"
    input_seed = random.randint(100,10000)
    x_min = -3
    x_max = 3
    n_interp = 20
    n_extrap = 40
    x = generate_grid(input_type=input_type, input_seed=input_seed, x_min=x_min, x_max=x_max, region_interp=region_interp, n_interp=n_interp, n_extrap=n_extrap)
    # Fraction of training points (relative to number of interpolation points)
    frac_train = 0.5
    # Fraction of validation points
    frac_val = 0.2
    # Stochastic checkpoint dict
    stoch_checkpoint_dict = {
    'metric_str': 'var',
    'region_interp': region_interp,
    'min_or_max': 'max',
    'interp_or_extrap': 'extrap'
    }
    # Deterministic checkpoint dict
    det_checkpoint_dict = {
    'metric_str': 'mse',
    'region_interp': region_interp,
    'min_or_max': 'min',
    'interp_or_extrap': 'interp'
    }
    checkpoint_dicts = {'stoch': stoch_checkpoint_dict, 'det': det_checkpoint_dict}
    # Plots
    plot_dict = {
        "Single": ["loss_vs_epoch", "mean_vs_x", "mse_vs_x", "nlpd_kde_vs_x", "pit_two_panel", "mse_db_vs_var_db" ],
        "Overlay": []
    }

    # Create experiment class instance
    exp_inst = SingleTaskExperiment(model_type=model_type, seeds=seeds, plot_dict=plot_dict)
    # Run experiment
    exp_inst.run_experiments(x=x, region_interp=region_interp, frac_train=frac_train, frac_val=frac_val,
                                epochs=epochs, beta_param_dict=beta_param_dict, checkpoint_dicts=checkpoint_dicts,
                                    MC_train=MC_train, MC_val=MC_val, MC_test=MC_test,
                                    analysis=analysis, save_switch=save_switch, save_path=save_path, ensemble_switch=ensemble_switch
                                    )
    print('Single Task Experiment Completed')
