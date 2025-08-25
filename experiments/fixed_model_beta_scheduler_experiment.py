import os
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

from experiments.base_experiment import BaseExperiment
from data.toy_functions import sample_function
from data.toy_functions import generate_grid 
from utils.loader.fixed_model_beta_scheduler_loader import load_scheduler_metrics, get_metric
from utils.plots.fixed_model_beta_scheduler_plots import plot_training_metrics_overlay, plot_final_metrics_vs_x_overlay

class FixedModelBetaSchedulerExperiment:
    def __init__(self, model_type=None, seeds=None, hidden_dim=32, hyper_hidden_dim=64):
        self.model_types = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'} if model_type is None else model_type.intersection({'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'})
        self.seeds = seeds if seeds is not None else [random.randint(0, 1000) for _ in range(3)]
        self.hidden_dim = hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

        self.kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        self.non_kl_models = {'MLPNet', 'DeepEnsembleNet', 'HyperNet'}
        self.no_variance_models = {'MLPNet', 'HyperNet'}

    def run_experiments(self, x=np.linspace(start=-10,stop=10,num=500),
                            region_interp=(-1,1),
                            frac_train=0.5, epochs=1000,
                            beta_scheduler_types=["constant"], beta_max_arr=[1], warmup_epochs_arr=[500], 
                            num_samples=100, MC=1, analysis=True, save_switch=False, run_analysis=True):
        # Parameters
        model_types = self.model_types
        seeds = self.seeds

        # Make dir
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Date and time stamp for run name
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Allocate dict for data
        metrics_dicts = {}
        for seed in seeds:
            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Allocate seed
            metrics_dicts[seed] = {}

            # Define interpolation and extrapolation region
            ind_interp = np.where((x >= region_interp[0]) & (x <= region_interp[1]))[0]
            ind_extrap = np.where((x < region_interp[0]) | (x > region_interp[1]))[0]
            x_test = torch.tensor(x, dtype=torch.float64).unsqueeze(-1)

            # Training data
            ind_train = np.random.choice(ind_interp, size=round(len(ind_interp)*frac_train), replace=False)
            ind_test = np.array([n for n in range(x.shape[0]) if n not in ind_train])
            x_train = torch.tensor(x[ind_train], dtype=torch.float64).unsqueeze(-1)

            # Generate function
            f, desc = sample_function(seed=seed)

            # Generate outputs
            y_test = torch.tensor(f(x_test), dtype=torch.float64)
            y_train = torch.tensor(f(x_train), dtype=torch.float64)

            # Store training, testing, and metadata
            data = (x_train, y_train, x_test, y_test, desc)
            
            # Generate metrics and save
            for model_type in model_types:
                # Allocate model
                metrics_dicts[seed][model_type] = {}
                for beta_scheduler in beta_scheduler_types:
                    # Allocate beta scheduler
                    metrics_dicts[seed][model_type][beta_scheduler] = {}
                    for beta_max in beta_max_arr:
                        # Allocate beta max
                        metrics_dicts[seed][model_type][beta_scheduler][beta_max] = {}
                        for warmup_epochs in warmup_epochs_arr:
                            # Beta dictionary
                            beta_param_dict = {"beta_scheduler": beta_scheduler,
                                            "warmup_epochs": warmup_epochs, "beta_max": beta_max}

                            # Base experiment instance
                            exp_inst = BaseExperiment(model_type=model_type, 
                                                    seed=seed, 
                                                    hidden_dim=self.hidden_dim, 
                                                    hyper_hidden_dim=self.hyper_hidden_dim
                                                    )

                            # Run experiment
                            preds, data, training_time, metric_outputs, trainer = exp_inst.run_experiments(
                                    data=data,
                                    epochs=epochs,
                                    beta_param_dict=beta_param_dict,
                                    MC_test=num_samples,
                                    MC_train = MC
                                    )
                            
                            mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nll = metric_outputs

                            if analysis:
                                metrics_dict = {
                                    "epoch": list(range(epochs)),
                                    "losses_per_epoch": trainer.losses,
                                    "mse_per_epoch": trainer.mses,
                                    "kls_per_epoch": trainer.kls,
                                    "beta_per_epoch": trainer.betas,
                                    "final_mean": mean,
                                    "final_variance": var,
                                    "final_bias": bias,
                                    "final_residual": res_prec,
                                    "final_mse": mse,
                                    "beta_scheduler": beta_scheduler,
                                    "beta_max": beta_max,
                                    "warmup_epochs": warmup_epochs,
                                    "seed": seed,
                                    "model_type": model_type,
                                    "x_values": x.cpu().numpy() if torch.is_tensor(x) else x,
                                    "y_values": y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test,
                                    "interp_region": region_interp,
                                    "frac_train": frac_train,
                                    "MC": MC,
                                    "num_samples": num_samples,
                                    "desc": desc,
                                    "x_train": x_train,
                                    "y_train": y_train
                                }
                                # Store data
                                metrics_dicts[seed][model_type][beta_scheduler][beta_max][warmup_epochs] = metrics_dict


        # Save metrics dict
        if save_switch:
            save_dir = os.path.join("results", "fixed_model_beta_scheduler_experiment", date_time.replace('-','_'), f"seed{seed}")
            os.makedirs(save_dir, exist_ok=True)
            np.savez_compressed(os.path.join(save_dir, "metrics_dict.npz"), metrics_dicts=metrics_dicts)

        if run_analysis:
            # Plot metrics vs x
            plot_final_metrics_vs_x_overlay(metrics_dicts=metrics_dicts, seeds=seeds, beta_scheduler_types=beta_scheduler_types, beta_max_arr=beta_max_arr, warmup_epochs_arr=warmup_epochs_arr, save_dir=os.path.join(save_dir,'plots'))
            # Plot training metrics
            plot_training_metrics_overlay(metrics_dicts=metrics_dicts, seeds=seeds, beta_scheduler_types=beta_scheduler_types, beta_max_arr=beta_max_arr, warmup_epochs_arr=warmup_epochs_arr, save_dir=os.path.join(save_dir,'plots'))


    def run_analysis(self, metrics_dicts=None, save_dir=None, run_name=None):
        """
        Load or reuse the metrics dict and dispatch plots.

        Args:
            metrics_dicts (dict or None): If None, loads from run_name or save_dir.
            save_dir (str or None): Path to directory with metrics_dict.npz.
            run_name (str or None): Optional run name string to infer save_dir.
        """
        # Infer save_dir if needed 
        if save_dir is None and run_name is not None:
            save_dir = os.path.join("results", "fixed_model_beta_scheduler_experiment", run_name)

        # Load if not passed
        if metrics_dicts is None:
            assert save_dir is not None, "You must specify save_dir or provide metrics_dicts"
            path = os.path.join(save_dir, "metrics_dict.npz")
            metrics_dicts = np.load(path, allow_pickle=True)["metrics_dicts"].item()


    def get_capabilities(self, model_type):
        capabilities = {"mean", "bias"}  # always
        if model_type not in self.no_variance_models:
            capabilities |= {"residuals", "variance", "nll"}
        return capabilities
    
    def format_run_path(self, seed, model_type, beta_scheduler, beta_max, warmup_epochs):
        param_str = f"{beta_scheduler}_beta{beta_max:.3f}_warmup{int(warmup_epochs)}"
        return os.path.join("results", "fixed_model_beta_scheduler_experiment", f"seed_{seed}", model_type, param_str)


if __name__ == "__main__":
    # Model type
    model_type = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'} 
    # Seeds
    seeds = [random.randint(100,10000) for _ in range(1)]
    # Number of epochs
    epochs = 10
    # Number of samples used in inference
    num_samples = 100
    # Number of Monte-Carlo trials used for training
    MC = 1
    # Beta scheduler types
    beta_scheduler_types = ["linear", "cosine", "sigmoid", "constant"]
    # Beta max array
    beta_max_arr = np.linspace(start=1, stop=1, num=1)
    # Warmup epochs array
    warmup_epochs_arr = np.unique(np.round(np.array([0.5])*epochs))
    # Perform analysis 
    analysis = True
    # Save switch
    save_switch = True
    # Training region
    region_interp = (-1,1)
    # Create data
    input_type = "uniform_random"
    input_seed = random.randint(100,10000)
    x_min = -100
    x_max = 100
    n_interp = 10
    n_extrap = 100
    x = generate_grid(input_type=input_type, input_seed=input_seed, x_min=x_min, x_max=x_max, region_interp=region_interp, n_interp=n_interp, n_extrap=n_extrap)
    # Fraction of points of data points in region used for training
    frac_train = 0.5
    # Create experiment class instance
    exp_class = FixedModelBetaSchedulerExperiment(model_type=model_type, seeds=seeds)
    # Run experiment
    exp_class.run_experiments(x=x, region_interp=region_interp, frac_train=frac_train,
                                epochs=epochs, beta_scheduler_types=beta_scheduler_types, 
                                beta_max_arr=beta_max_arr, warmup_epochs_arr=warmup_epochs_arr,
                                num_samples=num_samples, MC=MC, analysis=analysis, save_switch=save_switch
                                    )
    print('Fixed Model Beta Scheduler Experiment Completed')
