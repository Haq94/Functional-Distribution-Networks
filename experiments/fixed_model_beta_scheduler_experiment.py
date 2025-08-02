import os
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

from experiments.base_experiment import BaseExperiment
from data.toy_functions import sample_function
from utils.loader.fixed_model_beta_scheduler_loader import load_all_scheduler_metrics
from utils.plots.fixed_model_beta_scheduler_plots import plot_mse_vs_beta

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
                            num_samples=100, MC=1, analysis=True, save_switch=False):
        # Parameters
        model_types = self.model_types
        seeds = self.seeds

        # Make dir
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Date and time stamp for run name
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for seed in seeds:
            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

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
                for beta_scheduler in beta_scheduler_types:
                    for beta_max in beta_max_arr:
                        for warmup_epochs in warmup_epochs_arr:
                            # Beta dictionary
                            beta_param_dict = {"beta_scheduler": beta_scheduler,
                                            "warmup_epochs": warmup_epochs, "beta_max": beta_max}

                            # Save dir
                            save_dir = self.format_run_path(seed, model_type, beta_scheduler, beta_max, warmup_epochs)

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
                                    num_samples=num_samples,
                                    MC = MC
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
                                    "x_values": x.cpu().numpy() if torch.is_tensor(x) else x  # save input domain
                                }

                                if save_switch:
                                    # Save metrics
                                    os.makedirs(save_dir, exist_ok=True)
                                    np.savez(os.path.join(save_dir, "metrics.npz"), **metrics_dict)
        
        # Run analysis
        self.run_analysis()

    def run_analysis(self):
        # Parameters+
        
        model_types = self.model_types
        seeds = self.seeds
        # Load metrics
        data = load_all_scheduler_metrics()
        # Only keep data with the correct seed and model type
        data = data[seeds]
        # Perform Loss, MSE, KL, and Beta plots per model
        plot_mse_vs_beta(data, model_type="IC_FDNet", warmup_epochs=50)

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
    epochs = 100
    # Number of samples used in inference
    num_samples = 100
    # Number of Monte-Carlo trials used for training
    MC = 2
    # Beta scheduler types
    beta_scheduler_types = ["linear", "cosine", "sigmoid", "constant"]
    # Beta max array
    beta_max_arr = np.linspace(start=0.1, stop=1, num=1)
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
    if input_type == "uniform_random":
        np.random.seed(input_seed)
        x_l = np.random.uniform(low=x_min,high=region_interp[0],size=round(n_extrap/2)) 
        x_c = np.random.uniform(low=region_interp[0],high=region_interp[1],size=n_interp)
        x_r = np.random.uniform(low=region_interp[1],high=x_max,size=n_extrap-round(n_extrap/2))
    else:
        x_l = np.linspace(start=x_min,stop=region_interp[0],num=round(n_extrap/2))
        x_c = np.linspace(start=region_interp[0],stop=region_interp[1],num=n_interp+2)
        x_r = np.linspace(start=region_interp[1],stop=x_max,num=n_extrap-round(n_extrap/2))
    x = np.sort(np.unique(np.concatenate([x_l, x_c, x_r])))
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
