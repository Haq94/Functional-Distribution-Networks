import os
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

from experiments.base_experiment import BaseExperiment
from data.toy_functions import sample_function
from utils.saver.single_task_saver import single_task_saver
from utils.metrics import get_summary
from utils.plots.single_task_plots import single_task_plots

class SingleTaskExperiment:
    def __init__(self, model_type=None, seeds=None, hidden_dim=32, hyper_hidden_dim=64):
        self.model_types = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'DeepEnsembleNet'] if model_type is None else model_type
        self.seeds = seeds if seeds is not None else [random.randint(0, 1000) for _ in range(3)]
        self.hidden_dim = hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

        self.kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        self.non_kl_models = {'MLPNet', 'DeepEnsembleNet', 'HyperNet'}
        self.no_variance_models = {'MLPNet', 'HyperNet'}

    def run_experiments(self, x=np.linspace(start=-10,stop=10,num=500),
                            region_interp=(-1,1),
                            frac_train=0.5, epochs=1000, beta_param_dict=None, 
                            num_samples=100, MC=1, num_models=None, analysis=True, save_switch=False):
        # Parameters
        model_types = self.model_types
        seeds = self.seeds

        if model_types == 'DeepEnsembleNet' and num_models is None:
            num_models = num_samples

        # Make dir
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Define interpolation and extrapolation region
        ind_interp = np.where((x >= region_interp[0]) & (x <= region_interp[1]))[0]
        ind_extrap = np.where((x < region_interp[0]) | (x > region_interp[1]))[0]
        x_test = torch.tensor(x, dtype=torch.float64).unsqueeze(-1)
        
        # Date and time stamp for run name
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for model_type in tqdm(model_types):
            for seed in tqdm(seeds, leave=False):
                # Set seed
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Run name
                run_name = f"{model_type}_seed{seed}_{date_time}"

                # Pre-allocate save dir
                save_dir = None

                # Training data
                ind_train = np.random.choice(ind_interp, size=round(len(ind_interp)*frac_train), replace=False)
                ind_test = np.array([n for n in range(x.shape[0]) if n not in ind_train])
                x_train = torch.tensor(x[ind_train], dtype=torch.float64).unsqueeze(-1)

                # Generate function
                f, desc = sample_function(seed=seed)

                # Generate outputs
                y_test = torch.tensor(f(x_test), dtype=torch.float64)
                y_train = torch.tensor(f(x_train), dtype=torch.float64)

                # Base experiment instance
                exp_inst = BaseExperiment(model_type=model_type, 
                                          seed=seed, 
                                          hidden_dim=self.hidden_dim, 
                                          hyper_hidden_dim=self.hyper_hidden_dim, 
                                          num_models=num_models
                                          )

                # Run experiment
                data = (x_train, y_train, x_test, y_test, desc)
                preds, data, training_time, metric_outputs, trainer = exp_inst.run_experiments(
                        data=data,
                        epochs=epochs,
                        beta_param_dict=beta_param_dict,
                        num_samples=num_samples,
                        MC = MC
                        )

                if analysis:    
                    if save_switch:
                        # Create save dir
                        save_dir = os.path.join("results", 'single_task_experiment', model_type, run_name)
                        
                        # Generate summary
                        summary = get_summary(metric_outputs, y_test, trainer.model, desc, seed, training_time, epochs, beta_param_dict, x, region_interp, frac_train)

                        # Save experiment output
                        single_task_saver(metric_outputs, trainer.model, trainer, summary, x_train, y_train, x_test, y_test, save_dir)

                    # Plot and save visuals
                    name = desc + ', Model: ' + model_type
                    plot_save_dir = None if save_dir is None else os.path.join(save_dir, "plots")
                    capabilities = self.get_capabilities(model_type)
                    single_task_plots(trainer, preds, x_train, y_train, x_test, y_test, name, ind_train, region_interp, metric_outputs=metric_outputs, block=False, save_dir=plot_save_dir, capabilities=capabilities)
                    
            print(f"Completed: {model_type} | seed: {seed} | training time: {training_time}s")

    def get_capabilities(self, model_type):
        capabilities = {"mean", "bias"}  # always
        if model_type not in self.no_variance_models:
            capabilities |= {"residuals", "variance", "nll"}
        return capabilities

if __name__ == "__main__":
    # Model type
    model_type = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'DeepEnsembleNet'] 
    # model_type = ['DeepEnsembleNet', 'MLPNet', 'LP_FDNet'] 
    # model_type = ['LP_FDNet'] 
    # Seeds
    seeds = [random.randint(100,10000) for _ in range(1)]
    # Number of epochs
    epochs = 2
    # Number of samples used in inference
    num_samples = 100
    # Number of Monte-Carlo trials used for training
    MC = 2
    # Number of models for Deep Ensemble
    num_models = 10
    # Beta scheduler
    beta_scheduler = "linear"
    # Beta parameters
    if beta_scheduler == "linear":
        # Warm up epochs
        warmup_epochs = round(epochs/2)
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
    exp_class = SingleTaskExperiment(model_type=model_type, seeds=seeds)
    # Run experiment
    exp_class.run_experiments(x=x, region_interp=region_interp, frac_train=frac_train,
                                epochs=epochs, beta_param_dict=beta_param_dict,
                                    num_samples=num_samples, MC=MC, num_models=num_models,
                                    analysis=analysis, save_switch=save_switch
                                    )
    print('Single Task Experiment Completed')
