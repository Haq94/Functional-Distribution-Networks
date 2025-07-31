import os
import torch
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

from experiments.base_experiment import BaseExperiment
from data.toy_functions import sample_function
from utils.saver import save_experiment_outputs
from utils.metrics import get_summary
from utils.plots import single_task_regression_plots

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

            data = (x_train, y_train, x_test, y_test, desc)

            for model_type in model_types:
                for beta_scheduler in beta_scheduler_types:
                    for beta_max in beta_max_arr:
                        for warmup_epochs in warmup_epochs_arr:
                        
                            # Beta dictionary
                            beta_param_dict = {"beta_scheduler": beta_scheduler,
                                            "warmup_epochs": warmup_epochs, "beta_max": beta_max}

                            # Run name
                            run_name = f"{model_type}_seed{seed}_{beta_scheduler}_{warmup_epochs}_{beta_max}_{date_time}"

                            # Pre-allocate save dir
                            save_dir = None

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
                                    "final_mse": mse,
                                    "final_bias": bias,
                                    "beta_scheduler": beta_scheduler,
                                    "beta_max": beta_max,
                                    "warmup_epochs": warmup_epochs,
                                    "seed": seed
                                }
                                
                                if save_switch:
                                    # Create save dir
                                    save_dir = os.path.join("results", 'fixed_model_beta_scheduler_experiment', model_type, f"seed{seed}")
                                    np.savez(os.path.join(save_dir, "metrics.npz"), **metrics_dict)
                                #     # Generate summary
                                #     summary = get_summary(metric_outputs, y_test, trainer.model, desc, seed, training_time, epochs, beta_param_dict, x, region_interp, frac_train)

                                #     # Save experiment output
                                #     save_experiment_outputs(metric_outputs, trainer.model, trainer, summary, x_train, y_train, x_test, y_test, save_dir)

                                # # Plot and save visuals
                                # name = desc + ', Model: ' + model_type
                                # plot_save_dir = None if save_dir is None else os.path.join(save_dir, "plots")
                                # capabilities = self.get_capabilities(model_type)
                                # single_task_regression_plots(trainer, preds, x_train, y_train, x_test, y_test, name, ind_train, region_interp, metric_outputs=metric_outputs, block=False, save_dir=plot_save_dir, capabilities=capabilities)
                                
                print(f"Completed: {model_type} | seed: {seed} | training time: {training_time}s")

    def get_capabilities(self, model_type):
        capabilities = {"mean", "bias"}  # always
        if model_type not in self.no_variance_models:
            capabilities |= {"residuals", "variance", "nll"}
        return capabilities
    
    def get_beta_scheduler_dict(self):
        pass

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
    beta_max_arr = np.linspace(start=0.1, stop=1, num=10)
    # Warmup epochs array
    warmup_epochs_arr = np.unique(np.round(np.array([0.05, 0.1, 0.5, 0.9])*epochs))
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

# OLD CODE ========================================================================================================

# def Bias_Var_Debug_Func(preds, y_t_np):

#     N = preds.shape[0]
#     S = preds.shape[1]

#     mean = np.zeros(S)
#     var = np.zeros(S)
#     bias = np.zeros(S)
#     mse = np.zeros(S)
#     bias_var_diff = np.zeros(S)

#     for n in range(S):
#         yn = preds[:,n,:]
#         yn = yn.squeeze()
#         yn = yn.astype(np.float64)

#         truth_n = y_t_np[n]
#         truth_n = truth_n.astype(np.float64)

#         mean_n = yn.mean()
#         var_n = yn.var()
#         bias_n = mean_n - truth_n
#         mse_n = np.mean((yn - truth_n)**2)
#         bias_var_diff_n = abs(mse_n - (bias_n**2 + var_n))

#         mean[n] = mean_n
#         var[n] = var_n
#         bias[n] = bias_n
#         mse[n] = mse_n
#         bias_var_diff[n] = bias_var_diff_n  

#     return mean, var, bias, mse, bias_var_diff

# def plots(preds, x_c, y_c, x_t, y_t, desc, ind_c, block=False):
    
#     import matplotlib.pyplot as plt
#     # Convert all arrays from torch tensors to numpy arrays
#     x_c_np = x_c.cpu().numpy().squeeze().astype(np.float64)
#     y_c_np = y_c.cpu().numpy().squeeze().astype(np.float64)
#     x_t_np = x_t.cpu().numpy().squeeze().astype(np.float64)
#     y_t_np = y_t.cpu().numpy().squeeze().astype(np.float64)
#     preds_np = preds.astype(np.float64)
#     x_c_min = x_c_np.min()
#     x_c_max = x_c_np.max()

#     mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nll = metrics(preds_np, y_t_np)

#     # # DEBUG
#     # mean_db, var_db, bias_db, mse_db, bias_var_diff_db = Bias_Var_Debug_Func(preds, y_t_np)

#     # plt.figure(figsize=(8, 4))
#     # plt.plot(x_t_np, bias_var_diff)
#     # plt.title("Array Plot")
#     # plt.xlabel("Index")
#     # plt.ylabel("Value")
#     # plt.legend()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # # plt.show()
    

#     # plt.figure(figsize=(8, 4))
#     # plt.plot(mse, label='MSE')
#     # plt.plot(var.squeeze(), label='Var')
#     # plt.plot(mse - var.squeeze(), label="Diff")
#     # plt.title("Array Plot")
#     # plt.xlabel("Index")
#     # plt.ylabel("Value")
#     # plt.legend()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # # plt.show()


#     # plt.figure(figsize=(8, 4))
#     # plt.plot(mse, label='MSE')
#     # plt.plot(var.squeeze() + bias**2, label='Var+Bias**2')
#     # plt.plot(mse - (var.squeeze() + bias**2), label="Diff")
#     # plt.title("Array Plot")
#     # plt.xlabel("Index")
#     # plt.ylabel("Value")
#     # plt.legend()
#     # plt.grid(True)
#     # plt.tight_layout()
#     # # plt.show()

#     #####################

#     # Residual Scatter Plots 
#     _, axs = plt.subplots(2, 1, num="Residual Scatter Plot", figsize=(10, 8), sharex=True)

#     for ii in range(res_prec.shape[1]):
#         axs[0].scatter(x_t, res_prec[:, ii], alpha=0.05, color=np.random.rand(3))
#     axs[0].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
#     axs[0].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
#     axs[0].set_title(f"Residual Precision Task: {desc}")
#     axs[0].legend()
#     axs[0].grid(True)

#     for ii in range(res_acc.shape[1]):
#         axs[1].scatter(x_t, res_acc[:, ii], alpha=0.05, color=np.random.rand(3))
#     axs[1].plot(x_t_np, bias, label='Bias', color='red')
#     axs[1].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
#     axs[1].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
#     axs[1].set_title(f"Residual Accuracy Task: {desc}")
#     axs[1].legend()
#     axs[1].grid(True)

#     plt.tight_layout()
#     plt.show(block=block)

#     # Mean Prediction Plot
#     plt.figure(num="Mean Prediction Plot", figsize=(10, 8))
#     for ii in range(res_acc.shape[1]):
#         plt.scatter(x_t, preds[ii, :], alpha=0.05, color=np.random.rand(3))
#     plt.plot(x_t_np, y_t_np, label="Ground Truth", linestyle="--")
#     plt.plot(x_t_np, mean, label="Mean Prediction")
#     plt.fill_between(x_t_np, mean - std, mean + std,
#                     alpha=0.3, label="±1 Std Dev")
#     plt.scatter(x_c_np, y_c_np, color="red", label="Context Points")
#     plt.axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
#     plt.axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
#     plt.title(f"Mean Function Task: {desc}")
#     plt.legend()
#     plt.grid(True)
#     plt.show(block=block)

#     # Zoomed in Mean Prediction Plot
#     plt.figure(num="Zoomed in Mean Prediction Plot", figsize=(10, 8))
#     for ii in range(res_acc.shape[1]):
#         plt.scatter(x_t, preds[ii, :], alpha=0.05, color=np.random.rand(3))
#     plt.plot(x_t_np, y_t_np, label="Ground Truth", linestyle="--")
#     plt.plot(x_t_np, mean, label="Mean Prediction")
#     plt.fill_between(x_t_np, mean - std, mean + std, alpha=0.3, label="±1 Std Dev")
#     plt.scatter(x_c_np, y_c_np, color="red", alpha=0.5, label="Context Points")
#     plt.axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
#     plt.axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.ylim(mean.min(), mean.max())
#     plt.title(f'Mean Function Task: {desc} (Zoomed In)')
#     plt.legend()
#     plt.show(block=block)

#     # Variance and Mean Plot
#     _, axs1 = plt.subplots(2, 1, num="Variance and Mean Plot", figsize=(10, 8), sharex=True)

#     axs1[0].plot(x_t_np, mean)
#     axs1[0].scatter(x_t_np[ind_c], mean[ind_c], label='Context Points', color='red')
#     axs1[0].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
#     axs1[0].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
#     axs1[0].set_title(f"Mean Function Task: {desc}")
#     axs1[0].legend()
#     axs1[0].grid(True)

#     axs1[1].plot(x_t_np, 10*np.log10(var))
#     axs1[1].scatter(x_t_np[ind_c], 10*np.log10(var[ind_c]),label='Context Points', color='red')
#     axs1[1].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
#     axs1[1].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
#     axs1[1].set_title(f"Variance (dB) Task: {desc}")
#     axs1[1].legend()
#     axs1[1].grid(True)

#     plt.tight_layout()
#     plt.show(block=block)

#     # MSE and Bias Plot
#     _, axs2 = plt.subplots(2, 1, num="Bias and MSE Plot", figsize=(10, 8), sharex=True)

#     axs2[0].plot(x_t_np, bias)
#     axs2[0].scatter(x_t_np[ind_c], bias[ind_c], label='Context Points', color='red')
#     axs2[0].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
#     axs2[0].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
#     axs2[0].set_title(f"Bias Task: {desc}")
#     axs2[0].legend()
#     axs2[0].grid(True)

#     axs2[1].plot(x_t_np, 10*np.log10(mse))
#     axs2[1].scatter(x_t_np[ind_c], 10*np.log10(mse[ind_c]), label='Context Points', color='red')
#     axs2[1].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
#     axs2[1].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
#     axs2[1].set_title(f"MSE (dB) Task: {desc}")
#     axs2[1].legend()
#     axs2[1].grid(True)

#     plt.tight_layout()
#     plt.show(block=block)

#     if block == False:
#         plt.pause(0.5) 
#     plt.close('all')

#     print('stop')