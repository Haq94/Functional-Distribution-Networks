import os
import json
import torch
import numpy as np
import random
from datetime import datetime

from data.toy_functions import sample_function
from models.fdnet import IC_FDNetwork, LP_FDNetwork
from models.hypernet import HyperNetwork
from models.bayesnet import BayesNetwork
from models.gausshypernet import GaussianHyperNetwork
from models.mlpnet import DeterministicMLPNetwork
from models.deepensemblenet import DeepEnsembleNetwork
from training.SingleTaskTrainer import SingleTaskTrainer
from utils.saver import save_analysis_arrays
from utils.metrics import metrics
from utils.plots import plot_regression_diagnostics, plot_loss_curve

class Experiments:
    def __init__(self, model_type=None, seeds=None, hidden_dim=32, hyper_hidden_dim=64):
        self.model_types = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'DeepEnsembleNet'] if model_type is None else model_type
        self.seeds = seeds if seeds is not None else [random.randint(0, 1000) for _ in range(3)]
        self.hidden_dim = hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

    def run_experiments(self, x=np.linspace(start=-10,stop=10,num=500),
                            region_c=(-1,1),
                            frac_c=0.5, epochs=1000, warmup_epochs=500, beta_max=1.0, num_samples=100, analysis=True):
        # Parameters
        model_types = self.model_types
        seeds = self.seeds
        # Make dir
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        # Define interpolation and extrapolation region
        ind_interp = np.where((x >= region_c[0]) & (x <= region_c[1]))[0]
        x_t = torch.tensor(x, dtype=torch.float64).unsqueeze(-1)

        for model_type in model_types:
            for seed in seeds:
                # Set seed
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create save dir
                if analysis:
                    run_name = f"{model_type}_seed{seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                    save_dir = os.path.join("results", model_type, run_name)
                    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
                    os.makedirs(os.path.join(save_dir, "analysis"), exist_ok=True)

                # Context data
                ind_c = np.random.choice(ind_interp, size=round(len(ind_interp)*frac_c), replace=False)
                x_c = torch.tensor(x[ind_c], dtype=torch.float64).unsqueeze(-1)

                # Generate function
                f, desc = sample_function(seed=seed)

                # Generate outputs
                y_t = torch.tensor(f(x_t), dtype=torch.float64)
                y_c = torch.tensor(f(x_c), dtype=torch.float64)

                # Init model
                model = self.build_model(model_type, input_dim=1)

                # Create training class instance
                trainer = SingleTaskTrainer(model)

                # Train
                trainer.train(x=x_c, y=y_c, epochs=epochs, warmup_epochs=warmup_epochs, beta_max=beta_max)

                # Eval
                preds = trainer.evaluate(x=x_t, num_samples=num_samples)

                if analysis:

                    # Metrics
                    metric_outputs = metrics(preds, y_t, eps=1e-6)

                    # Save metrics
                    save_analysis_arrays(metric_outputs, os.path.join(save_dir, "analysis"))
                    np.savez(os.path.join(save_dir, "analysis", "loss_curve_data.npz"),
                                        losses=trainer.losses,
                                        mses=trainer.mses,
                                        kls=trainer.kls,
                                        betas=trainer.betas)


                    # Plot and save visuals
                    name = desc + ', Model: ' + model.__class__.__name__
                    plot_regression_diagnostics(preds, x_c, y_c, x_t, y_t, desc, ind_c, 
                                                metric_outputs=metric_outputs, block=False, 
                                                save_dir=os.path.join(save_dir, "plots"))
                    
                    plot_loss_curve(trainer.losses, trainer.mses, trainer.kls, trainer.betas, desc=name, 
                                    save_path=os.path.join(save_dir, "plots", "loss_curve.png"), block=False)
                    
                    # Save model
                    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

                    # Compute and save summary stats
                    mean_pred = metric_outputs[0]      # from metrics: mean
                    nll_per_sample = metric_outputs[-1]  # from metrics: per-sample NLL
                    rmse = float(np.sqrt(np.mean((mean_pred - y_t.cpu().numpy().squeeze())**2)))
                    mean_nll = float(np.mean(nll_per_sample))

                    summary = {
                        "desc": desc,
                        "model": model.__class__.__name__,
                        "seed": seed,
                        "rmse": rmse,
                        "mean_nll": mean_nll,
                        "timestamp": datetime.now().isoformat()
                    }

                    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                        json.dump(summary, f, indent=4)


    def build_model(self, model_type, input_dim=1):
        hidden_dim = self.hidden_dim
        hyper_hidden_dim = self.hyper_hidden_dim
        if model_type == 'IC_FDNet':
            return IC_FDNetwork(input_dim, hidden_dim, input_dim, hyper_hidden_dim)
        elif model_type == 'LP_FDNet':
            return LP_FDNetwork(input_dim, hidden_dim, input_dim, hyper_hidden_dim)
        elif model_type == 'HyperNet':
            return HyperNetwork(input_dim, hidden_dim, input_dim, hyper_hidden_dim)
        elif model_type == 'BayesNet':
            return BayesNetwork(input_dim, hidden_dim, input_dim)
        elif model_type == 'GaussHyperNet':
            return GaussianHyperNetwork(input_dim, hidden_dim, input_dim, hyper_hidden_dim)
        elif model_type == 'MLPNet':
            return DeterministicMLPNetwork(input_dim, hidden_dim, input_dim, dropout_rate=0.1)
        elif model_type == 'DeepEnsembleNet':
            return DeepEnsembleNetwork(
                network_class=DeterministicMLPNetwork,
                num_models=5,
                seed_list=[0, 1, 2, 3, 4],
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                dropout_rate=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    exp_class = Experiments(model_type=['DeepEnsembleNet'], seeds=[764])
    exp_class.run_experiments(epochs=10, analysis=False)
    print('END')


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