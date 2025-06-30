import os
import torch
import numpy as np
import random
from datetime import datetime

from data.toy_functions import generate_meta_task, sample_function
from models.fdnet import IC_FDNetwork, LP_FDNetwork
from models.hypernet import HyperNetwork
from models.bayesnet import BayesNetwork
from models.gausshypernet import GaussianHyperNetwork
from models.mlpnet import DeterministicMLPNetwork
from models.deepensemblenet import DeepEnsembleNetwork
from training.SingleTaskTrainer import SingleTaskTrainer
from utils.results_saver import save_results
from utils.metrics import compute_nll, compute_rmse


class Experiments:
    def __init__(self, model_type=None, seeds=None, hidden_dim=32, hyper_hidden_dim=64):
        self.model_types = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'DeepEnsembleNet'] if model_type is None else model_type
        self.seeds = seeds if seeds is not None else [random.randint(0, 1000) for _ in range(3)]
        self.hidden_dim = hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

    def run_experiments(self, x=np.linspace(start=-10,stop=10,num=500),
                            region_c=(-1,1),
                            frac_c=0.5, epochs=1000, warmup_epochs=500, beta_max=1.0, num_samples=100):
        # Parameters
        model_types = self.model_types
        seeds = self.seeds
        # Make dir
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        # Define interpolation and extrapolation region
        x_min = min(x)
        x_max = max(x)
        ind_interp = np.where((x >= region_c[0]) & (x <= region_c[1]))[0]
        ind_extrap = np.where((x < region_c[0]) | (x > region_c[1]))[0]
        ind_t = np.arange(0,len(x))
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)

        for model_type in model_types:
            for seed in seeds:
                # Set seed
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Context data
                ind_c = np.random.choice(ind_interp, size=round(len(ind_interp)*frac_c), replace=False)
                x_c = torch.tensor(x[ind_c], dtype=torch.float32)

                # Generate function
                f, desc = sample_function(seed=seed)

                # Generate outputs
                y_t = torch.tensor(f(x_t), dtype=torch.float32)
                y_c = torch.tensor(f(x_c), dtype=torch.float32)

                # Init model
                model = self.build_model(model_type, 1)

                # Create training class instance
                trainer = SingleTaskTrainer(model)

                # Train
                trainer.train(x_c=x_c, y_c=y_c, epochs=epochs, warmup_epochs=warmup_epochs, beta_max=beta_max)

                # Eval
                preds, mean, std = trainer.evaluate(x_t=x_t, num_samples=num_samples)

                # Compute metrics
                rmse = compute_rmse(mean, y_t.cpu().numpy())
                nll = compute_nll(mean, y_t.cpu().numpy(), std)

                import matplotlib.pyplot as plt
                x_c_np = x_c.cpu().numpy().squeeze()
                y_c_np = y_c.cpu().numpy().squeeze()
                x_t_np = x_t.cpu().numpy().squeeze()
                y_t_np = y_t.cpu().numpy().squeeze()
                x_c_min = x_c_np.min()
                x_c_max = x_c_np.max()

                res_acc = y_t_np.reshape(-1,1) - preds.squeeze()
                bias = res_acc.mean(1)
                mse = res_acc.var(1)

                res_prec = preds.mean(1) - preds.squeeze()

                # Residual Scatter Plots 
                _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                for ii in range(res_prec.shape[1]):
                    axs[0].scatter(x_t, res_prec[:, ii], alpha=0.05, color=np.random.rand(3))
                axs[0].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                axs[0].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                axs[0].set_title(f"Residual Precision Task: {desc}")
                axs[0].legend()
                axs[0].grid(True)

                for ii in range(res_acc.shape[1]):
                    axs[1].scatter(x_t, res_acc[:, ii], alpha=0.05, color=np.random.rand(3))
                axs[1].plot(x_t_np, bias, label='Bias', color='red')
                axs[1].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                axs[1].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                axs[1].set_title(f"Residual Accuracy Task: {desc}")
                axs[1].legend()
                axs[1].grid(True)

                plt.tight_layout()
                plt.show()

                # Mean Prediction Plot
                for ii in range(res_acc.shape[1]):
                    plt.scatter(x_t, preds[:, ii], alpha=0.05, color=np.random.rand(3))
                plt.plot(x_t_np, y_t_np, label="Ground Truth", linestyle="--")
                plt.plot(x_t_np, mean, label="Mean Prediction")
                plt.fill_between(x_t_np, mean - std, mean + std,
                                alpha=0.3, label="±1 Std Dev")
                plt.scatter(x_c_np, y_c_np, color="red", label="Context Points")
                plt.axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                plt.axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                plt.title(f"Mean Function Task: {desc}")
                plt.legend()
                plt.grid(True)
                plt.show()

                # Zoomed in Mean Prediction Plot
                for ii in range(res_acc.shape[1]):
                    plt.scatter(x_t, preds[:, ii], alpha=0.05, color=np.random.rand(3))
                plt.plot(x_t_np, y_t_np, label="Ground Truth", linestyle="--")
                plt.plot(x_t_np, mean, label="Mean Prediction")
                plt.fill_between(x_t_np, mean - std, mean + std, alpha=0.3, label="±1 Std Dev")
                plt.scatter(x_c_np, y_c_np, color="red", alpha=0.5, label="Context Points")
                plt.axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                plt.axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                plt.xlabel("x")
                plt.ylabel("y")
                plt.ylim(preds.min(), preds.max())
                plt.title(f'Mean Function Task: {desc} (Zoomed In)')
                plt.legend()
                plt.show()

                # Standard Deviation and Mean Plot
                _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                axs[0].plot(x_t_np, mean)
                axs[0].scatter(x_t_np[ind_c], mean[ind_c], label='Context Points', color='red')
                axs[0].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                axs[0].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                axs[0].set_title(f"Mean Function Task: {desc}")
                axs[0].legend()
                axs[0].grid(True)

                axs[1].plot(x_t_np, 10*np.log10(std))
                axs[1].scatter(x_t_np[ind_c], 10*np.log10(std[ind_c]),label='Context Points', color='red')
                axs[1].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                axs[1].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                axs[1].set_title(f"Standard Deviation (dB) Task: {desc}")
                axs[1].legend()
                axs[1].grid(True)

                # MSE and Bias Plot
                _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                axs[0].plot(x_t_np, bias)
                axs[0].scatter(x_t_np[ind_c], bias[ind_c], label='Context Points', color='red')
                axs[0].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                axs[0].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                axs[0].set_title(f"Bias Task: {desc}")
                axs[0].legend()
                axs[0].grid(True)
                
                # axs[1].plot(x_t_np, 20*np.log10(std))
                # axs[1].scatter(x_t_np[ind_c], 20*np.log10(std[ind_c]),label='Context Points', color='red')
                # axs[1].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                # axs[1].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                # axs[1].set_title(f"Task: {desc}: Variance (dB)")
                # axs[1].legend()
                # axs[1].grid(True)

                axs[1].plot(x_t_np, 10*np.log10(mse))
                axs[1].scatter(x_t_np[ind_c], 10*np.log10(mse[ind_c]), label='Context Points', color='red')
                axs[1].axvline(x=x_c_min, color='red', linestyle='--', linewidth=2)
                axs[1].axvline(x=x_c_max, color='red', linestyle='--', linewidth=2)
                axs[1].set_title(f"MSE (dB) Task: {desc}")
                axs[1].legend()
                axs[1].grid(True)

                plt.tight_layout()
                plt.show()

                print('stop')



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
    
# def run_all_experiments(save_switch=False):
#     model_types = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'DeepEnsembleNet']
#     seeds = [0, 1, 2, 3, 4]
#     input_dim = 10
#     hidden_dim = 32
#     results_dir = "results"
#     os.makedirs(results_dir, exist_ok=True)

#     for model_type in model_types:
#         for seed in seeds:
#             torch.manual_seed(seed)
#             np.random.seed(seed)

#             # Generate data
#             x_c, y_c, x_t, y_t, desc = generate_meta_task(n_context=input_dim, n_target=input_dim, seed=seed)

#             # Init model
#             model = build_model(model_type, input_dim, hidden_dim)

#             # Train and eval
#             preds, mean, std, y_true = train_single_task_regression(
#                 model=model,
#                 x_c=x_c, y_c=y_c, x_t=x_t, y_t=y_t, desc=desc,
#                 sample=True, seed=seed,
#                 epochs=2000, plots=False
#             )

#             # Compute metrics
#             rmse = compute_rmse(mean, y_true)
#             nll = compute_nll(mean, y_true, std)

#             # Save results
#             if save_switch:
#                 exp_id = f"{model_type}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#                 save_results(exp_id=exp_id, model_type=model_type, seed=seed,
#                             mean=mean, std=std, y_true=y_true, nll=nll, rmse=rmse,
#                             results_dir=results_dir)

#             print(f"[{model_type} | seed {seed}] RMSE: {rmse:.4f} | NLL: {nll:.4f}")


if __name__ == "__main__":
    exp_class = Experiments()
    exp_class.run_experiments(epochs=100)
