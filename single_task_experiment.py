import os
import torch
import numpy as np
from datetime import datetime
from data.toy_functions import generate_meta_task, sample_function
from models.fdnet import IC_FDNetwork, LP_FDNetwork
from models.hypernet import HyperNetwork
from models.bayesnet import BayesNetwork
from models.gausshypernet import GaussianHyperNetwork
from models.mlpnet import DeterministicMLPNetwork
from models.deepensemblenet import DeepEnsembleNetwork
from training.train_single_task_regression import train_single_task_regression
from results_saver import save_results
from utils.metrics import compute_nll, compute_rmse


class Experiments:
    def __init__(self, model_type=None, seeds=[0, 1, 2], hidden_dim=32, hyper_hidden_dim=64):
        self.model_types = ['IC_FDNet', 'LP_FDNet', 'HyperNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'DeepEnsembleNet'] if model_type is None else model_type
        self.seeds = seeds
        self.hidden_dim = hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim

    def run_experiments(self, x=np.linspace(start=-10,stop=10,num=200),
                            region_c=(-1,1), frac_c=0.5):
        x_min = min(x)
        x_max = max(x)
        ind_interp = np.where((x >= region_c[0]) & (x <= region_c[1]))[0]
        ind_extrap = np.where((x < region_c[0]) | (x > region_c[1]))[0]
        ind_t = np.arange(0,len(x))
        x_t = torch.tensor(x, dtype=torch.float32)

        model_types = self.model_types
        seeds = self.seeds
        hidden_dim = self.hidden_dim
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

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
                model = self.build_model(model_type, 1, hidden_dim)

                # Train and eval
                model, preds, _, _, _ = train_single_task_regression(
                    model=model,
                    x_c=x_c, y_c=y_c, x_t=x_t, y_t=y_t, desc=desc,
                    sample=True, seed=seed,
                    epochs=100, plots=False
                )

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
    exp_class.run_experiments()
