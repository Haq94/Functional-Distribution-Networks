import os
import time
import torch
import numpy as np
from tqdm import tqdm
from training.single_task_trainer import SingleTaskTrainer
# from utils.saver import save_experiment_outputs
from utils.metrics import metrics
# from utils.plots import single_task_regression_plots
from utils.saver import base_experiment_saver


class BaseExperiment:
    def __init__(self, model_type, seed=0, hidden_dim=32, hyper_hidden_dim=64, num_models=5, device=None):
        self.model_type = model_type
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.hyper_hidden_dim = hyper_hidden_dim
        self.num_models = num_models
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        # self.no_variance_models = {'MLPNet', 'HyperNet'}    

    def run_experiments(self,
                        data_loader_fn=None,
                        data=None,
                        epochs=1000,
                        beta_param_dict=None,
                        num_samples=100,
                        MC = 1,
                        save_dir_root="results\\base_experiment",
                        save_switch=False,
                        timestamp=None):
        """
        Runs a single experiment with the specified model type and data source.

        Returns:
            preds (torch.Tensor): Predictions from the trained model
            data (dict): Dictionary containing train/test data and metadata
            training_time (float): Time taken for training (seconds)
            metric_outputs (dict): Evaluation metrics on test set
        """
        # === Set float64 default ===
        torch.set_default_dtype(torch.float64)

        # === Set seed ===
        seed = self.seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # === Load data ===
        if data_loader_fn is None:
            x_train, y_train, x_test, y_test, metadata = data
        else:
            x_train, y_train, x_test, y_test, metadata = data_loader_fn(seed=seed)
            data = {"x_train": x_train, "y_train": y_train,
                     "x_test": x_test, "y_test": y_test,
                       "metadata": metadata}
            
        # === Store data on device ===
        x_train, y_train = x_train.double().to(self.device), y_train.double().to(self.device)
        x_test, y_test = x_test.double().to(self.device), y_test.double().to(self.device)

        # === Build model ===
        model = self.build_model(self.model_type, input_dim=x_train.shape[1])
        model.to(self.device).double()

        # === Train ===
        trainer = SingleTaskTrainer(model)
        start_time = time.time()
        trainer.train(x=x_train, y=y_train, epochs=epochs, beta_param_dict=beta_param_dict, MC=MC)
        training_time = time.time() - start_time

        # === Evaluate ===
        preds = trainer.evaluate(x=x_test, num_samples=num_samples)

        # === Metrics ===
        metric_outputs = metrics(preds, y_test)

        # === Save ===
        if save_switch:
            # Save dir
            os.makedirs(save_dir_root, exist_ok=True)
            if timestamp is None:
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
            save_dir = os.path.join(save_dir_root, self.model_type, f"seed_{seed}_{timestamp}")
            # Summary dict
            summary_dict = {
                "model_type": self.model_type,
                "seed": self.seed,
                "hidden_dim": self.hidden_dim,
                "hyper_hidden_dim": self.hyper_hidden_dim,
                "num_models": self.num_models
            }
            # Save
            base_experiment_saver(model=model,
                                trainer=trainer,
                                metric_outputs=metric_outputs,
                                summary_dict=summary_dict,
                                x_train=x_train,
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test,
                                metadata=metadata,
                                training_time=training_time,
                                save_dir=save_dir)

        return preds, data, training_time, metric_outputs, trainer.model


    # def get_capabilities(self, model_type):
    #     capabilities = {"mean", "bias"}
    #     if model_type not in self.no_variance_models:
    #         capabilities |= {"residuals", "variance", "nll"}
    #     return capabilities

    def build_model(self, model_type, input_dim):
        from models.fdnet import IC_FDNetwork, LP_FDNetwork
        from models.hypernet import HyperNetwork
        from models.bayesnet import BayesNetwork
        from models.gausshypernet import GaussianHyperNetwork
        from models.mlpnet import DeterministicMLPNetwork
        from models.deepensemblenet import DeepEnsembleNetwork

        if model_type == 'IC_FDNet':
            return IC_FDNetwork(input_dim, self.hidden_dim, input_dim, self.hyper_hidden_dim)
        elif model_type == 'LP_FDNet':
            return LP_FDNetwork(input_dim, self.hidden_dim, input_dim, self.hyper_hidden_dim)
        elif model_type == 'HyperNet':
            return HyperNetwork(input_dim, self.hidden_dim, input_dim, self.hyper_hidden_dim)
        elif model_type == 'BayesNet':
            return BayesNetwork(input_dim, self.hidden_dim, input_dim)
        elif model_type == 'GaussHyperNet':
            return GaussianHyperNetwork(input_dim, self.hidden_dim, input_dim, self.hyper_hidden_dim)
        elif model_type == 'MLPNet':
            return DeterministicMLPNetwork(input_dim, self.hidden_dim, input_dim, dropout_rate=0.1)
        elif model_type == 'DeepEnsembleNet':
            seed_list = [np.random.randint(0, 10000) for _ in range(self.num_models)]
            return DeepEnsembleNetwork(DeterministicMLPNetwork,
                                       self.num_models,
                                       seed_list,
                                       input_dim=input_dim,
                                       hidden_dim=self.hidden_dim,
                                       output_dim=input_dim,
                                       dropout_rate=0.1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")



if __name__ == "__main__":
    import random
    from utils.loader import load_toy_task_regression
    # Model type
    model_type = 'LP_FDNet' 
    # Data loader
    data_loader_fn = load_toy_task_regression
    # Seeds
    seed = random.randint(100,10000) 
    # Number of epochs
    epochs = 2
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
    save_switch = False
    # Base experiment class instance
    base_experiment = BaseExperiment(model_type=model_type, seed=seed)
    # Run experiment
    preds, data, training_time, metric_outputs, model = base_experiment.run_experiments(data_loader_fn=data_loader_fn)

    # # TAKE OUT THIS LOOP
    # for _ in range(300):
    #     # DELETE THIS
    #     seeds = [random.randint(100,10000) for _ in range(3)]
    #     # Create data
    #     input_type = "uniform_random"
    #     input_seed = random.randint(100,10000)
    #     x_min = -100
    #     x_max = 100
    #     n_interp = 1000
    #     n_extrap = 10000
    #     if input_type == "uniform_random":
    #         np.random.seed(input_seed)
    #         x_l = np.random.uniform(low=x_min,high=region_interp[0],size=round(n_extrap/2)) 
    #         x_c = np.random.uniform(low=region_interp[0],high=region_interp[1],size=n_interp)
    #         x_r = np.random.uniform(low=region_interp[1],high=x_max,size=n_extrap-round(n_extrap/2))
    #     else:
    #         x_l = np.linspace(start=x_min,stop=region_interp[0],num=round(n_extrap/2))
    #         x_c = np.linspace(start=region_interp[0],stop=region_interp[1],num=n_interp+2)
    #         x_r = np.linspace(start=region_interp[1],stop=x_max,num=n_extrap-round(n_extrap/2))
    #     x = np.sort(np.unique(np.concatenate([x_l, x_c, x_r])))
    #     # Fraction of points of data points in region used for training
    #     frac_train = 0.5
    #     # Create experiment class instance
    #     exp_class = Experiments(model_type=model_type, seeds=seeds)
    #     # Run experiment
    #     exp_class.run_experiments(x=x, region_interp=region_interp, frac_train=frac_train, epochs=epochs, beta_param_dict=beta_param_dict, analysis=analysis, save_switch=save_switch)
    #     print('END')


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