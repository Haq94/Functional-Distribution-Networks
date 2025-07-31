import os
import time
import torch
import numpy as np
from tqdm import tqdm
from training.single_task_trainer import SingleTaskTrainer
from utils.metrics import metrics
from utils.saver.base_experiment_saver import base_experiment_saver


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

        return preds, data, training_time, metric_outputs, trainer

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
    from utils.loader.general_loader import load_toy_task_regression
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
    save_switch = True
    # Base experiment class instance
    base_experiment = BaseExperiment(model_type=model_type, seed=seed)
    # Run experiment
    preds, data, training_time, metric_outputs, model = base_experiment.run_experiments(data_loader_fn=data_loader_fn, save_switch=save_switch)

    print("Base Experiment Completed")