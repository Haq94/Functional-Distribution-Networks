import os
import time
import torch
import numpy as np
from tqdm import tqdm

from utils.metrics import metrics
from utils.general import set_determinism
from utils.saver.base_experiment_saver import base_experiment_saver
from training.single_task_trainer import SingleTaskTrainer


class BaseExperiment:
    def __init__(self, **kwargs):
        # for key, val in kwargs.items():   
        #     self.model_type = kwargs.get('model_type', None)
        #     self.seed =  kwargs.get('seed', 0)
        #     self.hidden_dim =  kwargs.get('hidden_dim', 32)
        #     self.hyper_hidden_dim =  kwargs.get('hyper_hidden_dim', 64)
        #     self.latent_dim =  kwargs.get('latent_dim', 10)
        #     self.num_models =  kwargs.get('num_model', 5)
        #     self.ensemble_seed_list = kwargs.get('ensemble_seed_list', [np.random.randint(0, 10000) for _ in range(num_models)])
        #     self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        #     self.MC_val = kwargs.get('MC_val', 100)
        #     self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #     self.save_path = kwargs.get('save_path', "results\\base_experiment")

        # Define the list of allowed fields
        allowed_fields = {
            'model_type',
            'seed',
            'hidden_dim',
            'hyper_hidden_dim',
            'latent_dim',
            'num_models',
            'ensemble_seed_list',
            'dropout_rate',
            'device',
            'save_path'
        }

        # Populate only fields present in kwargs
        for key in allowed_fields:
            if key in kwargs:
                setattr(self, key, kwargs[key])

        # Optional: handle dependent defaults if needed
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if hasattr(self, 'num_model') and not hasattr(self, 'ensemble_seed_list'):
            self.ensemble_seed_list = [np.random.randint(0, 10000) for _ in range(self.num_model)]

        kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        stoch_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'DeepEnsembleNet', 'MLPDropoutNet'}
        mc_model = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'MLPDropoutNet'}
        self.is_stoch = self.model_type in stoch_models
        self.kl_exist = self.model_type in kl_models
        self.training_type = 'MC' if self.model_type in mc_model else 'Ensemble' if self.model_type == 'DeepEnsembleNet' else 'Deterministic'  

    def run_experiments(self,
                        data_loader_fn=None,
                        data=None,
                        epochs=1000,
                        beta_param_dict=None, val_data=None,
                        MC_train=1,
                        MC_val=100,
                        MC_test=100,
                        checkpoint_dict=None,
                        save_switch=False,
                        train_val_metrics=True,
                        info=None):
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

        # === Save path ===
        model_type = self.model_type
        # title = f"seed_{seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}" if title == None else title
        save_path = os.path.join(self.save_path, model_type) if save_switch else None

        # === Set seed ===
        seed = self.seed
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        set_determinism(seed=seed)

        # === Load data ===
        if data_loader_fn is None:
            x_train, y_train, x_val, y_val, x_test, y_test, desc = data
            metadata = {'description': desc}
        else:
            x_train, y_train, x_val, y_val, x_test, y_test, metadata = data_loader_fn(seed=seed)
        data = {
                    "x_train": x_train, "y_train": y_train,
                    "x_test": x_test  , "y_test" : y_test,
                    "x_val" : x_val   , "y_val"  : y_val,
                    "metadata": metadata
                }
        val_data = (x_val, y_val, MC_val)
            
        # === Store data on device ===
        x_train = x_train.double().to(self.device) if isinstance(x_train, torch.Tensor) else torch.tensor(x_train, dtype=torch.double, device=self.device)
        y_train = y_train.double().to(self.device) if isinstance(y_train, torch.Tensor) else torch.tensor(y_train, dtype=torch.double, device=self.device)

        x_val = x_val.double().to(self.device) if isinstance(x_val, torch.Tensor) else torch.tensor(x_val, dtype=torch.double, device=self.device)
        y_val = y_val.double().to(self.device) if isinstance(y_val, torch.Tensor) else torch.tensor(y_val, dtype=torch.double, device=self.device)
        
        x_test = x_test.double().to(self.device) if isinstance(x_test, torch.Tensor) else torch.tensor(x_test, dtype=torch.double, device=self.device)
        y_test = y_test.double().to(self.device) if isinstance(y_test, torch.Tensor) else torch.tensor(y_test, dtype=torch.double, device=self.device)


        # === Build model ===
        model = self.build_model(self.model_type, input_dim=x_train.shape[1], output_dim=y_train.shape[1])
        model.to(self.device).double()

        # === Train ===
        checkpoint_path = None if save_path is None else os.path.join(save_path, 'checkpoint')
        trainer = SingleTaskTrainer(model, checkpoint_path=checkpoint_path)
        start_time = time.time()
        trainer.train(x_train=x_train, y_train=y_train, epochs=epochs, 
                      beta_param_dict=beta_param_dict, val_data=val_data, MC=MC_train)
        training_time = time.time() - start_time

        # === Choose best model ===
        if checkpoint_dict != None:
            self.choose_best_model(trainer, x_val, checkpoint_dict)

        # === Evaluate ===
        preds_test = trainer.evaluate(x=x_test, MC=MC_test)

        # === Metrics ===
        metrics_test = metrics(preds_test, y_test)

        # === Evaluate train/ val metrics ===
        if train_val_metrics:
            preds_train = trainer.evaluate(x=x_train, MC=MC_test)
            preds_val = trainer.evaluate(x=x_val, MC=MC_test)

            metrics_train = metrics(preds_train, y_train)
            metrics_val = metrics(preds_val, y_val)
        else:
            metrics_train = None
            metrics_val = None

        # === Save ===
        if save_switch:
            # Save dir
            os.makedirs(save_path, exist_ok=True)
            # Summary dict
            summary_dict = self._gen_summary_dict(epochs=epochs, data=data, beta_param_dict=beta_param_dict,
                                                  training_time=training_time, trainer=trainer, 
                                                  metric_outputs=metrics_test, MC_train=MC_train, 
                                                  MC_val=MC_val, MC_test=MC_test)
            # Save
            base_experiment_saver(model=model,
                                trainer=trainer,
                                metrics_train=metrics_train,
                                metrics_val=metrics_val,
                                metrics_test=metrics_test,
                                summary_dict=summary_dict,
                                x_train=x_train,
                                y_train=y_train,
                                x_val=x_val,
                                y_val=y_val,
                                x_test=x_test,
                                y_test=y_test,
                                save_path=save_path,
                                info=info)

        if train_val_metrics:
            preds_dict = {'train': preds_train, 'val': preds_val, 'test': preds_test}
            metrics_dict = {'train': metrics_train, 'val': metrics_val, 'test': metrics_test}
            return preds_dict, data, training_time, metrics_dict, trainer
        else:
            return preds_test, data, training_time, metrics_test, trainer
    
    def choose_best_model(self, trainer, x_val, checkpoint_dict):
        if len(checkpoint_dict) == 0:
            return 
        
        # --- Make sure x_val is a NumPy array on CPU ---
        if isinstance(x_val, torch.Tensor):
            x_val_np = x_val.detach().cpu().numpy().squeeze()
        else:
            x_val_np = np.asarray(x_val).squeeze()

        # Checkpoint parameters
        metric_str = checkpoint_dict['metric_str']
        region_interp = checkpoint_dict['region_interp']
        min_or_max = checkpoint_dict['min_or_max']
        interp_or_extrap = checkpoint_dict['interp_or_extrap']

        # Interp and extrap indices
        ind_interp = np.where((x_val_np >= region_interp[0]) & (x_val_np <= region_interp[1]))[0]
        ind_extrap = np.where((x_val_np < region_interp[0]) | (x_val_np > region_interp[1]))[0]

        if interp_or_extrap == 'interp' and ind_interp.shape[0] == 0:
            print('No interpolation points in validation data set')
            return
        elif interp_or_extrap == 'extrap' and ind_extrap.shape[0] == None:
            print('No extrapolation points in validation data set')
            return

        # If model is deterministic then use MSE
        if trainer.training_type == 'Deterministic':
            metric_str = 'mse'
        metric_str += '_val'

        # Load metrics
        best_metric = getattr(trainer, metric_str, None)

        # Metric at interpolation points
        best_metric_interp = best_metric[:, ind_interp]
        best_metric_interp = best_metric_interp.mean(1) if best_metric_interp is not None else None

        # Metric at extrapolation points
        best_metric_extrap = best_metric[:, ind_extrap]
        best_metric_extrap= best_metric_extrap.mean(1) if best_metric_extrap is not None else None
        
        # Get best epoch according to metric
        if interp_or_extrap == 'interp':
            best_epoch = np.argmax(best_metric_interp) + 1 if min_or_max == 'max' else np.argmin(best_metric_interp) + 1
        else:
            best_epoch = np.argmax(best_metric_extrap) + 1 if min_or_max == 'max' else np.argmin(best_metric_extrap) + 1

        # Load model
        trainer.load_epoch_from_log(epoch=int(best_epoch))

    def _gen_summary_dict(self, epochs, data, beta_param_dict, training_time, trainer, metric_outputs, MC_train, MC_val, MC_test):
        
        num_params=trainer.num_params
        num_updates=trainer.num_updates
        num_param_updates=trainer.num_param_updates
        num_forward_passes=trainer.num_forward_passes

        summary_dict = {
                "model_type": self.model_type,
                "desc": data['metadata']['description'],
                "seed": self.seed,
                "epochs": epochs,
                "model_epoch": trainer.model_epoch,
                "training_time": training_time,
                "num_train": data['x_train'].shape[0],
                "num_val": data['x_val'].shape[0],
                "num_test": data['x_test'].shape[0],
                "MC_train": MC_train,
                "MC_val": MC_val,
                "MC_test": MC_test,
                "hidden_dim": self.hidden_dim,
                "num_params": num_params,
                "num_updates": num_updates,
                "num_param_updates": num_param_updates,
                "num_forward_passes": num_forward_passes
            }
        
        if beta_param_dict is not None:
            summary_dict['beta_scheduler'] = beta_param_dict['beta_scheduler']
            summary_dict['beta_max'] = beta_param_dict['beta_max']
            if beta_param_dict['beta_scheduler'] != 'constant':
                summary_dict['warmup_epochs'] = beta_param_dict['warmup_epochs']

        if self.model_type == 'DeepEnsembleNet':
            summary_dict['num_models'] = self.num_models
            summary_dict['ensemble_seed_list'] = self.ensemble_seed_list
            summary_dict['dropout_rate'] = self.dropout_rate
        elif self.model_type in {'MLPNet', 'MLPDropoutNet'}:
            summary_dict['dropout_rate'] = self.dropout_rate
        elif self.model_type in {'LP_FDNet', 'IC_FDNet', 'HyperNet'}:
            summary_dict['hyper_hidden_dim'] = self.hyper_hidden_dim
        elif self.model_type == 'GaussHyperNet':
            summary_dict['hyper_hidden_dim'] = self.hyper_hidden_dim
            summary_dict['latent_dim'] = self.latent_dim
        
        def store_metric(metric_str):
            metric = getattr(trainer, metric_str, None)
            if metric is not None:
                summary_dict[metric_str] = metric.mean()
        
        # Store training metrics
        summary_dict['final_loss'] = trainer.losses[-1]
        summary_dict['final_mse'] = trainer.mses[-1]
        summary_dict['final_kl_div'] = trainer.kls[-1]
        # Store average validation metrics
        store_metric('mse_val')
        store_metric('kl_val')
        store_metric('bias_sq_val')
        store_metric('var_val')
        store_metric('crps_val')
        store_metric('nlpd_val')
        # Store average test metrics
        summary_dict['mse_test'] = metric_outputs['mse'].mean()
        if len(trainer.val_metrics_set) > 1:
            summary_dict['bias_sq_test'] = (metric_outputs['bias']**2).mean()
            summary_dict['crps_test'] = metric_outputs['crps'].mean()
            summary_dict['nlpd_test'] = metric_outputs['nlpd_kde'].mean()

        return summary_dict

    def build_model(self, model_type, input_dim, output_dim):
        hidden_dim = getattr(self, 'hidden_dim', 32)
        hyper_hidden_dim = getattr(self, 'hyper_hidden_dim', 10)
        num_models = getattr(self, 'num_models', 5)
        dropout_rate = getattr(self, 'dropout_rate', 0.1)
        if model_type == 'IC_FDNet':
            from models.fdnet import IC_FDNet
            return IC_FDNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hyper_hidden_dim=hyper_hidden_dim)
        elif model_type == 'LP_FDNet':
            from models.fdnet import LP_FDNet
            return LP_FDNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hyper_hidden_dim=hyper_hidden_dim)
        elif model_type == 'HyperNet':
            from models.hypernet import HyperNet
            return HyperNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hyper_hidden_dim=hyper_hidden_dim)
        elif model_type == 'BayesNet':
            from models.bayesnet import BayesNet
            return BayesNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        elif model_type == 'GaussHyperNet':
            from models.gausshypernet import GaussHyperNet
            latent_dim = getattr(self, 'latent_dim', 10)
            return GaussHyperNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, hyper_hidden_dim=hyper_hidden_dim, latent_dim=latent_dim)
        elif model_type == 'MLPNet':
            from models.mlpnet import MLPNet
            return MLPNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate)
        elif model_type == 'MLPDropoutNet':
            from models.mlpdropoutnet import MLPDropoutNet
            return MLPDropoutNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate)
        elif model_type == 'DeepEnsembleNet':
            from models.mlpnet import MLPNet
            from models.deepensemblenet import DeepEnsembleNet
            seed_list = [np.random.randint(0, 10000) for _ in range(self.num_models)]
            return DeepEnsembleNet(MLPNet,
                                       num_models=num_models,
                                       seed_list=seed_list,
                                       input_dim=input_dim,
                                       hidden_dim=hidden_dim,
                                       output_dim=output_dim,
                                       dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unknown model type: {model_type}")



if __name__ == "__main__":
    import random
    from utils.loader.general_loader import load_toy_task_regression
    # Run title
    title='base_experiment'
    # Save path
    save_path = os.path.join("results", title)
    # Model parameters
    model_type = 'LP_FDNet'           
    seed = random.randint(100,10000) 
    hidden_dim = 32
    hyper_hidden_dim = 11
    latent_dim = 10
    dropout_rate = 0.1
    num_models = 10
    ensemble_seed_list = [np.random.randint(0, 10000) for _ in range(num_models)]
    # Data loader
    data_loader_fn = load_toy_task_regression 
    # Training parameters
    epochs = 6
    MC_train = 1
    # Validation parameters
    MC_val = 100
    checkpoint_dict = {
    'metric_str': 'mse',
    'region_interp': (-1.5, 0),
    'min_or_max': 'max',
    'interp_or_extrap': 'interp'
    }
    # Evaluation parameters
    MC_test = 51
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
    if model_type == 'IC_FDNet':
        base_experiment = BaseExperiment(model_type=model_type, seed=seed, hidden_dim=hidden_dim, hyper_hidden_dim=hyper_hidden_dim, save_path=save_path)
    elif model_type == 'LP_FDNet':
        base_experiment = BaseExperiment(model_type=model_type, seed=seed, hidden_dim=hidden_dim, hyper_hidden_dim=hyper_hidden_dim, save_path=save_path)
    elif model_type == 'HyperNet':
        base_experiment = BaseExperiment(model_type=model_type, seed=seed, hidden_dim=hidden_dim, hyper_hidden_dim=hyper_hidden_dim, save_path=save_path)
    elif model_type == 'BayesNet':
        base_experiment = BaseExperiment(model_type=model_type, seed=seed, hidden_dim=hidden_dim, save_path=save_path)
    elif model_type == 'GaussHyperNet':
        base_experiment = BaseExperiment(model_type=model_type, seed=seed, hidden_dim=hidden_dim, hyper_hidden_dim=hyper_hidden_dim, latent_dim=latent_dim, save_path=save_path)
    elif model_type == 'MLPNet':
        base_experiment = BaseExperiment(model_type=model_type, seed=seed, hidden_dim=hidden_dim, dropout_rate=dropout_rate, save_path=save_path)
    elif model_type == 'MLPDropoutNet':
        base_experiment = BaseExperiment(model_type=model_type, seed=seed, hidden_dim=hidden_dim, dropout_rate=dropout_rate, save_path=save_path)
    elif model_type == 'DeepEnsembleNet':
        base_experiment = BaseExperiment(model_type=model_type, seed=seed, num_models=num_models, hidden_dim=hidden_dim, dropout_rate=dropout_rate, ensemble_seed_list=ensemble_seed_list, save_path=save_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run experiment
    preds, data, training_time, metric_outputs, model = base_experiment.run_experiments(data_loader_fn=data_loader_fn,
                                                                                         epochs=epochs, 
                                                                                         beta_param_dict=beta_param_dict, 
                                                                                         MC_train=MC_train,
                                                                                         MC_val=MC_val, 
                                                                                         MC_test=MC_test, 
                                                                                         checkpoint_dict=checkpoint_dict,
                                                                                         save_switch=save_switch)

    print("Base Experiment Completed")