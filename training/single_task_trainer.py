import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import numpy as np
import random

from utils.general import set_determinism, count_parameters
from utils.metrics import per_x_nlpd_from_samples_kde, energy_score_from_samples
# from utils.plots import plot_meta_task, plot_loss_curve
# from utils.debug_tools import debug_requires_grad, get_param_and_grad_dict

class SingleTaskTrainer:
    def __init__(self, model, optimizer=None, lr=1e-3, device=None, checkpoint_path=None):
        self.model = model.double()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.checkpoint_path = checkpoint_path
        if checkpoint_path != None:
            self.path_log = {}

        kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        stoch_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'DeepEnsembleNet', 'MLPDropoutNet'}
        mc_model = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet', 'MLPDropoutNet'}
        self.is_stoch = model.__class__.__name__ in stoch_models
        self.kl_exist = model.__class__.__name__ in kl_models
        self.training_type = 'MC' if model.__class__.__name__ in mc_model else 'Ensemble' if model.__class__.__name__ == 'DeepEnsembleNet' else 'Deterministic'

        self.losses = []
        self.mses = []
        self.kls = []
        self.betas = []

        # Handle optimizer(s)
        if model.__class__.__name__ == 'DeepEnsembleNet':
            self.optimizers = [
                torch.optim.Adam(submodel.parameters(), lr=lr)
                for submodel in model.models
            ]
            self.optimizer = None  # Not used for ensembles
        else:
            self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr)

    def _model_forward(self, x, return_kl=True, sample=True):
        if self.kl_exist:
            return self.model(x, return_kl=return_kl, sample=sample)
        else:
            return self.model(x), torch.tensor(0.0, device=x.device)

    def train(self, x_train, y_train, epochs=1000, beta_param_dict=None, val_data=None,
            print_every=100, batch_size=10, grad_clip=True, MC=1):

        # Check if validation data exist
        self.val_exist = True if  (val_data[0] is not None and val_data[1] is not None) else False
        if self.val_exist and self.is_stoch:
            self.x_val = val_data[0]; self.y_val = val_data[1]; self.MC_val = val_data[2]
            self.val_metrics_set = {'crps', 'nlpd', 'var', 'mse', 'bias_sq'}
            self.mse_val = []; self.bias_sq_val = []; self.crps_val = []; self.nlpd_val = []; self.var_val = []
        elif self.val_exist:
            self.x_val = val_data[0]; self.y_val = val_data[1]; self.MC_val = val_data[2]
            self.val_metrics_set = {'mse'}
            self.mse_val = []

        # Calculate link budget
        num_models = sum([1 for _ in self.model.models]) if self.training_type == 'Ensemble' else 1
        self._link_budget(num_train_points=x_train.shape[0], epochs=epochs, batch_size=batch_size, MC=MC, num_models=num_models)

        x_train, y_train = x_train.double().to(self.device), y_train.double().to(self.device)
        N_train = x_train.shape[0]
        if N_train == 0:
            raise ValueError("Empty context set provided.")
        
        # If beta parameter dictionary is None then default to linear
        if beta_param_dict is None:
            beta_scheduler = "linear"
            # Warm up epochs
            warmup_epochs = round(epochs/2)
            # Beta max
            beta_max = 1.0
            # Beta parameter dictionary
            beta_param_dict = {"beta_scheduler": beta_scheduler,
                            "warmup_epochs": warmup_epochs, "beta_max": beta_max}

        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_kl = 0.0
            total_mse = 0.0
            beta = self._compute_beta(epoch, beta_param_dict)

            for x_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch}"):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # ==== Monte Carlo stochastic models ====
                if self.training_type == 'MC':
                    mse_total, kl_total = None, None
                    for _ in range(MC):
                        y_pred, kl = self._model_forward(x_batch)
                        mse = F.mse_loss(y_pred, y_batch)
                        mse_total = mse if mse_total is None else mse_total + mse
                        kl_total = kl if kl_total is None else kl_total + kl
                    mse = mse_total / MC
                    kl = kl_total / MC
                    loss = mse + beta * kl

                    self.optimizer.zero_grad()
                    loss.backward()
                    if grad_clip:
                        clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                # ==== Deep Ensemble ====
                elif self.training_type == 'Ensemble':
                    mse = 0.0
                    loss = 0.0
                    kl = torch.tensor(0.0, device=self.device)

                    for i, (submodel, subopt) in enumerate(zip(self.model.models, self.optimizers)):
                        subopt.zero_grad()
                        y_pred_i = submodel(x_batch)
                        sub_loss = F.mse_loss(y_pred_i, y_batch)
                        sub_loss.backward()
                        if grad_clip:
                            clip_grad_norm_(submodel.parameters(), 1.0)
                        subopt.step()

                        loss += sub_loss.item() 
                        mse += sub_loss.item()

                    loss /= len(self.model.models)
                    mse /= len(self.model.models)

                    loss = torch.tensor(loss, device=self.device)
                    mse  = torch.tensor(mse, device=self.device)

                # ==== Deterministic model ====
                else:
                    y_pred, kl = self._model_forward(x_batch)
                    mse = F.mse_loss(y_pred, y_batch)
                    loss = mse + beta * kl

                    self.optimizer.zero_grad()
                    loss.backward()
                    if grad_clip:
                        clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                # Accumulate stats
                total_loss += loss.item() * x_batch.shape[0]
                total_mse  += mse.item() * x_batch.shape[0]
                total_kl   += kl.item() * x_batch.shape[0]

            total_loss /= N_train
            total_mse  /= N_train
            total_kl   /= N_train

            self.losses.append(total_loss)
            self.mses.append(total_mse)
            self.kls.append(total_kl)
            self.betas.append(beta)

            # Save checkpoints
            if self.checkpoint_path != None:
                self._save_checkpoint(epoch)
            
            # Calculate validation metrics
            self.val_exist and self._val_metrics()

            if epoch % print_every == 0:
                print(f"[Epoch {epoch}] Loss: {total_loss:.4f} | MSE: {total_mse:.4f} | KL: {total_kl:.4f} | β: {beta:.2f}")            
        
        self.model_epoch = epoch
        if self.val_exist:
            for metric in self.val_metrics_set:
                if metric == 'mse':
                    self.mse_val = np.stack(self.mse_val)
                elif metric == 'bias_sq':
                    self.bias_sq_val = np.stack(self.bias_sq_val)
                elif metric == 'var':
                    self.var_val = np.stack(self.var_val)
                elif metric == 'nlpd':
                    self.nlpd_val = np.stack(self.nlpd_val)
                elif metric == 'crps':
                    self.crps_val = np.stack(self.crps_val)
                    
    def _val_metrics(self):
        x_val = self.x_val
        y_val = self.y_val.cpu().numpy().squeeze().astype(np.float64) # (batch_size,)
        MC_val = self.MC_val
        val_metrics_set = self.val_metrics_set

        if self.kl_exist:
            # preds shape: (num_samples, batch_size, output_dim)
            preds, kl = self.evaluate(x_val, MC=MC_val, return_kl=True)
            self.kl_val = kl
        else:
            preds = self.evaluate(x_val, MC=MC_val).astype(np.float64) # shape: (num_samples, batch_size, output_dim)

        for metric in val_metrics_set:
            metric == 'mse' and self.mse_val.append((((preds.squeeze().T - y_val.reshape(-1, 1))** 2).mean(1).reshape(-1, 1)).squeeze())
            metric == 'bias_sq' and self.bias_sq_val.append(((preds.mean(0) - y_val.reshape(-1, 1))**2).squeeze())
            metric == 'var' and self.var_val.append((preds.var(0)).squeeze())
            metric == 'nlpd' and self.nlpd_val.append(per_x_nlpd_from_samples_kde(preds=preds, y_true=y_val).squeeze())
            metric == 'crps' and self.crps_val.append(energy_score_from_samples(preds=preds, y_true=y_val).squeeze())

    def _save_checkpoint(self, epoch):
        # where to write checkpoints
        checkpoint_path = getattr(self, "checkpoint_path", "checkpoints")         # you can set self.exp_dir externally
        model_type = self.model.__class__.__name__
        os.makedirs(checkpoint_path, exist_ok=True)

        # handle ensembles vs single models
        if model_type == "DeepEnsembleNet":
            state_dict = [m.state_dict() for m in self.model.models]
            # opt_state  = [opt.state_dict() for opt in self.optimizers]
        else:
            state_dict = self.model.state_dict()
            # opt_state  = self.optimizer.state_dict() if self.optimizer is not None else None

        payload = {
            "epoch": epoch,
            "model_class": model_type,
            "state_dict": state_dict,
            # "optimizer_state": opt_state
        }

        # Save payload
        path = os.path.join(checkpoint_path, f"epoch_{epoch:04d}.pt")
        torch.save(payload, path)

        # Log path for loading
        self.path_log[epoch] = path

    def evaluate(self, x, MC=30, sample=True, return_kl=False):
        self.model.eval()
        x = x.double().to(self.device) if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float64, device=self.device).view(-1, 1)
        N = x.shape[0]
        preds = []

        with torch.no_grad():
            if self.training_type == 'MC' and self.kl_exist:
                if return_kl:
                    y_pred = []; kldiv = []
                    for _ in range(MC):
                        pred, kl = self._model_forward(x=x, return_kl=return_kl, sample=sample)
                        y_pred.append(pred.squeeze(0).detach().cpu().numpy()); kldiv.append(kl.detach().cpu().numpy())
                    kldiv = np.array(kldiv)
                else:
                    y_pred = [self._model_forward(x=x, return_kl=return_kl, sample=sample).squeeze(0).detach().cpu().numpy() for _ in range(MC)]
            elif self.training_type == 'Ensemble':
                y_pred = self._model_forward(x=x)[0].squeeze(0).detach().cpu().numpy()
            elif self.training_type == 'MC' and not self.kl_exist:
                y_pred = [self._model_forward(x=x, sample=sample)[0].squeeze(0).detach().cpu().numpy() for _ in range(MC)]
            else:
                y_pred = self._model_forward(x=x)[0].squeeze(0).detach().cpu().numpy()
                y_pred = [y_pred for _ in range(MC)]

        preds = np.stack(y_pred)  # shape: (num_samples, batch_size, output_dim)

        return (preds, kldiv) if (self.kl_exist and return_kl) else preds
        
    def _compute_beta(self, epoch, beta_param_dict):
        """
        Compute the beta value at a given epoch based on the beta scheduling strategy.

        Args:
            epoch (int): Current epoch
            beta_param_dict (dict): Dictionary containing:
                - beta_scheduler: str, one of ['zero', 'constant', 'linear', 'cosine', 'sigmoid']
                - beta_max: float
                - warmup_epochs: int

        Returns:
            float: Beta value for this epoch
        """
        beta_scheduler = beta_param_dict["beta_scheduler"]
        beta_max = beta_param_dict["beta_max"]

        if beta_scheduler == "constant":
            return beta_max
        
        warmup_epochs = beta_param_dict["warmup_epochs"]
        progress = min(epoch / warmup_epochs, 1.0)

        if beta_scheduler == "linear":
            return beta_max * progress

        elif beta_scheduler == "cosine":
            return beta_max * (1 - np.cos(np.pi * progress)) / 2

        elif beta_scheduler == "sigmoid":
            slope = 12  # Adjust for steepness
            midpoint = 0.5
            sigmoid_val = 1 / (1 + np.exp(-slope * (progress - midpoint)))
            sigmoid_norm = sigmoid_val / (1 / (1 + np.exp(-slope * (1 - midpoint))))  # normalize to hit beta_max
            return beta_max * sigmoid_norm

        else:
            raise ValueError(f"Unsupported beta scheduler: {beta_scheduler}")

    def load_epoch_from_log(self, epoch, map_location=None, strict=True, eval_mode=True):
        """
        Reload the model weights for a specific epoch using the saved path in the log dict.
        Returns the loaded checkpoint dict.
        """

        # accept int 12 or string "epoch_12"
        key_str = f"epoch_{int(epoch)}"
        path_log = self.path_log
        if not path_log:
            raise ValueError("No path log found (expected self.path_log or self.log_path).")

        # try exact matches in order of preference
        path = None
        if key_str in path_log:
            path = path_log[key_str]
        elif isinstance(epoch, int) and epoch in path_log:
            path = path_log[epoch]
        else:
            # some users store with zero-padded keys
            key_pad = f"epoch_{int(epoch):04d}"
            if key_pad in path_log:
                path = path_log[epoch]

        if path is None:
            raise KeyError(f"No checkpoint path recorded for epoch {epoch}.")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Recorded checkpoint path does not exist: {path}")

        # load checkpoint
        chkpt = torch.load(path, map_location=map_location or self.device)
        state = chkpt.get("state_dict", None)
        if state is None:
            raise KeyError(f"'state_dict' missing in checkpoint: {path}")

        # handle ensemble vs single model (mirrors your _save_checkpoint)
        model_name = self.model.__class__.__name__
        if model_name == "DeepEnsembleNet":
            if not isinstance(state, (list, tuple)) or len(state) != len(self.model.models):
                raise ValueError("Ensemble state_dict format/size mismatch.")
            for subm, sd in zip(self.model.models, state):
                subm.load_state_dict(sd, strict=strict)
        else:
            self.model.load_state_dict(state, strict=strict)

        self.model.to(self.device)
        if eval_mode:
            self.model.eval()

        # keep a note
        self.model_epoch = chkpt.get("epoch", int(epoch))
    
    def _link_budget(self, num_train_points, epochs, batch_size, MC, num_models=1):
        """
        Estimates training compute budget:
        - param_count: total trainable parameters
        - num_updates: optimizer steps across all models
        - num_forward_passes: total forward passes incl. MC sampling
        """
        # Number of trainable parameters
        self.num_params = count_parameters(self.model)
        # Number of updates
        self.num_updates = int(epochs*np.ceil(num_train_points / batch_size)*num_models)
        # Number of parameter updates
        self.num_param_updates = self.num_params*self.num_updates
        # Number of forward passes
        self.num_forward_passes = int(self.num_updates*MC)

        print(f'Number of Parameter: {self.num_params}')
        print(f'Number of Updates: {self.num_updates}')
        print(f'Number of Parameter Updates: {self.num_param_updates}')
        print(f'Number of Forward Passes: {self.num_forward_passes}')

if __name__=='__main__':

    from models.fdnet import LP_FDNet, IC_FDNet
    from models.hypernet import HyperNet
    from models.bayesnet import BayesNet
    from models.gausshypernet import GaussHyperNet
    from models.mlpnet import MLPNet
    from models.mlpdropoutnet import MLPDropoutNet
    from models.deepensemblenet import DeepEnsembleNet
    from data.toy_functions import generate_meta_task
    
    # Parameters
    input_dim = 1               
    hidden_dim = 32
    hyper_hidden_dim = 64
    latent_dim = 10
    epochs = 5
    print_every = 20
    sample = True
    seed = 10
    model_type = 'MLPDropoutNet'
    MC_val = 100

    if seed:
        set_determinism(seed=seed)
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)

    # Train and test data
    x_train, y_train, x_val, y_val, x_test, y_test, desc = generate_meta_task(n_train=20, n_val=7, n_test=100, seed=seed)
    
    # Create model
    if model_type == 'LP_FDNet':
        model = LP_FDNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, hyper_hidden_dim=hyper_hidden_dim)
    elif model_type == 'IC_FDNet':
        model = IC_FDNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, hyper_hidden_dim=hyper_hidden_dim)
    elif model_type == 'HyperNet':
        model = HyperNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, hyper_hidden_dim=hyper_hidden_dim)
    elif model_type == 'BayesNet':
        model = BayesNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, prior_std=1.0)
    elif model_type == 'GaussHyperNet':
        model = GaussHyperNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, hyper_hidden_dim=hyper_hidden_dim, latent_dim=latent_dim, prior_std=1.0)
    elif model_type == 'MLPNet':
        model = MLPNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, dropout_rate=0.1)
    elif model_type == 'MLPDropoutNet':
        model = MLPDropoutNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, dropout_rate=0.1)
    elif model_type == 'DeepEnsembleNet':
        model = DeepEnsembleNet(
            network_class=MLPNet,
            num_models=5,
            seed_list=[0, 1, 2, 3, 4],
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            dropout_rate=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print('Running', type(model).__name__, '======================================================================')

    # Create training class instance
    trainer = SingleTaskTrainer(model, checkpoint_path='results\\trainer_test')
    # Train
    trainer.train(x_train=x_train, y_train=y_train, epochs=epochs, val_data=(x_val, y_val, MC_val))
    # Evaluate
    preds = trainer.evaluate(x=x_test)
    # Load an epoch
    epoch_load = epochs//2    
    trainer.load_epoch_from_log(epoch=epoch_load)
    # Evaluate loaded model
    preds_load = trainer.evaluate(x=x_test)

    print('Single Task Trainer Regression Test Complete')

