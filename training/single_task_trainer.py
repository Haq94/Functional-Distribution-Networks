import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import numpy as np
import random
# from utils.plots import plot_meta_task, plot_loss_curve
# from utils.debug_tools import debug_requires_grad, get_param_and_grad_dict

class SingleTaskTrainer:
    def __init__(self, model, optimizer=None, lr=1e-3, device=None):
        self.model = model.double()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
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
        if hasattr(self.model, 'forward') and ('return_kl' and 'sample' in self.model.forward.__code__.co_varnames):
            return self.model(x, return_kl=return_kl, sample=sample)
        else:
            return self.model(x), torch.tensor(0.0, device=x.device)

    def train(self, x_train, y_train, epochs=1000, beta_param_dict=None, x_val=None, y_val=None,
            print_every=100, batch_size=10, grad_clip=True, MC=1):

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
                if self.model.__class__.__name__ in self.kl_models:
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
                elif self.model.__class__.__name__ == 'DeepEnsembleNet':
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

            if epoch % print_every == 0:
                print(f"[Epoch {epoch}] Loss: {total_loss:.4f} | MSE: {total_mse:.4f} | KL: {total_kl:.4f} | β: {beta:.2f}")            

    def evaluate(self, x, num_samples=30, sample=True):
        self.model.eval()
        x = x.double().to(self.device)
        N = x.shape[0]
        preds = []

        with torch.no_grad():
            if hasattr(self.model, 'forward') and ('return_kl' and 'sample' in self.model.forward.__code__.co_varnames):
                y_pred = [self._model_forward(x=x, return_kl=False, sample=sample).squeeze(0).detach().cpu().numpy() for _ in range(num_samples)]
            elif self.model.__class__.__name__ == 'DeepEnsembleNet':
                y_pred = self._model_forward(x=x)[0].squeeze(0).detach().cpu().numpy()
            else:
                y_pred = self._model_forward(x=x)[0].squeeze(0).detach().cpu().numpy()
                y_pred = [y_pred for _ in range(num_samples)]

        preds = np.stack(y_pred)  # shape: (num_samples, batch_size, output_dim)

        return preds
        
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
        warmup_epochs = beta_param_dict["warmup_epochs"]

        if beta_scheduler == "constant":
            return beta_max

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



if __name__=='__main__':

    from models.fdnet import LP_FDNet, IC_FDNet
    from models.hypernet import HyperNet
    from models.bayesnet import BayesNet
    from models.gausshypernet import GaussianHyperNet
    from models.mlpnet import MLPNet
    from models.deepensemblenet import DeepEnsembleNet
    from data.toy_functions import generate_meta_task
    
    # Parameters
    input_dim = 1               
    hidden_dim = 32
    epochs = 500
    print_every = 20
    sample = True
    seed = 10
    model_type = 'DeepEnsembleNet'

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Train and test data
    x_train, y_train, x_val, y_val, x_test, y_test, desc = generate_meta_task(n_train=10, n_val=5, n_test=100, seed=seed)
    
    # Create model
    if model_type == 'LP_FDNet':
        model = LP_FDNet(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'IC_FDNet':
        model = IC_FDNet(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'HyperNet':
        model = HyperNet(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'BayesNet':
        model = BayesNet(input_dim, hidden_dim, input_dim, prior_std=1.0)
    elif model_type == 'GaussHyperNet':
        model = GaussianHyperNet(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64, latent_dim=10, prior_std=1.0)
    elif model_type == 'MLPNet':
        model = MLPNet(input_dim, hidden_dim, input_dim, dropout_rate=0.1)
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
    trainer = SingleTaskTrainer(model)
    # Train
    trainer.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    # Evaluate
    preds = trainer.evaluate(x=x_test)