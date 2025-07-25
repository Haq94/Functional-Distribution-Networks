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
        if model.__class__.__name__ == 'DeepEnsembleNetwork':
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

    def train(self, x, y, epochs=1000, beta_param_dict=None,
            print_every=100, batch_size=10, grad_clip=True, MC=1):

        x, y = x.double().to(self.device), y.double().to(self.device)
        N = x.shape[0]
        if N == 0:
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

        dataset = TensorDataset(x, y)
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
                elif self.model.__class__.__name__ == 'DeepEnsembleNetwork':
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

                        loss += sub_loss.item() * x_batch.shape[0]
                        mse += sub_loss.item() * x_batch.shape[0]

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

            total_loss /= N
            total_mse  /= N
            total_kl   /= N

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
            elif self.model.__class__.__name__ == 'DeepEnsembleNetwork':
                y_pred = self._model_forward(x=x)[0].squeeze(0).detach().cpu().numpy()
            else:
                y_pred = self._model_forward(x=x)[0].squeeze(0).detach().cpu().numpy()
                y_pred = [y_pred for _ in range(num_samples)]

        preds = np.stack(y_pred)  # shape: (num_samples, batch_size, output_dim)

        return preds
    
    def _compute_beta(self, epoch, beta_param_dict):
        beta_scheduler = beta_param_dict["beta_scheduler"]
        if beta_scheduler == "zero":
            return 0
        elif beta_scheduler == "linear":
            beta_max = beta_param_dict["beta_max"] 
            warmup_epochs = beta_param_dict["warmup_epochs"]
            return min(beta_max, epoch / warmup_epochs)

    # def plot_results(self, x_c, y_c, x_t, y_t, mean, std, desc=""):
    #     plot_loss_curve(self.losses, self.mses, self.kls, self.betas, desc=desc)
    #     plot_meta_task(x_c.cpu(), y_c.cpu(), x_t.cpu(), y_t.cpu(), mean, std, desc=desc)

if __name__=='__main__':

    from models.fdnet import LP_FDNetwork, IC_FDNetwork
    from models.hypernet import HyperNetwork
    from models.bayesnet import BayesNetwork
    from models.gausshypernet import GaussianHyperNetwork
    from models.mlpnet import DeterministicMLPNetwork
    from models.deepensemblenet import DeepEnsembleNetwork
    from data.toy_functions import generate_meta_task
    
    # Parameters
    input_dim = 1               
    hidden_dim = 32
    epochs = 500
    print_every = 20
    sample = True
    seed = 10
    model_type = 'IC_FDNet'

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # One task: context + target
    x_c, y_c, x_t, y_t, desc = generate_meta_task(n_context=10, n_target=100, seed=seed)
    
    # Create model
    if model_type == 'LP_FDNet':
        model = LP_FDNetwork(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'IC_FDNet':
        model = IC_FDNetwork(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'HyperNet':
        model = HyperNetwork(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'BayesNet':
        model = BayesNetwork(input_dim, hidden_dim, input_dim, prior_std=1.0)
    elif model_type == 'GaussHyperNet':
        model = GaussianHyperNetwork(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64, latent_dim=10, prior_std=1.0)
    elif model_type == 'MLPNet':
        model = DeterministicMLPNetwork(input_dim, hidden_dim, input_dim, dropout_rate=0.1)
    elif model_type == 'DeepEnsembleNet':
        model = DeepEnsembleNetwork(
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
    
    print('Running', type(model).__name__, '======================================================================')
    
    # Create training class instance
    trainer = SingleTaskTrainer(model)
    # Train
    trainer.train(x=x_c, y=y_c)
    # Evaluate
    preds, mean, std = trainer.evaluate(x=x_t)
    # Plot
    trainer.plot_results(x_c=x_c, y_c=y_c, x_t=x_t, y_t=y_t, mean=mean, std=std, desc=desc)


# OLD CODE ===================================================================================================

    # def train(self, x, y, epochs=1000, warmup_epochs=500, beta_max=1.0, print_every=100):
    #     x, y = x.to(self.device), y.to(self.device)
    #     N = x.shape[0]
    #     if N == 0:
    #         raise ValueError("Empty context set provided.")

    #     for epoch in range(1, epochs + 1):
    #         self.model.train()
    #         self.optimizer.zero_grad()
    #         total_loss = None
    #         total_kl = 0.0
    #         total_mse = 0.0

    #         for n in range(N):
    #             x_n = x[n:n+1]
    #             y_n = y[n:n+1]
    #             y_predn, kl = self._model_forward(x_n)
    #             mse = F.mse_loss(y_predn.squeeze(), y_n.squeeze())
    #             beta = self._compute_beta(epoch, warmup_epochs, beta_max)

    #             loss = (mse + beta * kl)/N
    #             total_loss = loss if total_loss is None else total_loss + loss
    #             total_kl += kl/N
    #             total_mse += mse/N

    #         total_loss.backward()
    #         self.optimizer.step()

    #         # Store for plotting
    #         self.losses.append(total_loss.item())
    #         self.mses.append(total_mse.detach().item())
    #         self.kls.append(total_kl.detach().item())
    #         self.betas.append(beta)

    #         if epoch % print_every == 0:
    #             print(f"[Epoch {epoch}] Loss: {total_loss:.4f} | MSE: {total_mse:.4f} | KL: {total_kl:.4f} | β: {beta:.2f}")

