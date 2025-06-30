import torch
import torch.nn.functional as F
import numpy as np
import random
from utils.plots import plot_meta_task, plot_loss_curve

class SingleTaskTrainer:
    def __init__(self, model, optimizer=None, lr=1e-3, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr)

        self.losses = []
        self.mses = []
        self.kls = []
        self.betas = []

    def _model_forward(self, x, return_kl=True, sample=True):
        if hasattr(self.model, 'forward') and ('return_kl' and 'sample' in self.model.forward.__code__.co_varnames):
            return self.model(x, return_kl=return_kl, sample=sample)
        else:
            return self.model(x), 0.0

    def train(self, x_c, y_c, epochs=1000, warmup_epochs=500, beta_max=1.0, print_every=100):
        x_c, y_c = x_c.to(self.device), y_c.to(self.device)
        N_c = x_c.size(0)
        if N_c == 0:
            raise ValueError("Empty context set provided.")

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            total_loss = None
            total_kl = 0.0
            total_mse = 0.0

            for n in range(N_c):
                x_n = x_c[n:n+1]
                y_n = y_c[n:n+1]
                y_predn, kl = self._model_forward(x_n)
                mse = F.mse_loss(y_predn.squeeze(), y_n.squeeze())
                beta = self._compute_beta(epoch, warmup_epochs, beta_max)

                loss = (mse + beta * kl)/N_c
                total_loss = loss if total_loss is None else total_loss + loss
                total_kl += kl/N_c
                total_mse += mse/N_c

            total_loss.backward()
            self.optimizer.step()

            # Store for plotting
            self.losses.append(total_loss.item())
            self.mses.append(total_mse.detach().item())
            self.kls.append(total_kl.detach().item())
            self.betas.append(beta)

            if epoch % print_every == 0:
                print(f"[Epoch {epoch}] Loss: {total_loss:.4f} | MSE: {total_mse:.4f} | KL: {total_kl:.4f} | β: {beta:.2f}")

    def evaluate(self, x_t, num_samples=30, sample=True):
        self.model.eval()
        x_t = x_t.to(self.device)
        N_t = x_t.shape[0]
        preds = []

        with torch.no_grad():
            for n in range(N_t):    
                x_tn = x_t[n:n+1]
                if hasattr(self.model, 'forward') and ('return_kl' and 'sample' in self.model.forward.__code__.co_varnames):
                    y_pred = [self._model_forward(x=x_tn, return_kl=False, sample=sample).squeeze(0).cpu().numpy() for _ in range(num_samples)]
                else:
                    y_pred = [self._model_forward(x=x_tn).squeeze(0).cpu().numpy() for _ in range(num_samples)]
                preds.append(np.stack(y_pred))

        preds = np.stack(preds)  # shape: (batch_size, num_samples, output_dim)
        mean = preds.mean(1)
        std = preds.var(1)

        return preds, mean.squeeze(), std.squeeze()
    
    def _compute_beta(self, epoch, warmup_epochs, beta_max):
        return min(beta_max, epoch / warmup_epochs)

    def plot_results(self, x_c, y_c, x_t, y_t, mean, std, desc=""):
        plot_loss_curve(self.losses, self.mses, self.kls, self.betas, desc=desc)
        plot_meta_task(x_c.cpu(), y_c.cpu(), x_t.cpu(), y_t.cpu(), mean, std, desc=desc)

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
    trainer.train(x_c=x_c, y_c=y_c)
    # Evaluate
    preds, mean, std = trainer.evaluate(x_t=x_t)
    # Plot
    trainer.plot_results(x_c=x_c, y_c=y_c, x_t=x_t, y_t=y_t, mean=mean, std=std, desc=desc)


