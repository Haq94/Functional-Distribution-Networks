from utils.plots import plot_meta_task
from utils.plots import plot_loss_curve
import torch
import torch.nn.functional as F
import numpy as np
import random

from data.toy_functions import generate_meta_task, sample_function
from models.fdnet import LP_FDNetwork, IC_FDNetwork
from models.hypernet import HyperNetwork
from models.bayesnet import BayesNetwork
from models.gausshypernet import GaussianHyperNetwork
from models.mlpnet import DeterministicMLPNetwork
from models.deepensemblenet import DeepEnsembleNetwork

def train_single_task_regression(
    model,
    x_c, y_c,
    x_t, y_t, desc,
    sample = True,
    epochs=1000,
    beta_max=1.0,
    warmup_epochs=500,
    print_every=100,
    num_samples=30,
    device=None,
    seed = None):
    
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Context and target points for task
    x_c, y_c, x_t, y_t = (
        torch.from_numpy(x_c).float().to(device),
        torch.from_numpy(y_c).float().to(device),
        torch.from_numpy(x_t).float().to(device),
        torch.from_numpy(y_t).float().to(device)
    )
    # Losses, mse, and kl
    losses_list = []
    mse_list = []
    kl_list = []
    beta_list = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if isinstance(model, LP_FDNetwork) or isinstance(model, IC_FDNetwork) or isinstance(model, BayesNetwork) or isinstance(model, GaussianHyperNetwork):
            outputs = model(x_c, return_kl=True)
        elif isinstance(model, HyperNetwork) or isinstance(model, DeterministicMLPNetwork):
            outputs = model(x_c)
        elif isinstance(model, DeepEnsembleNetwork):
            outputs = model(x_c).mean(dim=0)
        else:
            print("Unknown model type.")

        y_pred, kl = outputs if isinstance(outputs, tuple) else (outputs, 0.0)
        mse = F.mse_loss(y_pred.squeeze(), y_c.squeeze())

        beta = min(beta_max, epoch / warmup_epochs)
        loss = mse + beta * kl
        loss.backward()
        optimizer.step()

        # Store losses
        losses_list.append(loss)
        mse_list.append(mse)
        kl_list.append(kl)
        beta_list.append(beta)
        if epoch % print_every == 0:
            print(f"[Epoch {epoch}] Loss: {loss:.4f} | MSE: {mse.item():.4f} | KL: {kl:.4f} | β: {beta:.2f}")

    if __name__ == "__main__":
        # Eval
        model.eval()
        preds = []
        
        with torch.no_grad():
            if isinstance(model, IC_FDNetwork) or isinstance(model, LP_FDNetwork) or isinstance(model, BayesNetwork) or isinstance(model, GaussianHyperNetwork):
                preds = [model(x_t, sample=sample).cpu().numpy() for _ in range(num_samples)]
            elif isinstance(model, HyperNetwork) or isinstance(model, DeterministicMLPNetwork):
                preds = [model(x_t).cpu().numpy() for _ in range(num_samples)]
            elif isinstance(model, DeepEnsembleNetwork):
                # preds = [model(x_t).mean(dim=0).cpu().numpy() for _ in range(num_samples)]
                preds= model(x_t).cpu().numpy()

        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0).squeeze()
        std = preds.std(axis=0).squeeze()

        plot_loss_curve(losses_list, mse_list, kl_list, beta_list, desc=desc)
        plot_meta_task(x_c, y_c, x_t, y_t, mean, std, desc=desc)

        return preds, mean, std, y_t.cpu().numpy().squeeze()
    else:
        return model



if __name__ == "__main__":
    x_min = -5
    x_max = 5
    batches = 4
    input_dim = 10
    hidden_dim = 32
    epochs = 5000
    print_every = 100
    sample = True
    seed=None
    model_type = 'DeepEnsembleNet'

    # One task: context + target
    f, desc = sample_function(seed=seed)
    x_c = np.empty((0, input_dim))
    y_c = np.empty((0, input_dim))

    for n in range(batches):
        x_cn = np.random.uniform(low=x_min, high=x_max, size=input_dim)
        y_cn = f(x_cn)
        x_c = np.vstack((x_c, x_cn))
        y_c = np.vstack((y_c, y_cn))

    x_t = np.random.uniform(low=x_min, high=x_max, size=(1, input_dim))
    y_t = f(x_t)

    # x_c, y_c, x_t, y_t, desc = generate_meta_task(n_context=input_dim, n_target=input_dim, seed=seed)
    if model_type == 'LP_FDNet':
        hyper_hidden_dim = 64
        model = LP_FDNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, hyper_hidden_dim=hyper_hidden_dim)
    elif model_type == 'IC_FDNet': 
        hyper_hidden_dim = 64
        model = IC_FDNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, hyper_hidden_dim=hyper_hidden_dim)
    elif model_type == 'HyperNet':
        hyper_hidden_dim = 64
        model = HyperNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, hyper_hidden_dim=hyper_hidden_dim)
    elif model_type == 'BayesNet':
        model = BayesNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, prior_std=1.0)
    elif model_type == 'GaussHyperNet':
        hyper_hidden_dim = 64
        latent_dim = 10
        model = GaussianHyperNetwork(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, hyper_hidden_dim=hyper_hidden_dim, latent_dim=latent_dim, prior_std=1.0)
    elif model_type == 'MLPNet':
        dropout_rate = 0.1
        model = DeterministicMLPNetwork(input_dim, hidden_dim=hidden_dim, output_dim=input_dim, dropout_rate=dropout_rate)
    elif model_type == 'DeepEnsembleNet':
        network_class = DeterministicMLPNetwork
        num_models = 5
        dropout_rate = 0.1
        seed_list = [0, 1, 2, 3, 4]
        model = DeepEnsembleNetwork(network_class=DeterministicMLPNetwork, num_models=num_models, seed_list=seed_list, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim, dropout_rate=dropout_rate)

    train_single_task_regression(model=model, 
                          x_c=x_c, y_c=y_c, x_t=x_t, y_t=y_t, desc=desc, 
                          sample=sample, seed=seed, epochs=epochs, print_every=print_every)



