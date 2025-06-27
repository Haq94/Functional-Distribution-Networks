import torch
import torch.nn.functional as F
import numpy as np
import random
from utils.plots import plot_meta_task, plot_loss_curve
from data.toy_functions import generate_meta_task
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
    sample=True,
    epochs=1000,
    beta_max=1.0,
    warmup_epochs=500,
    print_every=100,
    num_samples=30,
    device=None,
    seed=None,
    plots=True
):
    # Number of context and target points
    n_target = x_t.numel()
    n_context = x_c.numel()
    # Set seed
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    # Store on proper device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Context and target points for task
    x_c, y_c, x_t, y_t = (
        x_c.to(device),
        y_c.to(device),
        x_t.to(device),
        y_t.to(device)
    )

    # === Train ===

    # Loss tracking
    losses_list, mse_list, kl_list, beta_list = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = None
        kl_total = 0.0
        mse_total = 0.0
        for n in range(n_context):
            x_cn = x_c[n:n+1]
            y_cn = y_c[n:n+1]
            if isinstance(model, (LP_FDNetwork, IC_FDNetwork, BayesNetwork, GaussianHyperNetwork)):
                outputs = model(x_cn.unsqueeze(0), return_kl=True)
            elif isinstance(model, (HyperNetwork, DeterministicMLPNetwork)):
                outputs = model(x_cn.unsqueeze(0))
            elif isinstance(model, DeepEnsembleNetwork):
                outputs = model(x_cn.unsqueeze(0)).mean(dim=0)
            else:
                raise TypeError("Unknown model type.")

            y_predn, kln = outputs if isinstance(outputs, tuple) else (outputs, 0.0)
            msen = F.mse_loss(y_predn.squeeze(), y_cn.squeeze())

            beta = min(beta_max, epoch / warmup_epochs)
            term = msen + beta * kln
            loss = term if loss is None else loss + term
            mse_total = mse_total + msen
            kl_total = kl_total + kln

        loss = loss / n_context
        mse_total = mse_total / n_context
        kl_total = kl_total / n_context

        loss.backward()
        optimizer.step()

        losses_list.append(loss.item())
        mse_list.append(mse_total.item())
        kl_list.append(kl_total.item() if torch.is_tensor(kl_total) else 0.0)
        beta_list.append(beta)

        if epoch % print_every == 0:
            print(f"[Epoch {epoch}] Loss: {loss:.4f} | MSE: {mse_total.item():.4f} | KL: {kl_total:.4f} | β: {beta:.2f}")

    # === Evaluation ===
    model.eval()
    preds = []
    all_preds = []

    with torch.no_grad():
        for n in range(n_target):
            x_tn = x_t[n:n+1]
            if isinstance(model, (IC_FDNetwork, LP_FDNetwork, BayesNetwork, GaussianHyperNetwork)):
                preds = [model(x_tn.unsqueeze(0), sample=sample).cpu().numpy() for _ in range(num_samples)]
            elif isinstance(model, (HyperNetwork, DeterministicMLPNetwork)):
                preds = [model(x_tn.unsqueeze(0)).cpu().numpy() for _ in range(num_samples)]
            elif isinstance(model, DeepEnsembleNetwork):
                preds = model(x_tn.unsqueeze(0)).cpu().numpy()  # Already ensemble-averaged

            all_preds.append(np.stack(preds, axis=0))  # shape: [num_samples, output_dim]

    all_preds = np.stack(all_preds, axis=1)  # shape: [num_samples, num_points, output_dim]
    mean = all_preds.mean(axis=0).squeeze()  # shape: [num_points, output_dim]
    std = all_preds.std(axis=0).squeeze()    # shape: [num_points, output_dim]

    # === Plots ===
    if plots:
        plot_loss_curve(losses_list, mse_list, kl_list, beta_list, desc=desc)
        plot_meta_task(x_c, y_c, x_t, y_t, mean, std, desc=desc)

    return model, all_preds, mean, std, y_t.cpu().numpy().squeeze()


if __name__ == "__main__":
    input_dim = 1               # <-- Use 1D input for clean visualizations
    hidden_dim = 32
    epochs = 500
    print_every = 20
    sample = True
    seed = 10
    model_type = 'IC_FDNet'

    # One task: context + target
    x_c, y_c, x_t, y_t, desc = generate_meta_task(n_context=10, n_target=100, seed=seed)

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

    train_single_task_regression(
        model=model,
        x_c=x_c, y_c=y_c,
        x_t=x_t, y_t=y_t,
        desc=desc,
        sample=sample,
        seed=seed,
        epochs=epochs,
        print_every=print_every
    )
