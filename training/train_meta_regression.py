import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_meta_task(x_c, y_c, x_t, y_t, mean, std, desc=None):
    x_c_np = x_c.cpu().numpy().squeeze()
    y_c_np = y_c.cpu().numpy().squeeze()
    x_t_np = x_t.cpu().numpy().squeeze()
    y_t_np = y_t.cpu().numpy().squeeze()

    idx = np.argsort(x_t_np)
    plt.plot(x_t_np[idx], y_t_np[idx], label="Ground Truth", linestyle="--")
    plt.plot(x_t_np[idx], mean[idx], label="Mean Prediction")
    plt.fill_between(x_t_np[idx], mean[idx] - std[idx], mean[idx] + std[idx],
                     alpha=0.3, label="±1 Std Dev")
    plt.scatter(x_c_np, y_c_np, color="red", label="Context Points")
    plt.title(f"Meta-Task: {desc}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plotting function
def plot_loss_curve(losses, title="Training Loss Curve", desc=None):
    title = title + ': ' + desc
    plt.figure(figsize=(8, 5))
    plt.plot([l.item() for l in losses], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_meta_regression(
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

    # # One task: context + target
    # x_c, y_c, x_t, y_t, desc = generate_task_fn()
    x_c, y_c, x_t, y_t = (
        x_c.unsqueeze(0).to(device),
        y_c.unsqueeze(0).to(device),
        x_t.unsqueeze(0).to(device),
        y_t.unsqueeze(0).to(device)
    )
    # Losses
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if isinstance(model, FDNNetwork) or isinstance(model, BayesNetwork):
            outputs = model(x_c, return_kl=True)
        elif isinstance(model, HyperNetwork):
            outputs = model(x_c)
        else:
            print("Unknown model type.")

        y_pred, kl = outputs if isinstance(outputs, tuple) else (outputs, 0.0)
        mse = F.mse_loss(y_pred.squeeze(), y_c.squeeze())

        beta = min(beta_max, epoch / warmup_epochs)
        loss = mse + beta * kl
        loss.backward()
        optimizer.step()

        # if resample:
        #     x_c, y_c, x_t, y_t, desc = generate_task_fn()
        #     x_c, y_c, x_t, y_t = (
        #         x_c.unsqueeze(0).to(device),
        #         y_c.unsqueeze(0).to(device),
        #         x_t.unsqueeze(0).to(device),
        #         y_t.unsqueeze(0).to(device)
        #     )
        # Store losses
        losses.append(loss)
        if epoch % print_every == 0:
            print(f"[Epoch {epoch}] Loss: {loss:.4f} | MSE: {mse.item():.4f} | KL: {kl:.4f} | β: {beta:.2f}")

    # Eval
    model.eval()
    preds = []
    with torch.no_grad():
        if isinstance(model, FDNNetwork) or isinstance(model, BayesNetwork):
            preds = [model(x_t, sample=sample).cpu().numpy() for _ in range(num_samples)]
        elif isinstance(model, HyperNetwork):
            preds = [model(x_t).cpu().numpy() for _ in range(num_samples)]
        # for _ in range(num_samples):
        #     y_sample = model(x_t)
        #     preds.append(y_sample.cpu().numpy())

    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0).squeeze()
    std = preds.std(axis=0).squeeze()

    plot_loss_curve(losses, desc=desc)
    plot_meta_task(x_c, y_c, x_t, y_t, mean, std, desc=desc)


if __name__ == "__main__":
    from data.toy_functions import generate_meta_task
    from models.fdnet import FDNNetwork
    from models.hypernet import HyperNetwork
    from models.bayesnet import BayesNetwork
    input_dim = 10
    epochs = 2000
    print_every = 10
    sample = True
    seed=21234
    model_type = 'FDNet'
    model_type = 'BayesNet'
    # One task: context + target
    x_c, y_c, x_t, y_t, desc = generate_meta_task(n_context=input_dim, n_target=input_dim, seed=seed)
    if model_type == 'FDNet':
        model = FDNNetwork(input_dim=input_dim, hidden_dim=32, output_dim=input_dim, hyper_hidden_dim=64)
    elif model_type == 'HyperNet':
        model = HyperNetwork(input_dim=input_dim, hidden_dim=32, output_dim=input_dim, hyper_hidden_dim=64)
    elif model_type == 'BayesNet':
        model = BayesNetwork(input_dim, hidden_dim=32, output_dim=input_dim, prior_std=1.0)
    train_meta_regression(model=model, 
                          x_c=x_c, y_c=y_c, x_t=x_t, y_t=y_t, desc=desc, 
                          sample=sample, seed=seed, epochs=epochs, print_every=print_every)
