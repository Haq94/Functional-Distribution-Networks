import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models.fdnet import FDNNetwork
from data.toy_functions import generate_meta_task
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Visualization ---
def plot_prediction(x, y_true, y_pred_samples, save_path=None):
    x_np = x.numpy().squeeze()
    idx = np.argsort(x_np)
    x_sorted = x_np[idx]
    y_true_sorted = y_true.numpy().squeeze()[idx]

    y_samples_sorted = [y[idx] for y in y_pred_samples]
    mean_pred = torch.stack(y_samples_sorted).mean(dim=0)
    std_pred = torch.stack(y_samples_sorted).std(dim=0)

    plt.figure(figsize=(6, 4))
    plt.plot(x_sorted, y_true_sorted, label='True Function', color='black')
    for i, y in enumerate(y_samples_sorted[:5]):
        plt.plot(x_sorted, y.numpy(), color='blue', alpha=0.3)
    plt.fill_between(x_sorted,
                     (mean_pred - std_pred).numpy(),
                     (mean_pred + std_pred).numpy(),
                     color='blue', alpha=0.2, label='Uncertainty')
    plt.plot(x_sorted, mean_pred.numpy(), color='red', label='Mean Prediction')
    plt.legend()
    plt.title("FDN Prediction with Sampled Weights")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
# --- Training ---
def train_model(model, epochs=1000, print_every=100):
    input_dim = model.input_dim if hasattr(model, 'input_dim') else 10
    output_dim = model.output_dim if hasattr(model, 'output_dim') else input_dim

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()

        x, y, _, _, _ = generate_meta_task(n_context=input_dim)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y.squeeze())
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
            
            # Visualization
            model.eval()
            with torch.no_grad():
                x_vis, y_vis = generate_dataset(n_tasks=1000, n_points=input_dim)
                x_vis = torch.tensor(x_vis, dtype=torch.float32).to(device)
                y_vis = torch.tensor(y_vis, dtype=torch.float32).to(device)
                y_samples = [model(x_vis) for _ in range(10)]
                plot_prediction(x_vis, y_vis, y_samples, save_path=f"Results/fdn_epoch{epoch}.png")


if __name__ == "__main__":

    input_dim = 10
    hidden_dim = 20
    output_dim = input_dim
    hyper_hidden_dim = 15

    model = FDNNetwork(input_dim, hidden_dim, output_dim, hyper_hidden_dim)    

    train_model(model=model)