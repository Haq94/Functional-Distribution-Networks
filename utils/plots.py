import matplotlib.pyplot as plt
import numpy as np

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

def plot_loss_curve(losses, mses, kls, betas, title="Training Loss Curve", desc=None):
    """
    Plots total loss, MSE, KL divergence, and beta schedule in subplots.
    """
    if desc:
        title += f": {desc}"

    losses = [l.item() if hasattr(l, 'item') else l for l in losses]
    mses   = [m.item() if hasattr(m, 'item') else m for m in mses]
    kls    = [k.item() if hasattr(k, 'item') else k for k in kls]
    betas  = [b.item() if hasattr(b, 'item') else b for b in betas]

    epochs = range(len(losses))

    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top subplot: Loss curves
    axs[0].plot(epochs, losses, label="Total Loss", color="blue")
    axs[0].plot(epochs, mses,   label="MSE", color="green")
    axs[0].plot(epochs, kls,    label="KL Divergence", color="red")
    axs[0].set_ylabel("Loss Value")
    axs[0].set_title(title)
    axs[0].grid(True)
    axs[0].legend()

    # Bottom subplot: Beta schedule
    axs[1].plot(epochs, betas, label="Beta", color="black")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("β")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()