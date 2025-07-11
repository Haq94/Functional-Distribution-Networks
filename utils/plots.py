import os
import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import metrics
from utils.saver import save_analysis_arrays

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



def plot_loss_curve(losses, mses, kls, betas, title="Training Loss Curve", desc=None, save_path=None, block=False):
    """
    Plots total loss, MSE, KL divergence, and beta schedule in subplots.

    Args:
        losses (list): total loss per epoch
        mses (list): mse per epoch
        kls (list): kl divergence per epoch
        betas (list): beta value per epoch
        title (str): overall title for the plot
        desc (str): additional description
        save_path (str): full file path to save the figure (e.g., 'results/ModelX/loss_curve.png')
        block (bool): if True, blocks the plot window (interactive mode)
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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if block:
        plt.show()

    plt.close()

def plot_regression_diagnostics(preds, x_c, y_c, x_t, y_t, desc, ind_c, metric_outputs=None, block=False, save_dir=None):
    """
    Plot uncertainty diagnostics for regression tasks.

    Args:
        preds (np.ndarray): (n_samples, n_points, 1) multiple stochastic predictions
        x_c (torch.Tensor): context x values
        y_c (torch.Tensor): context y values
        x_t (torch.Tensor): target x values
        y_t (torch.Tensor): target y values
        desc (str): plot title description
        ind_c (np.ndarray): indices of context points in x_t
        metric_outputs (tuple): consist of relevant metrics
        block (bool): whether to block plt.show() for interactive use
        save_dir (str): save directory
    """
    # Convert all arrays from torch tensors to numpy arrays
    x_c_np = x_c.cpu().numpy().squeeze().astype(np.float64)
    y_c_np = y_c.cpu().numpy().squeeze().astype(np.float64)
    x_t_np = x_t.cpu().numpy().squeeze().astype(np.float64)
    y_t_np = y_t.cpu().numpy().squeeze().astype(np.float64)
    preds_np = preds.astype(np.float64)
    x_c_min = x_c_np.min()
    x_c_max = x_c_np.max()

    # Compute metrics if they're not already computed
    if metric_outputs is None:
        mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nll = metrics(preds_np, y_t_np)
    else:
        mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nll = metric_outputs


    # Residual Scatter
    _, axs = plt.subplots(2, 1, num="Residual Scatter Plot", figsize=(10, 8), sharex=True)
    for ii in range(res_prec.shape[1]):
        axs[0].scatter(x_t_np, res_prec[:, ii], alpha=0.05, color=np.random.rand(3))
    axs[0].axvline(x=x_c_min, color='red', linestyle='--')
    axs[0].axvline(x=x_c_max, color='red', linestyle='--')
    axs[0].set_title(f"Residual Precision Task: {desc}")
    axs[0].grid(True)

    for ii in range(res_acc.shape[1]):
        axs[1].scatter(x_t_np, res_acc[:, ii], alpha=0.05, color=np.random.rand(3))
    axs[1].plot(x_t_np, bias, label='Bias', color='red')
    axs[1].axvline(x=x_c_min, color='red', linestyle='--')
    axs[1].axvline(x=x_c_max, color='red', linestyle='--')
    axs[1].set_title(f"Residual Accuracy Task: {desc}")
    axs[1].set_xlabel("x") 
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "residual_scatter.png"), dpi=300)
    if block:
        plt.show()

    # Mean Prediction
    plt.figure(num="Mean Prediction Plot", figsize=(10, 8))
    for ii in range(res_acc.shape[1]):
        plt.scatter(x_t_np, preds[ii, :], alpha=0.05, color=np.random.rand(3))
    plt.plot(x_t_np, y_t_np, label="Ground Truth", linestyle="--")
    plt.plot(x_t_np, mean, label="Mean Prediction")
    plt.fill_between(x_t_np, mean - std, mean + std, alpha=0.3, label="±1 Std Dev")
    plt.scatter(x_c_np, y_c_np, color="red", label="Context Points")
    plt.axvline(x=x_c_min, color='red', linestyle='--')
    plt.axvline(x=x_c_max, color='red', linestyle='--')
    plt.title(f"Mean Function Task: {desc}")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "mean_prediction.png"), dpi=300)
    if block:
        plt.show()

    # Zoomed in Mean Prediction
    plt.figure(num="Zoomed in Mean Prediction Plot", figsize=(10, 8))
    for ii in range(res_acc.shape[1]):
        plt.scatter(x_t_np, preds[ii, :], alpha=0.05, color=np.random.rand(3))
    plt.plot(x_t_np, y_t_np, label="Ground Truth", linestyle="--")
    plt.plot(x_t_np, mean, label="Mean Prediction")
    plt.fill_between(x_t_np, mean - std, mean + std, alpha=0.3, label="±1 Std Dev")
    plt.scatter(x_c_np, y_c_np, color="red", label="Context Points", alpha=0.5)
    plt.axvline(x=x_c_min, color='red', linestyle='--')
    plt.axvline(x=x_c_max, color='red', linestyle='--')
    plt.title(f"Zoomed Mean Function Task: {desc}")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "zoomed_mean_prediction.png"), dpi=300)
    if block:
        plt.show()

    # Mean + Variance in dB
    _, axs1 = plt.subplots(2, 1, num="Variance and Mean Plot", figsize=(10, 8), sharex=True)
    axs1[0].plot(x_t_np, mean)
    axs1[0].scatter(x_t_np[ind_c], mean[ind_c], label='Context', color='red')
    axs1[0].axvline(x=x_c_min, color='red', linestyle='--')
    axs1[0].axvline(x=x_c_max, color='red', linestyle='--')
    axs1[0].set_title(f"Mean Function: {desc}")
    axs1[0].grid(True)

    axs1[1].plot(x_t_np, 10 * np.log10(var))
    axs1[1].scatter(x_t_np[ind_c], 10 * np.log10(var[ind_c]), label='Context', color='red')
    axs1[1].axvline(x=x_c_min, color='red', linestyle='--')
    axs1[1].axvline(x=x_c_max, color='red', linestyle='--')
    axs1[1].set_title(f"Variance (dB): {desc}")
    axs1[1].set_xlabel("x")
    axs1[1].grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "mean_and_var_dB.png"), dpi=300)
    if block:
        plt.show()

    # Bias and MSE in dB
    _, axs2 = plt.subplots(2, 1, num="Bias and MSE Plot", figsize=(10, 8), sharex=True)
    axs2[0].plot(x_t_np, bias)
    axs2[0].scatter(x_t_np[ind_c], bias[ind_c], label='Context', color='red')
    axs2[0].axvline(x=x_c_min, color='red', linestyle='--')
    axs2[0].axvline(x=x_c_max, color='red', linestyle='--')
    axs2[0].set_title(f"Bias: {desc}")
    axs2[0].grid(True)

    axs2[1].plot(x_t_np, 10 * np.log10(mse))
    axs2[1].scatter(x_t_np[ind_c], 10 * np.log10(mse[ind_c]), label='Context', color='red')
    axs2[1].axvline(x=x_c_min, color='red', linestyle='--')
    axs2[1].axvline(x=x_c_max, color='red', linestyle='--')
    axs2[1].set_title(f"MSE (dB): {desc}")
    axs2[1].set_xlabel("x")
    axs2[1].grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "bias_and_mse_dB.png"), dpi=300)
    if block:
        plt.show()

    # NLL
    plt.figure(num="NLL Plot", figsize=(10, 6))
    plt.plot(x_t_np, nll, label="NLL", color="purple")
    plt.scatter(x_t_np[ind_c], nll[ind_c], label="Context Points", color="red")
    plt.axvline(x=x_c_min, color='red', linestyle='--')
    plt.axvline(x=x_c_max, color='red', linestyle='--')
    plt.title(f"NLL per Target Point: {desc}")
    plt.xlabel("x")
    plt.ylabel("NLL")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "nll.png"), dpi=300)
    if block:
        plt.show()


    plt.close()

    # # Save analysis arrays
    # if save_dir:
    #     analysis_dir = os.path.join(save_dir, "..", "analysis")
    #     save_analysis_arrays(
    #         (mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nll),
    #         analysis_dir
    #     )


# OLD CODE =======================================================================================================

# def plot_loss_curve(losses, mses, kls, betas, title="Training Loss Curve", desc=None):
#     """
#     Plots total loss, MSE, KL divergence, and beta schedule in subplots.
#     """
#     if desc:
#         title += f": {desc}"

#     losses = [l.item() if hasattr(l, 'item') else l for l in losses]
#     mses   = [m.item() if hasattr(m, 'item') else m for m in mses]
#     kls    = [k.item() if hasattr(k, 'item') else k for k in kls]
#     betas  = [b.item() if hasattr(b, 'item') else b for b in betas]

#     epochs = range(len(losses))

#     _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     # Top subplot: Loss curves
#     axs[0].plot(epochs, losses, label="Total Loss", color="blue")
#     axs[0].plot(epochs, mses,   label="MSE", color="green")
#     axs[0].plot(epochs, kls,    label="KL Divergence", color="red")
#     axs[0].set_ylabel("Loss Value")
#     axs[0].set_title(title)
#     axs[0].grid(True)
#     axs[0].legend()

#     # Bottom subplot: Beta schedule
#     axs[1].plot(epochs, betas, label="Beta", color="black")
#     axs[1].set_xlabel("Epoch")
#     axs[1].set_ylabel("β")
#     axs[1].grid(True)
#     axs[1].legend()

#     plt.tight_layout()
#     plt.show()