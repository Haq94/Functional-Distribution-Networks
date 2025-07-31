import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import metrics
from utils.saver.general_saver import save_plot


def plot_loss_curve(losses, mses, kls, betas, title="Training Loss Curve", desc=None, save_dir=None, block=False):
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

    _, axs = plt.subplots(2, 1, num="Total Loss, MSE, and KL Divergence Plot", figsize=(10, 8), sharex=True)

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

    if save_dir:
        save_plot(save_dir, "loss_curve")
    if block:
        plt.show()
    plt.close()

def plot_residual_scatter(x_t_np, res_prec, res_acc, bias, x_c_min, x_c_max, desc, save_dir=None, block=False):
    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
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
    axs[1].legend()
    axs[1].set_xlabel("x")
    axs[1].grid(True)
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "residual_scatter")
    if block:
        plt.show()
    plt.close()


def plot_mean_prediction(x_t_np, y_t_np, mean, std, preds, x_c_np, y_c_np, x_c_min, x_c_max, desc, save_dir=None, block=False, zoom=False):
    plt.figure(figsize=(10, 8))
    for ii in range(preds.shape[0]):
        plt.scatter(x_t_np, preds[ii, :], alpha=0.05, color=np.random.rand(3))
    plt.plot(x_t_np, y_t_np, label="Ground Truth", linestyle="--")
    plt.plot(x_t_np, mean, label="Mean Prediction")
    plt.fill_between(x_t_np, mean - std, mean + std, alpha=0.3, label="±1 Std Dev")
    plt.scatter(x_c_np, y_c_np, color="red", label="Training Points", alpha=0.5)
    plt.axvline(x=x_c_min, color='red', linestyle='--')
    plt.axvline(x=x_c_max, color='red', linestyle='--')
    if zoom:
        plt.ylim([y_t_np.min()-1, y_t_np.max()+1])
    title = "Zoomed " if zoom else ""
    plt.title(f"{title}Mean Function Task: {desc}")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    if save_dir:
        save_plot(save_dir, "zoomed_mean_prediction" if zoom else "mean_prediction")
    if block:
        plt.show()
    plt.close()


def plot_variance(x_t_np, var, mean, ind_c, x_c_min, x_c_max, desc, save_dir=None, block=False):
    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(x_t_np, mean)
    axs[0].scatter(x_t_np[ind_c], mean[ind_c], label='Training Points', color='red')
    axs[0].set_title(f"Mean Function: {desc}")
    axs[0].axvline(x=x_c_min, color='red', linestyle='--')
    axs[0].axvline(x=x_c_max, color='red', linestyle='--')
    axs[0].grid(True)

    axs[1].plot(x_t_np, 10 * np.log10(var))
    axs[1].scatter(x_t_np[ind_c], 10 * np.log10(var[ind_c]), label='Training Points', color='red')
    axs[1].set_title(f"Variance (dB): {desc}")
    axs[1].axvline(x=x_c_min, color='red', linestyle='--')
    axs[1].axvline(x=x_c_max, color='red', linestyle='--')
    axs[1].grid(True)
    axs[1].set_xlabel("x")
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "mean_and_var_dB")
    if block:
        plt.show()
    plt.close()


def plot_bias_mse(x_t_np, bias, mse, ind_c, x_c_min, x_c_max, desc, save_dir=None, block=False):
    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(x_t_np, bias)
    axs[0].scatter(x_t_np[ind_c], bias[ind_c], label='Training Points', color='red')
    axs[0].set_title(f"Bias: {desc}")
    axs[0].axvline(x=x_c_min, color='red', linestyle='--')
    axs[0].axvline(x=x_c_max, color='red', linestyle='--')
    axs[0].grid(True)

    axs[1].plot(x_t_np, 10 * np.log10(mse))
    axs[1].scatter(x_t_np[ind_c], 10 * np.log10(mse[ind_c]), label='Training Points', color='red')
    axs[1].set_title(f"MSE (dB): {desc}")
    axs[1].axvline(x=x_c_min, color='red', linestyle='--')
    axs[1].axvline(x=x_c_max, color='red', linestyle='--')
    axs[1].grid(True)
    axs[1].set_xlabel("x")
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "bias_and_mse_dB")
    if block:
        plt.show()
    plt.close()

def plot_nll(x_t_np, nll, ind_c, x_c_min, x_c_max, desc, save_dir=None, block=False):
    plt.figure(figsize=(10, 6))
    plt.plot(x_t_np, nll, label="NLL", color="purple")
    plt.scatter(x_t_np[ind_c], nll[ind_c], label="Training Points", color="red")
    plt.axvline(x=x_c_min, color='red', linestyle='--')
    plt.axvline(x=x_c_max, color='red', linestyle='--')
    plt.title(f"NLL per Target Point: {desc}")
    plt.xlabel("x")
    plt.ylabel("NLL")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "nll")
    if block:
        plt.show()
    plt.close()

def plot_mse_bias_sq_scatter(bias, mse, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=None, block=False):
    plt.figure(figsize=(10,8))
    plt.scatter(10*np.log10(bias[ind_test_interp]**2), 10*np.log10(mse[ind_test_interp]), alpha=1.0, s=10, c="red", label="Test Interpolation Points")
    plt.scatter(10*np.log10(bias[ind_test_extrap]**2), 10*np.log10(mse[ind_test_extrap]), alpha=1.0, s=10, c="blue", label="Test Extrapolation Points")
    plt.scatter(10*np.log10(bias[ind_train]**2), 10*np.log10(mse[ind_train]), alpha=1.0, s=10, c="green", label="Training Points")
    plt.xlabel("$Bias^{2}$ (dB)")
    plt.ylabel("$MSE$ (dB)")
    plt.title("$MSE$ vs $Bias^{2}$ (dB): " + f"{desc}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "mse_dB_bias_sq_vs_scatter")
    if block:
        plt.show()
    plt.close()

def plot_mse_bias_sq_scatter_2x2(bias, mse, ind_test, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=None, block=False):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    titles = ["Test Points", "Training Points", "Test Interpolation Points", "Test Extrapolation Points"]
    indices = [ind_test, ind_train, ind_test_interp, ind_test_extrap]
    colors = ["red", "green", "blue", "orange"]

    for ax, title, idx, color in zip(axes.flat, titles, indices, colors):
        ax.scatter(10 * np.log10(bias[idx]**2), 10 * np.log10(mse[idx]),
                   alpha=1.0, s=10, c=color)
        ax.set_title(title)
        ax.grid(True)
    
    fig.suptitle(f"$MSE$ vs $Bias^2$ (dB): {desc}", fontsize=16)
    fig.supxlabel("$Bias^2$ (dB)")
    fig.supylabel("$MSE$ (dB)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    if save_dir:
        save_plot(save_dir, f"mse_vs_bias_sq_2x2_grid_{desc}")
    if block:
        plt.show()    
    plt.close()

def plot_mse_var(var, mse, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=None, block=False):
    plt.figure(figsize=(10,8))
    plt.scatter(10*np.log10(var[ind_test_interp]**2), 10*np.log10(mse[ind_test_interp]), alpha=1.0, s=10, c="red", label="Test Interpolation Points")
    plt.scatter(10*np.log10(var[ind_test_extrap]**2), 10*np.log10(mse[ind_test_extrap]), alpha=1.0, s=10, c="blue", label="Test Extrapolation Points")
    plt.scatter(10*np.log10(var[ind_train]**2), 10*np.log10(mse[ind_train]), alpha=1.0, s=10, c="green", label="Training Points")
    plt.xlabel("$\sigma_{\hat{y}}^{2}$ (dB)")
    plt.ylabel("$MSE$ (dB)")
    plt.title("$MSE$ vs $\sigma_{\hat{y}}^{2}$ (dB): " + f"{desc}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "mse_dB_vs_var_dB_scatter")
    if block:
        plt.show()
    plt.close()

def plot_mse_var_2x2(var, mse, ind_test, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=None, block=False):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    titles = ["Test Points", "Training Points", "Test Interpolation Points", "Test Extrapolation Points"]
    indices = [ind_test, ind_train, ind_test_interp, ind_test_extrap]
    colors = ["red", "green", "blue", "orange"]

    for ax, title, idx, color in zip(axes.flat, titles, indices, colors):
        ax.scatter(10 * np.log10(var[idx]**2), 10 * np.log10(mse[idx]),
                   alpha=1.0, s=10, c=color)
        ax.set_title(title)
        ax.grid(True)
    
    fig.suptitle("$MSE$ vs $\sigma_{\hat{y}}^{2}$ (dB): " + f"{desc}", fontsize=16)
    fig.supxlabel("$\sigma_{\hat{y}}^{2}$ (dB)")
    fig.supylabel("$MSE$ (dB)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    if save_dir:
        save_plot(save_dir, "mse_dB_vs_var_dB_2x2_grid")
    if block:
        plt.show()
    plt.close()

def plot_bias_sq_var(var, bias, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=None, block=False):
    plt.figure(figsize=(10,8))
    plt.scatter(10*np.log10(var[ind_test_interp]**2), 10*np.log10(bias[ind_test_interp]**2), alpha=1.0, s=10, c="red", label="Test Interpolation Points")
    plt.scatter(10*np.log10(var[ind_test_extrap]**2), 10*np.log10(bias[ind_test_extrap]**2), alpha=1.0, s=10, c="blue", label="Test Extrapolation Points")
    plt.scatter(10*np.log10(var[ind_train]**2), 10*np.log10(bias[ind_train]**2), alpha=1.0, s=10, c="green", label="Training Points")
    plt.xlabel("$\sigma_{\hat{y}}^{2}$ (dB)")
    plt.ylabel("$Bias^{2}$ (dB)")
    plt.title("$Bias^{2}$ vs $\sigma_{\hat{y}}^{2}$ (dB): " + f"{desc}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "bias_sq_dB_vs_var_dB_scatter")
    if block:
        plt.show()
    plt.close()

def plot_bias_sq_var_2x2(var, bias, ind_test, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=None, block=False):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    titles = ["Test Points", "Training Points", "Test Interpolation Points", "Test Extrapolation Points"]
    indices = [ind_test, ind_train, ind_test_interp, ind_test_extrap]
    colors = ["red", "green", "blue", "orange"]

    for ax, title, idx, color in zip(axes.flat, titles, indices, colors):
        ax.scatter(10 * np.log10(var[idx]**2), 10 * np.log10(bias[idx]**2),
                   alpha=1.0, s=10, c=color)
        ax.set_title(title)
        ax.grid(True)
    
    fig.suptitle("$Bias^{2}$ vs $\sigma_{\hat{y}}^{2}$ (dB): " + f"{desc}", fontsize=16)
    fig.supxlabel("$\sigma_{\hat{y}}^{2}$ (dB)")
    fig.supylabel("$Bias^{2}$ (dB)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    if save_dir:
        save_plot(save_dir, "bias_sq_dB_vs_var_dB_2x2_grid")
    if block:
        plt.show()
    plt.close()


def plot_prediction_histogram_waterfall(preds, x, y, bins=50, desc=None, save_dir=None, block=False):
    """
    Waterfall-style histogram of predictions over x.
    """
    n_x = preds.shape[1]
    hist_matrix = []

    for i in range(n_x):
        hist, _ = np.histogram(preds[:, i], bins=bins, range=(preds.min(), preds.max()), density=True)
        hist_matrix.append(hist)

    hist_matrix = np.array(hist_matrix).T  # shape: [bins, n_x]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(hist_matrix, aspect='auto', origin='lower',
               extent=[x.min(), x.max(), preds.min(), preds.max()],
               cmap='viridis')
    plt.colorbar(label='Density')
    plt.plot(x,y,c="red")
    plt.title(f"Prediction Histogram Waterfall: {desc}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "pred_waterfall_hist")
    if block:
        plt.show()
    plt.close()

def single_task_plots(trainer, preds, x_train, y_train, x_test, y_test, desc, ind_train, region_interp, metric_outputs=None, block=False, save_dir=None, capabilities=set()):
    """
    Single task regression plots.

    Args:
        preds (np.ndarray): (n_samples, n_points, 1) multiple stochastic predictions
        x_train (torch.Tensor): training x values
        y_train (torch.Tensor): training y values
        x_test (torch.Tensor): test x values
        y_test (torch.Tensor): test y values
        desc (str): plot title description
        ind_train (np.ndarray): indices of training points in x_test
        region_interp (tuple): min and max of interpolation region
        metric_outputs (tuple): consist of relevant metrics
        block (bool): whether to block plt.show() for interactive use
        save_dir (str): save directory
        capabilities (set): plotting capabilities
    """
    # Convert all arrays from torch tensors to numpy arrays
    x_train_np = x_train.cpu().numpy().squeeze().astype(np.float64)
    y_train_np = y_train.cpu().numpy().squeeze().astype(np.float64)
    x_test_np = x_test.cpu().numpy().squeeze().astype(np.float64)
    y_test_np = y_test.cpu().numpy().squeeze().astype(np.float64)
    preds_np = preds.astype(np.float64)
    x_c_min = x_train_np.min()
    x_c_max = x_train_np.max()

    # Get indice of test interpolation and extrapolation region
    N_test = x_test_np.shape[0]
    x_min = region_interp[0]
    x_max = region_interp[1]
    ind_interp = np.array([n for n in range(N_test) if x_test_np[n] > x_min and x_test_np[n] < x_max])
    ind_test = np.array([n for n in range(N_test) if x_test_np[n] not in x_train_np])
    ind_test_interp = np.array([n for n in ind_interp if n not in ind_train])
    ind_test_extrap = np.array([n for n in range(N_test) if n not in ind_test_interp and n not in ind_train])

    # Compute metrics if they're not already computed
    if metric_outputs is None:
        mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nll = metrics(preds_np, y_test_np)
    else:
        mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nll = metric_outputs

    plot_loss_curve(trainer.losses, trainer.mses, trainer.kls, trainer.betas, desc=desc, 
                save_dir=save_dir, block=False)

    if "residuals" in capabilities:
        plot_residual_scatter(x_test_np, res_prec, res_acc, bias, x_c_min, x_c_max, desc, save_dir=save_dir, block=block)
    if "mean" in capabilities:
        plot_mean_prediction(x_test_np, y_test_np, mean, std, preds, x_train_np, y_train_np, x_c_min, x_c_max, desc, save_dir=save_dir, block=block, zoom=False)
        plot_mean_prediction(x_test_np, y_test_np, mean, std, preds, x_train_np, y_train_np, x_c_min, x_c_max, desc, save_dir=save_dir, block=block, zoom=True)
    if "variance" in capabilities:
        plot_variance(x_test_np, var, mean, ind_train, x_c_min, x_c_max, desc, save_dir=save_dir, block=block)

        plot_prediction_histogram_waterfall(preds, x_test_np, y_test_np, bins=50, desc=desc, save_dir=save_dir, block=block)

        plot_mse_bias_sq_scatter(bias, mse, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=save_dir, block=block)
        plot_mse_bias_sq_scatter_2x2(bias, mse, ind_test, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=save_dir, block=block)

        plot_mse_var(var, mse, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=save_dir, block=block)
        plot_mse_var_2x2(var, mse, ind_test, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=save_dir, block=block)

        plot_bias_sq_var(var, bias, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=save_dir, block=block)
        plot_bias_sq_var_2x2(var, bias, ind_test, ind_test_interp, ind_test_extrap, ind_train, desc, save_dir=save_dir, block=block)
    if "bias" in capabilities:
        plot_bias_mse(x_test_np, bias, mse, ind_train, x_c_min, x_c_max, desc, save_dir=save_dir, block=block)
    if "nll" in capabilities:
        plot_nll(x_test_np, nll, ind_train, x_c_min, x_c_max, desc, save_dir=save_dir, block=block)

def overlay_plot_metrics(metric_dicts, x, metric_name="mse", title=None, save_path=None, log_scale=False, block=False):
    """
    Overlay a metric (e.g., MSE, bias) from multiple models on the same plot.

    Args:
        metric_dicts (list of tuples): List of (label, metric_array) tuples.
        x (np.ndarray): 1D array for x-axis (e.g., test inputs).
        metric_name (str): Metric being plotted ("mse", "bias", "mean", etc.).
        title (str or None): Title of the plot.
        save_path (str or None): If provided, saves the plot to this path.
        log_scale (bool): Whether to use dB scale (10 * log10).
        block (bool): Whether to block on plt.show().
    """
    plt.figure(figsize=(10, 6))

    for label, values in metric_dicts:
        if log_scale:
            values = 10 * np.log10(values)
        plt.plot(x, values, label=label)

    plt.title(title or f"{metric_name.upper()} Comparison")
    plt.xlabel("x")
    plt.ylabel(f"{metric_name} (dB)" if log_scale else metric_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    if block:
        plt.show()

    plt.close()
