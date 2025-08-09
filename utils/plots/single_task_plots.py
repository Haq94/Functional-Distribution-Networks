import os
import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import metrics
from utils.saver.general_saver import save_plot

# Single Plots=========================================================================================

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

def plot_residual_scatter(x, res_prec, res_acc, bias, interp_min, interp_max, save_dir=None, block=False):
    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ii in range(res_prec.shape[1]):
        axs[0].scatter(x, res_prec[:, ii], alpha=0.05, color=np.random.rand(3))
    axs[0].axvline(x=interp_min, color='red', linestyle='--')
    axs[0].axvline(x=interp_max, color='red', linestyle='--')
    axs[0].set_title(f"Residual Precision Task")
    axs[0].grid(True)

    for ii in range(res_acc.shape[1]):
        axs[1].scatter(x, res_acc[:, ii], alpha=0.05, color=np.random.rand(3))
    axs[1].plot(x, bias, label='Bias', color='red')
    axs[1].axvline(x=interp_min, color='red', linestyle='--')
    axs[1].axvline(x=interp_max, color='red', linestyle='--')
    axs[1].set_title(f"Residual Accuracy Task")
    axs[1].legend()
    axs[1].set_xlabel("x")
    axs[1].grid(True)
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "residual_scatter")
    if block:
        plt.show()
    plt.close()

def plot_mean_prediction(x_t_np, y_t_np, mean, std, preds, x_c_np, y_c_np, x_c_min, x_c_max, save_dir=None, block=False, zoom=False):
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
    plt.title(r"$\mu$ vs $x$")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    if save_dir:
        save_plot(save_dir, "zoomed_mean_prediction" if zoom else "mean_prediction")
    if block:
        plt.show()
    plt.close()

def plot_variance(x, var, mean, ind_c, interp_min, interp_max, save_dir=None, block=False):
    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(x, mean)
    axs[0].scatter(x[ind_c], mean[ind_c], label='Training Points', color='red')
    axs[0].set_title(r"$\mu$ vs $x$")
    axs[0].axvline(x=interp_min, color='red', linestyle='--')
    axs[0].axvline(x=interp_max, color='red', linestyle='--')
    axs[0].set_ylabel(r"$\mu$")
    axs[0].grid(True)

    axs[1].plot(x, 10 * np.log10(var))
    axs[1].scatter(x[ind_c], 10 * np.log10(var[ind_c]), label='Training Points', color='red')
    axs[1].set_title(r"$\sigma_{\hat{y}}^2$ $(dB)$ vs $x$")
    axs[1].axvline(x=interp_min, color='red', linestyle='--')
    axs[1].axvline(x=interp_max, color='red', linestyle='--')
    axs[1].set_ylabel( r"$\sigma_{\hat{y}}^2$")
    axs[1].grid(True)
    axs[1].set_xlabel("x")
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "mean_and_var_dB")
    if block:
        plt.show()
    plt.close()

def plot_bias_mse(x, bias, mse, ind_c, interp_min, interp_max, save_dir=None, block=False):
    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(x, bias)
    axs[0].scatter(x[ind_c], bias[ind_c], label='Training Points', color='red')
    axs[0].set_title(r"$Bias$ vs $x$")
    axs[0].axvline(x=interp_min, color='red', linestyle='--')
    axs[0].axvline(x=interp_max, color='red', linestyle='--')
    axs[0].grid(True)

    axs[1].plot(x, 10 * np.log10(mse))
    axs[1].scatter(x[ind_c], 10 * np.log10(mse[ind_c]), label='Training Points', color='red')
    axs[1].set_title(r"$MSE$ (dB) vs $x$")
    axs[1].axvline(x=interp_min, color='red', linestyle='--')
    axs[1].axvline(x=interp_max, color='red', linestyle='--')
    axs[1].grid(True)
    axs[1].set_xlabel("x")
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "bias_and_mse_dB")
    if block:
        plt.show()
    plt.close()

def plot_nlpd(x, nlpd, ind_c, interp_min, interp_max, save_dir=None, block=False):
    plt.figure(figsize=(10, 6))
    plt.plot(x, nlpd, label="NLPD", color="purple")
    plt.scatter(x[ind_c], nlpd[ind_c], label="Training Points", color="red")
    plt.axvline(x=interp_min, color='red', linestyle='--')
    plt.axvline(x=interp_max, color='red', linestyle='--')
    plt.title(r"$NLPD$ vs $x$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$NLPD$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "nlpd")
    if block:
        plt.show()
    plt.close()

def plot_y_vs_x_2x2(x, y, xlabel, ylabel, fname, ind_interp, ind_extrap, ind_test, ind_train, db_scale=True, save_dir=None, block=False):
    ind_test_interp = np.intersect1d(ind_test, ind_interp)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    titles = ["Test Points", "Training Points", "Test Interpolation Points", "Test Extrapolation Points"]
    indices = [ind_test, ind_train, ind_test_interp, ind_extrap]
    colors = ["red", "green", "blue", "orange"]

    for ax, title, idx, color in zip(axes.flat, titles, indices, colors):
        ax.scatter(x[idx], y[idx],
                   alpha=1.0, s=10, c=color)
        ax.set_title(title)
        ax.grid(True)
    
    fig.suptitle(f"{ylabel} vs {xlabel}", fontsize=16)
    xlabel = xlabel + r" $(dB)$" if db_scale and xlabel != r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$" else xlabel
    ylabel = ylabel + r" $(dB)$" if db_scale and ylabel != r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$" else ylabel
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    if save_dir:
        save_plot(save_dir, fname)
    if block:
        plt.show()
    plt.close()

def plot_y_vs_x(x, y, xlabel, ylabel, fname, ind_interp, ind_extrap, ind_test, ind_train, db_scale, save_dir=None, block=False):
    ind_test_interp = np.intersect1d(ind_test, ind_interp)
    plt.figure(figsize=(10,8))
    plt.scatter(x[ind_test_interp], y[ind_test_interp], alpha=1.0, s=10, c="red", label="Test Interpolation Points")
    plt.scatter(x[ind_extrap], y[ind_extrap], alpha=1.0, s=10, c="blue", label="Test Extrapolation Points")
    plt.scatter(x[ind_train], y[ind_train], alpha=1.0, s=10, c="green", label="Training Points")
    plt.title(f"{ylabel} vs {xlabel}")
    xlabel = xlabel + r" $(dB)$" if db_scale and xlabel != r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$" else xlabel
    ylabel = ylabel + r" $(dB)$" if db_scale and ylabel != r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$" else ylabel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, fname)
    if block:
        plt.show()
    plt.close()

def pdf_and_nll_heatmap(preds, x, y, x_min, x_max, bins=50, save_dir=None, block=False):
    """
    Waterfall-style histogram of pdf and nll
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
    plt.axvline(x=x_min, color='black', linestyle='--')
    plt.axvline(x=x_max, color='black', linestyle='--')
    plt.title(r"$p(y|x)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "pdf_heatmap")
    if block:
        plt.show()
    plt.close()

    nll = -np.log(hist_matrix + 1e-12)
    plt.figure(figsize=(12, 6))
    plt.imshow(nll, aspect='auto', origin='lower',
               extent=[x.min(), x.max(), preds.min(), preds.max()],
               cmap='viridis')
    plt.colorbar(label='Density')
    plt.plot(x,y,c="red")
    plt.axvline(x=x_min, color='black', linestyle='--')
    plt.axvline(x=x_max, color='black', linestyle='--')
    plt.title(r"$- \log\left( p(y \mid x) \right)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "nll_heatmap")
    if block:
        plt.show()
    plt.close()

def single_task_plots(trainer, preds, x_train, y_train, x_test, y_test, desc, ind_train, ind_test, ind_interp, ind_extrap, region_interp, metric_outputs=None, block=False, save_dir=None, capabilities=set()):
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
    x_min = min(region_interp)
    x_max = max(region_interp)

    # Compute metrics if they're not already computed
    if metric_outputs is None:
        mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nlpd = metrics(preds_np, y_test_np)
    else:
        mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nlpd = metric_outputs

    plot_loss_curve(trainer.losses, trainer.mses, trainer.kls, trainer.betas, desc=desc, 
                save_dir=save_dir, block=False)

    if "residuals" in capabilities:
        plot_residual_scatter(x_test_np, res_prec, res_acc, bias, x_min, x_max, save_dir=save_dir, block=block)
    if "mean" in capabilities:
        plot_mean_prediction(x_test_np, y_test_np, mean, std, preds, x_train_np, y_train_np, x_min, x_max, save_dir=save_dir, block=block, zoom=False)
        plot_mean_prediction(x_test_np, y_test_np, mean, std, preds, x_train_np, y_train_np, x_min, x_max, save_dir=save_dir, block=block, zoom=True)
    if "variance" in capabilities:
        plot_variance(x_test_np, var, mean, ind_train, x_min, x_max, save_dir=save_dir, block=block)

        pdf_and_nll_heatmap(preds, x_test_np, y_test_np, x_min, x_max, bins=50, save_dir=save_dir, block=block)

        var_str = r"$\sigma_{\hat{y}}^2$"
        bias_sq_str = r"$Bias^2$"
        bias_str = r"$Bias$"
        mse_str = r"$MSE$"
        nlpd_str = r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$"

        plot_y_vs_x(x=_dB(bias**2), y=_dB(mse), xlabel=bias_sq_str, ylabel=mse_str, fname='mse_db_vs_bias_sq_db', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x(x=_dB(var), y=_dB(mse), xlabel=var_str, ylabel=mse_str, fname='mse_db_vs_var_db', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x(x=_dB(var), y=_dB(bias**2), xlabel=var_str, ylabel=bias_sq_str, fname='bias_sq_db_vs_var_db', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x(x=nlpd, y=_dB(mse), xlabel=nlpd_str, ylabel=mse_str, fname='mse_db_vs_nlpd', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x(x=nlpd, y=_dB(bias**2), xlabel=nlpd_str, ylabel=bias_sq_str, fname='bias_sq_db_vs_nlpd', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x(x=_dB(var), y=nlpd, xlabel=var_str, ylabel=nlpd_str, fname='nlpd_vs_var_db', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        

        plot_y_vs_x_2x2(x=_dB(bias**2), y=_dB(mse), xlabel=bias_sq_str, ylabel=mse_str, fname='mse_db_vs_bias_sq_db_2x2', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x_2x2(x=_dB(var), y=_dB(mse), xlabel=var_str, ylabel=mse_str, fname='mse_db_vs_var_db_2x2', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x_2x2(x=_dB(var), y=_dB(bias**2), xlabel=var_str, ylabel=bias_sq_str, fname='bias_sq_db_vs_var_db_2x2', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x_2x2(x=nlpd, y=_dB(mse), xlabel=nlpd_str, ylabel=mse_str, fname='mse_db_vs_nlpd_2x2', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x_2x2(x=nlpd, y=_dB(bias**2), xlabel=nlpd_str, ylabel=bias_sq_str, fname='bias_sq_db_vs_nlpd_2x2', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        plot_y_vs_x_2x2(x=_dB(var), y=nlpd, xlabel=var_str, ylabel=nlpd_str, fname='nlpd_vs_var_db_2x2', ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=save_dir, block=block)
        
    if "bias" in capabilities:
        plot_bias_mse(x_test_np, bias, mse, ind_train, x_min, x_max, save_dir=save_dir, block=block)
    if "nlpd" in capabilities:
        plot_nlpd(x_test_np, nlpd, ind_train, x_min, x_max, save_dir=save_dir, block=block)


# Overlay Plots================================================================================================================

def plot_single_task_overlay(
    seed_date_time_list,
    model_types,
    x_train,
    y_train,
    x_test,
    y_test,
    ind_train,
    ind_test,
    ind_interp,
    ind_extrap,
    metrics,
    losses,
    summary,
    stoch_models,
    stoch_metrics,
    model_colors,
    save_dir=None,
    show_figs=True,
    use_db_scale=True
):
    
    for seed_date_time in seed_date_time_list:
        x = x_test[seed_date_time]
        y = y_test[seed_date_time]
        x0 = x_train[seed_date_time]
        y0 = y_train[seed_date_time]
        
        region_interp = summary['region_interp'][seed_date_time]
        region_interp = region_interp[next(iter(region_interp))]
        
        ind_train0 = ind_train[seed_date_time]
        ind_test0 = ind_test[seed_date_time]
        ind_interp0 = ind_interp[seed_date_time]
        ind_extrap0 = ind_extrap[seed_date_time]
        ind_test_interp0 = np.intersect1d(ind_interp0, ind_test0)
        ind_test_extrap0 = ind_extrap0
        indices_map = {
                    "Test": np.asarray(ind_test0),
                    "Train": np.asarray(ind_train0),
                    "Test Interpolation": np.asarray(ind_test_interp0),
                    "Test Extrapolation": np.asarray(ind_test_extrap0),
                    }

        x_min = np.min(x_train[seed_date_time])
        x_max = np.max(x_train[seed_date_time])

        # --- Metric Plots ---
        for metric_label in {"mean", "var", "std", "bias", "mse", "nlpd"}:
            stoch = metric_label in stoch_metrics
            models = sorted(stoch_models.intersection(model_types)) if stoch else sorted(model_types)

            fig, ax = plt.subplots(figsize=(10, 8))
            metric_min, metric_max = None, None

            for model in models:
                metric = metrics[metric_label][seed_date_time][model]
                if metric_label in {"var", "std", "mse"} and use_db_scale:
                    metric = 10 * np.log10(np.maximum(metric, 1e-10))
                    plot_label = "$\sigma^{2}$ (dB)" if metric_label == "var" else "$\sigma$ (dB)" if metric_label == "std" else "$MSE$ (dB)"
                else:
                    plot_label = "$\mu$" if metric_label == "mean" else "$Bias$" if metric_label == "bias" else metric_label

                ax.plot(x, metric, label=model, color=model_colors.get(model, None))
                metric_min = np.min(metric) if metric_min is None else min(metric_min, np.min(metric))
                metric_max = np.max(metric) if metric_max is None else max(metric_max, np.max(metric))

            if metric_label == "mean":
                metric_min = min(metric_min, np.min(y))
                metric_max = max(metric_max, np.max(y))
                ax.plot(x, y, label="Truth", linestyle='--', color='black')

            ax.axvline(x=x_min, color='red', linestyle='--', label='Train Region')
            ax.axvline(x=x_max, color='red', linestyle='--')
            ax.set_ylim([metric_min, metric_max])
            ax.set_xlabel("x")
            ax.set_ylabel(plot_label)
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            if save_dir:
                save_plot(os.path.join(save_dir, 'metric_vs_x'), f"{metric_label}_vs_x")
            if show_figs:
                plt.show()
            plt.close()

        # --- Loss Plots ---
        for label in losses:
            fig, ax = plt.subplots(figsize=(10, 8))
            value_max = None

            for model in model_types:
                if (model not in stoch_models and label in {"kls", "losses", "betas"}) or (model == "DeepEnsembleNet" and label in {"kls, betas"}):
                    continue
                

                value = losses[label][seed_date_time][model]
                ax.plot(np.arange(1, len(value)+1), value, label=model, color=model_colors.get(model, None))
                value_max = np.max(value) if value_max is None else max(value_max, np.max(value))

            label_map = {"mses": "$MSE$", "losses": "$Losses$", "kls": "$KLS$", "betas": r"$\beta$"}
            ax.set_ylim([0, value_max])
            ax.set_xlabel("Epochs")
            ax.set_ylabel(label_map.get(label, label))
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            if save_dir:
                save_plot(os.path.join(save_dir, 'loss_vs_epoch'), f"{label}_vs_epoch")
            if show_figs:
                plt.show()
            plt.close()

        # --- Scatter Plots ---
        def scatter_plot(x, y, xlabel, ylabel, fname, model_colors, title=None):
            title = title = f"{ylabel} vs {xlabel}".replace(" (dB)", "")
            fig = plt.figure(figsize=(10, 8))
            for model in sorted(stoch_models.intersection(model_types)):
                plt.scatter(x[model], y[model], s=10, alpha=0.7, label=model, color=model_colors.get(model, None))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if save_dir:
                save_plot(os.path.join(save_dir, 'calibration_scatter_plots'), fname)
            if show_figs:
                plt.show()
            plt.close()

        # Collect data for scatter plots
        var = {m: metrics["var"][seed_date_time][m] for m in stoch_models.intersection(model_types)}
        mse = {m: metrics["mse"][seed_date_time][m] for m in stoch_models.intersection(model_types)}
        bias = {m: metrics["bias"][seed_date_time][m] for m in stoch_models.intersection(model_types)}
        nlpd = {m: metrics["nlpd"][seed_date_time][m] for m in stoch_models.intersection(model_types)}
        # dB Scatter Plots
        var_db = {m: _dB(np.maximum(v, 1e-10)) for m, v in var.items()}
        mse_db = {m: _dB(np.maximum(mv, 1e-10)) for m, mv in mse.items()}
        bias_sq_db = {m: _dB(np.maximum(b**2, 1e-10)) for m, b in bias.items()}

        var_str = r"$\sigma_{\hat{y}}^2$"
        bias_sq_str = r"$Bias^2$"
        bias_str = r"$Bias$"
        mse_str = r"$MSE$"
        nlpd_str = r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$"

        # Scatter Plot (dB)
        scatter_plot(x=var_db, y=mse_db, xlabel=var_str + " (dB)", ylabel=mse_str + " (dB)", fname="mse_db_vs_var_db", model_colors=model_colors)
        scatter_plot(x=bias_sq_db, y=mse_db, xlabel=bias_sq_str + " (dB)", ylabel=mse_str + " (dB)", fname="mse_db_vs_bias_sq_db", model_colors=model_colors)
        scatter_plot(x=var_db, y=bias_sq_db, xlabel=var_str + " (dB)", ylabel=bias_sq_str + " (dB)", fname="bias_sq_db_vs_var_db", model_colors=model_colors)

        
        scatter_plot(x=nlpd, y=mse_db, xlabel=nlpd_str, ylabel=mse_str + " (dB)", fname="mse_db_vs_nlpd", model_colors=model_colors)
        scatter_plot(x=nlpd, y=bias_sq_db, xlabel=nlpd_str, ylabel=bias_sq_str + " (dB)", fname="bias_sq_db_vs_nlpd", model_colors=model_colors)
        scatter_plot(x=var_db, y=nlpd, xlabel=var_str + " (dB)", ylabel=nlpd_str, fname="nlpd_vs_var_db", model_colors=model_colors)
        
        # Scatter Plots
        scatter_plot(x=var, y=mse, xlabel=var_str, ylabel=mse_str, fname="mse_vs_var", model_colors=model_colors)
        scatter_plot(x=bias, y=mse, xlabel=bias_str, ylabel=mse_str, fname="mse_vs_bias", model_colors=model_colors)
        scatter_plot(x=var, y=bias, xlabel=var_str, ylabel=bias_str, fname="bias_vs_var", model_colors=model_colors)

        
        scatter_plot(x=nlpd, y=mse, xlabel=nlpd_str, ylabel=mse_str, fname="mse_vs_nlpd", model_colors=model_colors)
        scatter_plot(x=nlpd, y=bias, xlabel=nlpd_str, ylabel=bias_str, fname="bias_vs_nlpd", model_colors=model_colors)
        scatter_plot(x=var, y=nlpd, xlabel=var_str, ylabel=nlpd_str, fname="nlpd_vs_var", model_colors=model_colors)

        # 2x2 Scatter Plots

        # 2×2: NLPD vs Var (dB)
        overlay_scatter_2x2(
            x_dict_db=var_db,
            y_dict_db=nlpd,
            indices_map=indices_map,
            title=nlpd_str + " vs " + var_str,
            xlabel=var_str + r" $(dB)$",
            ylabel=nlpd_str,
            model_colors=model_colors,
            save_dir=os.path.join(save_dir, 'calibration_scatter_plots') if save_dir else None,
            fname="nlpd_vs_var_db_2x2",
            show_figs=show_figs
        )

        # 2×2: Bias² (dB) vs NLPD
        overlay_scatter_2x2(
            x_dict_db=nlpd,
            y_dict_db=bias_sq_db,
            indices_map=indices_map,
            title=bias_sq_str + " vs " + nlpd_str,
            xlabel=nlpd_str,
            ylabel=bias_sq_str,
            model_colors=model_colors,
            save_dir=os.path.join(save_dir, 'calibration_scatter_plots') if save_dir else None,
            fname="bias_sq_dB_vs_nlpd_2x2",
            show_figs=show_figs
        )

        # 2×2: MSE (dB) vs NLPD
        overlay_scatter_2x2(
            x_dict_db=nlpd,
            y_dict_db=mse_db,
            indices_map=indices_map,
            title=mse_str + " vs " + nlpd_str,
            xlabel=nlpd_str,
            ylabel=mse_str + r" $(dB)$",
            model_colors=model_colors,
            save_dir=os.path.join(save_dir, 'calibration_scatter_plots') if save_dir else None,
            fname="mse_dB_vs_nlpd_2x2",
            show_figs=show_figs
        )

        # 2×2: MSE (dB) vs Var (dB)
        overlay_scatter_2x2(
            x_dict_db=var_db,
            y_dict_db=mse_db,
            indices_map=indices_map,
            title=mse_str + " vs " + var_str,
            xlabel=var_str + r" $(dB)$",
            ylabel=mse_str + r" $(dB)$",
            model_colors=model_colors,
            save_dir=os.path.join(save_dir, 'calibration_scatter_plots') if save_dir else None,
            fname="mse_dB_vs_var_dB_2x2",
            show_figs=show_figs
        )

        # 2×2: MSE (dB) vs Bias² (dB)
        overlay_scatter_2x2(
            x_dict_db=bias_sq_db,
            y_dict_db=mse_db,
            indices_map=indices_map,
            title=mse_str + " vs " + bias_sq_str,
            xlabel=bias_sq_str + r" $(dB)$",
            ylabel=mse_str + r" $(dB)$",
            model_colors=model_colors,
            save_dir=os.path.join(save_dir, 'calibration_scatter_plots') if save_dir else None,
            fname="mse_dB_vs_bias_sq_dB_2x2",
            show_figs=show_figs
        )

        # 2×2: Bias² (dB) vs Var (dB)
        overlay_scatter_2x2(
            x_dict_db=var_db,
            y_dict_db=bias_sq_db,
            indices_map=indices_map,
            title=bias_sq_str + " vs " +  var_str,
            xlabel=var_str + r" $(dB)$",
            ylabel=bias_sq_str + r" $(dB)$",
            model_colors=model_colors,
            save_dir=os.path.join(save_dir, 'calibration_scatter_plots') if save_dir else None,
            fname="bias_sq_dB_vs_var_dB_2x2",
            show_figs=show_figs
        )

def _dB(x):
    EPS = 1e-12
    return 10.0 * np.log10(np.maximum(x, EPS))

def _collect_panel_limits(x_by_panel, y_by_panel):
    xs, ys = [], []
    for k in x_by_panel:
        for arr in x_by_panel[k].values():  # per-model arrays
            xs.append(arr)
        for arr in y_by_panel[k].values():
            ys.append(arr)
    if not xs or not ys:
        return (-40, 10), (-40, 10)
    xcat = np.concatenate(list(xs))
    ycat = np.concatenate(list(ys))
    pad_x = 0.03 * (xcat.max() - xcat.min() + 1e-9)
    pad_y = 0.03 * (ycat.max() - ycat.min() + 1e-9)
    return (xcat.min()-pad_x, xcat.max()+pad_x), (ycat.min()-pad_y, ycat.max()+pad_y)

def overlay_scatter_2x2(
    x_dict_db, y_dict_db,
    indices_map,                    # dict: {"Test": idx, "Train": idx, "Test Interp": idx, "Test Extrap": idx}
    title, xlabel, ylabel,
    model_colors, save_dir=None, fname=None, show_figs=True
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    ax_map = {
        "Test": axes[0,0],
        "Train": axes[0,1],
        "Test Interpolation": axes[1,0],
        "Test Extrapolation": axes[1,1],
    }

    # Precompute per-panel, per-model slices and common limits
    x_by_panel, y_by_panel = {}, {}
    for panel, idx in indices_map.items():
        xd, yd = {}, {}
        for m in sorted(x_dict_db.keys()):
            xd[m] = x_dict_db[m][idx]
            yd[m] = y_dict_db[m][idx]
        x_by_panel[panel] = xd
        y_by_panel[panel] = yd

    xlim, ylim = _collect_panel_limits(x_by_panel, y_by_panel)

    # Plot
    for panel, ax in ax_map.items():
        for m in sorted(x_dict_db.keys()):
            ax.scatter(x_by_panel[panel][m], y_by_panel[panel][m],
                       s=10, alpha=0.7, color=model_colors.get(m, None), label=m)
        ax.set_title(panel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Single legend
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', color=model_colors.get(m, 'k')) 
               for m in sorted(x_dict_db.keys())]
    labels = list(sorted(x_dict_db.keys()))
    axes[0,0].legend(handles, labels, loc='best', fontsize=9)

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    if save_dir and fname:
        save_plot(save_dir, fname)
    if show_figs:
        plt.show()
    plt.close()


#####################################################

# levels are central interval coverages, e.g. [0.5, 0.8, 0.9]
def _levels_to_quantiles(levels):
    # central interval: [ (1-a)/2 , (1+a)/2 ]
    levels = np.asarray(levels, dtype=float)
    q_lo = (1.0 - levels) / 2.0
    q_hi = 1.0 - q_lo
    return q_lo, q_hi

def coverage_curve_from_samples(preds, y, levels=(0.5, 0.8, 0.9, 0.95)):
    """
    preds: (S, N) predictive samples per x
    y    : (N,)   ground truth
    returns:
      nominal: (K,) nominal coverages (levels)
      empirical: (K,) coverage fraction in data
      sharpness: (K,) mean interval width (E[U-L])  -> useful as "sharpness"
      per_point_inside: (K, N) booleans for each x and level (optional downstream)
    """
    S, N = preds.shape
    y = np.asarray(y)
    levels = np.asarray(levels, dtype=float)

    q_lo, q_hi = _levels_to_quantiles(levels)  # (K,), (K,)
    # Compute per-x quantiles for all levels at once
    # result shape (K, N) each for lo/hi
    lo = np.quantile(preds, q_lo[:, None], axis=0)  # (K, N)
    hi = np.quantile(preds, q_hi[:, None], axis=0)  # (K, N)

    # Broadcast y to (K, N)
    y_mat = np.broadcast_to(y[None, :], lo.shape)

    inside = (y_mat >= lo) & (y_mat <= hi)   # (K, N)
    empirical = inside.mean(axis=1)          # (K,)
    sharpness = (hi - lo).mean(axis=1)       # (K,)

    return {
        "nominal": levels,
        "empirical": empirical,
        "sharpness": sharpness,
        "per_point_inside": inside
    }

def pit_from_samples(preds, y, jitter=True):
    """
    Probability Integral Transform values using empirical CDF from samples.
    preds: (S, N)
    y    : (N,)
    returns: (N,) PIT values in [0,1]
    """
    S, N = preds.shape
    y = np.asarray(y)
    # rank / S with optional randomized tie-breaking
    less = (preds <= y[None, :]).sum(axis=0).astype(float)  # (N,)
    if jitter:
        # add small uniform in [0, 1/S] to break ties
        rng = np.random.default_rng()
        less += rng.uniform(0.0, 1.0, size=N)
    pit = less / (S + 1.0)
    # clip for safety
    return np.clip(pit, 0.0, 1.0)

def reliability_points_from_samples(preds, y, levels=(0.5, 0.8, 0.9, 0.95)):
    """
    Convenience wrapper returning (nominal, empirical, sharpness)
    for immediate plotting of a reliability (coverage) diagram.
    """
    out = coverage_curve_from_samples(preds, y, levels)
    return out["nominal"], out["empirical"], out["sharpness"]


if __name__=="__main__":
    levels = [0.5, 0.8, 0.9, 0.95]
    nom, emp, sharp = reliability_points_from_samples(preds, y, levels)  # preds shape (S,N)
    pit_vals = pit_from_samples(preds, y)

    # Plot reliability: y=emp, x=nom  (plus a y=x diagonal)
    # Plot sharpness vs coverage: y=sharp, x=nom
    # Plot PIT histogram: histogram of pit_vals, compare to uniform
