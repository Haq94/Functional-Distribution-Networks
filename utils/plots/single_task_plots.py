import os
import matplotlib.pyplot as plt
import numpy as np
import itertools

from utils.metrics import metrics
from utils.saver.general_saver import save_plot
from utils.loader.single_task_loader import single_task_overlay_loader
from utils.plots.plot_helpers import label_subplots, iclr_figsize, merge_and_sort_by_x, sort_and_track_indices, comb_metric_dict
plt.style.use("utils/plots/iclr.mplstyle")

# Single Plots=========================================================================================

def single_task_plots(trainer, preds, x_train, y_train, x_val, y_val, x_test, y_test,
    region_interp, metrics_train=None, metrics_val=None, metrics_test=None, kl_exist=True, is_stoch=True, block=True, save_dir=None, plot_train=True, plot_val=True, plot_types=list()):
    """
    Single task regression plots.

    Args:
        preds (np.ndarray): (n_samples, n_points, 1) multiple stochastic predictions
        x_train (torch.Tensor): training x values
        y_train (torch.Tensor): training y values
        x_test (torch.Tensor): test x values
        y_test (torch.Tensor): test y values
        ind_train (np.ndarray): indices of training points in x_test
        region_interp (tuple): min and max of interpolation region
        metric_outputs (tuple): consist of relevant metrics
        block (bool): whether to block plt.show() for interactive use
        save_dir (str): save directory
        capabilities (set): plotting capabilities
    """
    def to_numpy(x):
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().squeeze().astype(np.float64)
        return x  # Assume it's already a NumPy array
    
    # Convert all arrays from torch tensors to numpy arrays
    preds_np = to_numpy(preds)
    x_train_np = to_numpy(x_train).squeeze(); y_train_np = to_numpy(y_train).squeeze()
    x_val_np = to_numpy(x_val).squeeze(); y_val_np = to_numpy(y_val).squeeze()
    x_test_np = to_numpy(x_test).squeeze(); y_test_np = to_numpy(y_test).squeeze()
    x_min = min(region_interp); x_max = max(region_interp)

    # Save folders
    calibration_save_path = os.path.join(save_dir,'calibration')
    losses_save_path = os.path.join(save_dir,'losses')
    metric_path = os.path.join(save_dir,'metrics_vs_x')

    # Compute metrics if they're not already computed
    if metrics_test is None:
        metrics_test = metrics(preds_np, y_test_np)
    # Unpack variables
    # for key, value in metric_outputs.items():
    #     exec(f"{key} = metric_outputs['{key}']")
    # mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nlpd_kde, grid_kde, nlpd_hist, grid_hist, pdf_kde, pdf_hist, crps = metric_outputs.values()
    mean      = metrics_test["mean"]
    var       = metrics_test["var"]
    std       = metrics_test["std"]
    res_prec  = metrics_test["res_prec"]
    res_acc   = metrics_test["res_acc"]
    bias      = metrics_test["bias"]
    mse       = metrics_test["mse"]
    nlpd_kde  = metrics_test["nlpd_kde"]
    grid_kde  = metrics_test["grid_kde"]
    nlpd_hist = metrics_test["nlpd_hist"]
    grid_hist = metrics_test["grid_hist"]
    pdf_kde   = metrics_test["pdf_kde"]
    pdf_hist  = metrics_test["pdf_hist"]
    crps      = metrics_test["crps"]

    plot_all = True if len(plot_types) == 0  else False

    # Loss, MSE, KL, and Beta Scheduler Plot
    if 'loss_vs_epoch' in plot_types or plot_all:
        plot_loss_curve(trainer.losses, trainer.mses, trainer.kls, trainer.betas, title=None, kl_exist=kl_exist, 
                save_dir=losses_save_path, block=False)
        
    # Metric vs x Plots
    if 'mean_vs_x' in plot_types or plot_all:
        plot_mean_prediction(x_test_np, y_test_np, mean, std, preds, x_train_np, y_train_np, x_min, x_max, is_stoch=is_stoch, save_dir=metric_path, block=block, zoom=False)

    if ("nlpd_kde_vs_x" in plot_types or plot_all) and is_stoch:
        plot_metric_vs_x(x=x_test_np, metric=nlpd_kde, metric_label=r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$", interp_min=x_min, interp_max=x_max,
         x_train=x_train_np, metric_train=metrics_train['nlpd_kde'], plot_train=plot_train, x_val=x_val_np, metric_val=metrics_val['nlpd_kde'], plot_val=plot_val, save_name='nlpd_kde_vs_x', title=None, save_dir=metric_path, block=block)
        
    if ("nlpd_hist_vs_x" in plot_types or plot_all) and is_stoch:
        plot_metric_vs_x(x=x_test_np, metric=nlpd_hist, metric_label=r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$", interp_min=x_min, interp_max=x_max,
         x_train=x_train_np, metric_train=metrics_train['nlpd_hist'], plot_train=plot_train, x_val=x_val_np, metric_val=metrics_val['nlpd_hist'], plot_val=plot_val, save_name='nlpd_hist_vs_x', title=None, save_dir=metric_path, block=block)


    if ('crps_vs_x' in plot_types or plot_all) and is_stoch:
        plot_metric_vs_x(x=x_test_np, metric=_dB(crps), metric_label=r"$CRPS$ $(dB)$", interp_min=x_min, interp_max=x_max,
         x_train=x_train_np, metric_train=_dB(metrics_train['crps']), plot_train=plot_train, x_val=x_val_np, metric_val=_dB(metrics_val['crps']), plot_val=plot_val, save_name='crps_db_vs_x', title=None, save_dir=metric_path, block=block)


    if ('residulas_vs_x' in plot_types or plot_all) and is_stoch:
        plot_residual_scatter(x_test_np, res_prec, res_acc, bias, x_min, x_max, save_dir=metric_path, block=block)

    if ("mean_var_stacked_vs_x" in plot_types or plot_all) and is_stoch:
        plot_two_panel_metric_vs_x(x=x_test_np, metrics=[mean, _dB(var)], 
        plot_train=plot_train, x_train=x_train, metrics_train = [metrics_train['mean'], _dB(metrics_train['var'])],
        plot_val=plot_val, x_val=x_val, metrics_val = [metrics_val['mean'], _dB(metrics_val['var'])],
        interp_min=x_min, interp_max=x_max, ylabels=[r"$\mu_{\hat{y}}$", r"$\sigma_{\hat{y}}^{2}$ $(dB)$"], 
        save_dir=metric_path, fname='mean_var_db_vs_x', block=block)
    elif not is_stoch:
        plot_metric_vs_x(x=x_test_np, metric=mean, metric_label=r"$\mu_{\hat{y}}$", interp_min=x_min, interp_max=x_max,
         x_train=x_train_np, metric_train=metrics_train['mean'], plot_train=plot_train, x_val=x_val_np, metric_val=metrics_val['mean'], plot_val=plot_val, save_name='mean_vs_x', title=None, save_dir=metric_path, block=block)

    if ("bias_mse_stacked_vs_x" in plot_types or plot_all) and is_stoch:        
        plot_two_panel_metric_vs_x(x=x_test_np, metrics=[_dB(bias**2), _dB(mse)], 
        plot_train=plot_train, x_train=x_train, metrics_train = [_dB(metrics_train['bias']**2), _dB(metrics_train['mse'])],
        plot_val=plot_val, x_val=x_val, metrics_val = [_dB(metrics_val['bias']**2), _dB(metrics_val['mse'])],
        interp_min=x_min, interp_max=x_max, ylabels=[r"$Bias^{2}$ $(dB)$", r"$MSE$ $(dB)$"], 
        save_dir=metric_path, fname="bias_sq_db_mse_db_vs_x", block=block)
    elif not is_stoch:
        plot_metric_vs_x(x=x_test_np, metric=_dB(mse), metric_label=r"$MSE$ $(dB)$", interp_min=x_min, interp_max=x_max,
         x_train=x_train_np, metric_train=_dB(metrics_train['mse']), plot_train=plot_train, x_val=x_val_np, metric_val=_dB(metrics_val['mse']), plot_val=plot_val, save_name='mse_db_vs_x', title=None, save_dir=metric_path, block=block)

    # Calibration Plots
    if is_stoch:
        if 'pit_two_panel' in plot_types or plot_all:
            plot_pit_two_panel(x=x_test_np, y_true=y_test_np, preds=preds_np, interp_region=region_interp, save_dir=calibration_save_path, block=block)

        if 'pdf_kde_heatmap' in plot_types or plot_all:
            metric_heatmap(metric=pdf_kde, grid=grid_kde, x=x_test_np, y=y_test_np, x_min=x_min, x_max=x_max, save_name='pdf_kde_heatmap', save_dir=calibration_save_path, block=block)
        
        if 'nll_kde_heatmap' in plot_types or plot_all:
            metric_heatmap(metric=-np.log(pdf_kde+1e-12), grid=grid_kde, x=x_test_np, y=y_test_np, x_min=x_min, x_max=x_max, save_name='nll_kde_heatmap', save_dir=calibration_save_path, block=block)
        
        if 'pdf_hist_heatmap' in plot_types or plot_all:   
            metric_heatmap(metric=pdf_hist, grid=grid_hist, x=x_test_np, y=y_test_np, x_min=x_min, x_max=x_max, save_name='pdf_hist_heatmap', save_dir=calibration_save_path, block=block)
        
        if 'nll_hist_heatmap' in plot_types or plot_all:
            metric_heatmap(metric=-np.log(pdf_hist+1e-12), grid=grid_hist, x=x_test_np, y=y_test_np, x_min=x_min, x_max=x_max, save_name='nll_hist_heatmap', save_dir=calibration_save_path, block=block) 

        var_str = r"$\sigma_{\hat{y}}^2$"
        bias_sq_str = r"$Bias^2$"
        bias_str = r"$Bias$"
        mse_str = r"$MSE$"
        nlpd_str = r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$"
        crps_str = r"$CRPS$"

        metric_key = ["var_db", "mse_db", "bias_sq_db", "nlpd_kde", "nlpd_hist", "crps_db"]
        db_metrics = ["var_db", "mse_db", "bias_sq_db", "crps_db"]
        metric_dict = {
            "mse_db": (mse_str, _dB(mse)),
            "bias_sq_db": (bias_sq_str, _dB(bias**2)),
            "var_db": (var_str, _dB(var)),
            "nlpd_hist": (nlpd_str, nlpd_hist),
            "nlpd_kde": (nlpd_str, nlpd_kde),
            "crps_db": (crps_str, _dB(crps))
        }

        metric_dict_train = {
            "mse_db": _dB(metrics_train['mse']),
            "bias_sq_db": _dB(metrics_train['bias']**2),
            "var_db": _dB(metrics_train['var']),
            "nlpd_hist": metrics_train['nlpd_hist'],
            "nlpd_kde": metrics_train['nlpd_kde'],
            "crps_db": _dB(metrics_train['crps'])
        }

        metric_dict_val = {
            "mse_db": _dB(metrics_val['mse']),
            "bias_sq_db": _dB(metrics_val['bias']**2),
            "var_db": _dB(metrics_val['var']),
            "nlpd_hist": metrics_val['nlpd_hist'],
            "nlpd_kde": metrics_val['nlpd_kde'],
            "crps_db": _dB(metrics_val['crps'])
        }

        for combo in list(itertools.combinations(metric_key, r=2)):
            key_x = combo[0]
            key_y = combo[1]
            if key_x in {"nlpd_hist", "nlpd_kde"} and key_y in {"nlpd_hist", "nlpd_kde"}:
                continue
            xlabel = metric_dict[key_x][0]
            metric_x = metric_dict[key_x][1]

            ylabel = metric_dict[key_y][0]
            metric_y = metric_dict[key_y][1]

            fname = key_y + "_vs_" + key_x
            fname_2x2 = fname + "_2x2"

            if plot_train:
                metric_x_train = metric_dict_train[key_x]
                metric_y_train = metric_dict_train[key_y]
            else:
                metric_x_train = None
                metric_y_train = None

            if plot_val:
                metric_x_val = metric_dict_val[key_x]
                metric_y_val = metric_dict_val[key_y]
            else:
                metric_x_val = None
                metric_y_val = None

            # plot_types.append(fname)
            # plot_types.append(fname_2x2)

            if fname in plot_types or plot_all:
                plot_y_vs_x(x=metric_x, y=metric_y, grid=x_test_np, xlabel=xlabel, ylabel=ylabel, fname=fname, 
                    db_scale=True, region_interp=region_interp,
                    plot_train=plot_train, x_train=metric_x_train, y_train=metric_y_train, 
                    plot_val=plot_val, grid_val=x_val_np, x_val=metric_x_val, y_val=metric_y_val,  
                    save_dir=calibration_save_path, block=block)

                # plot_y_vs_x(x=metric_x, y=metric_y, xlabel=xlabel, ylabel=ylabel, fname=fname, ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=calibration_save_path, block=block)



            if fname_2x2 in plot_types or plot_all:
                plot_y_vs_x_2x2(
                    x=metric_x, y=metric_y, grid=x_test_np, xlabel=xlabel, ylabel=ylabel,
                    fname=fname_2x2, db_scale=True, region_interp=region_interp,
                    plot_train=plot_train, x_train=metric_x_train, y_train=metric_y_train, 
                    plot_val=plot_val, grid_val=x_val_np, x_val=metric_x_val, y_val=metric_y_val,  
                    save_dir=calibration_save_path, block=block)


                # plot_y_vs_x_2x2(x=metric_x, y=metric_y, xlabel=xlabel, ylabel=ylabel, fname=fname_2x2, ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=calibration_save_path, block=block)
        
    # print(plot_types)





# Single Task Plots: Loss Curve Plots=========================================================================================

def plot_loss_curve(losses, mses, kls, betas, title="Training Loss Curve", kl_exist=True, save_dir=None, block=False):
    """
    Plots total loss, MSE, KL divergence, and beta schedule in subplots.

    Args:
        losses (list): total loss per epoch
        mses (list): mse per epoch
        kls (list): kl divergence per epoch
        betas (list): beta value per epoch
        title (str): overall title for the plot
        save_path (str): full file path to save the figure (e.g., 'results/ModelX/loss_curve.png')
        block (bool): if True, blocks the plot window (interactive mode)
    """

    losses = [l.item() if hasattr(l, 'item') else l for l in losses]
    mses   = [m.item() if hasattr(m, 'item') else m for m in mses]
    kls    = [k.item() if hasattr(k, 'item') else k for k in kls]
    betas  = [b.item() if hasattr(b, 'item') else b for b in betas]

    epochs = range(1, len(losses)+1)

    if kl_exist:
        _, axs = plt.subplots(2, 1, num="Total Loss, MSE, and KL Divergence Plot", figsize=iclr_figsize(layout="stacked"), sharex=True)

        # Top subplot: Loss curves
        axs[0].plot(epochs, losses, label="Total Loss", color="blue")
        axs[0].plot(epochs, mses,   label="MSE", color="green")
        axs[0].plot(epochs, kls,    label="KL Divergence", color="red")
        axs[0].set_ylabel("Loss Value")
        if title is not None:
            axs[0].set_title(title)
        axs[0].grid(True)
        axs[0].legend()

        # Bottom subplot: Beta schedule
        axs[1].plot(epochs, betas, label="Beta", color="black")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("β")
        axs[1].grid(True)
        axs[1].legend()

        label_subplots(axs)

        plt.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=iclr_figsize(layout="single"),
                            num="Total Loss Plot")

        # Single subplot: Loss only
        ax.plot(epochs, losses, label="Total Loss", color="blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        if title is not None:
            ax.set_title(title)
        ax.grid(True)
        ax.legend()

        plt.tight_layout()

    if save_dir:
        save_plot(save_dir, "loss_vs_epoch")
    if block:
        plt.show()
    plt.close()

# Single Task Plots: Metric Plots=========================================================================================

def plot_residual_scatter(x, res_prec, res_acc, bias, interp_min, interp_max, save_dir=None, set_title=False, block=False):

    _, axs = plt.subplots(2, 1, figsize=iclr_figsize(layout="stacked"), sharex=True)
    for ii in range(res_prec.shape[1]):
        axs[0].scatter(x, res_prec[:, ii], alpha=0.05, color=np.random.rand(3))
    axs[0].axvline(x=interp_min, color='red', linestyle='--')
    axs[0].axvline(x=interp_max, color='red', linestyle='--')
    if set_title:
        axs[0].set_title(f"Residual Precision") 
    axs[0].grid(True)

    for ii in range(res_acc.shape[1]):
        axs[1].scatter(x, res_acc[:, ii], alpha=0.05, color=np.random.rand(3))
    axs[1].plot(x, bias, label='Bias', color='red')
    axs[1].axvline(x=interp_min, color='red', linestyle='--')
    axs[1].axvline(x=interp_max, color='red', linestyle='--')
    if set_title:
        axs[1].set_title(f"Residual Accuracy")
    axs[1].legend()
    axs[1].set_xlabel(r"$x$")
    axs[1].grid(True)

    label_subplots(axs)
    plt.tight_layout()
    
    if save_dir:
        save_plot(save_dir, "residual_scatter_vs_x")
    if block:
        plt.show()
    plt.close()

def plot_mean_prediction(x, y, mean, std, preds, x_train, y_train, x_min, x_max, is_stoch=True, title=None, save_dir=None, block=False, zoom=False):
    plt.figure(figsize=iclr_figsize(layout="single"))
    if is_stoch:
        for ii in range(preds.shape[0]):
            plt.scatter(x, preds[ii, :], alpha=0.05, color=np.random.rand(3))

    plt.plot(x, y, label="Ground Truth", linestyle="--")
    plt.plot(x, mean, label=r"$\mu_{\hat{y}}$")

    if is_stoch:
        plt.fill_between(x, mean - std, mean + std, alpha=0.3, label=r"$\pm \sigma_{\hat{y}}$")

    plt.scatter(x_train, y_train, color="red", label="Training Points", alpha=0.5)
    plt.axvline(x=x_min, color='red', linestyle='--')
    plt.axvline(x=x_max, color='red', linestyle='--')
    if zoom:
        plt.ylim([y.min()-1, y.max()+1])
    if title is not None:
        plt.title(r"$\mu$ vs $x$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\mu_{\hat{y}}$")
    plt.legend()
    plt.grid(True)
    if save_dir:
        save_plot(save_dir, "zoomed_mean_vs_x" if zoom else "mean_vs_x")
    if block:
        plt.show()
    plt.close()



def plot_metric_vs_x(x, metric, metric_label, interp_min, interp_max, x_train=None, metric_train=None, plot_train=True, x_val=None, metric_val=None, plot_val=True, save_name=None, title=None, save_dir=None, block=False):
    plt.figure(figsize=iclr_figsize(layout="single"))
    plt.plot(x, metric, color="black")

    if isinstance(x_train, np.ndarray) and isinstance(metric_train, np.ndarray) and plot_train:
        plt.scatter(x_train, metric_train, label="Training Points", color="red", s=10)

    if isinstance(x_val, np.ndarray) and isinstance(metric_val, np.ndarray) and plot_val:
        plt.scatter(x_val, metric_val, label="Validation Points", color="blue", s=10)

    plt.axvline(x=interp_min, color='red', linestyle='--')
    plt.axvline(x=interp_max, color='red', linestyle='--')
    title = metric_label + r" vs $x$" if title is not None else title
    if title is not None:
        plt.title(title)
    plt.xlabel(r"$x$")
    plt.ylabel(metric_label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_name = metric_label if save_name is None else save_name
        save_plot(save_dir, save_name)
    if block:
        plt.show()
    plt.close()  


def plot_two_panel_metric_vs_x(
    x,
    metrics,          # list of arrays to plot, length = 2
    plot_train=True,
    x_train=None,
    metrics_train=None,
    plot_val=True,
    x_val=None,
    metrics_val=None,
    interp_min=None,
    interp_max=None,
    ylabels=None,     # list of y-axis labels, length = 2
    titles=None,      # list of subplot titles (optional)
    save_dir=None,
    fname="two_panel_plot",
    block=False
):
    """
    Generic two-panel plotter.

    Args:
        x (array): x-axis values
        metrics (list of arrays): [metric1, metric2], each of length len(x)
        ind_c (array or None): training indices for scatter overlay
        interp_min, interp_max (float or None): vertical lines for interpolation region
        ylabels (list of str): y-axis labels for the two panels
        titles (list of str): optional subplot titles
        save_dir (str or None): directory to save figure
        fname (str): filename stem
        block (bool): if True, blocks plt.show()
    """
    assert len(metrics) == 2, "Must provide exactly two metrics"
    if ylabels is None:
        ylabels = ["Metric 1", "Metric 2"]

    _, axs = plt.subplots(2, 1, figsize=iclr_figsize(layout="stacked"), sharex=True)

    for i, ax in enumerate(axs):
        y = metrics[i]
        ax.plot(x, y, color="black", lw=1.5)

        if isinstance(x_train, np.ndarray) and isinstance(metrics_train, list) and plot_train:
            y_train = metrics_train[i]
            ax.scatter(x_train, y_train, label="Training Points",
                       color="red", s=10)
            
        if isinstance(x_val, np.ndarray) and isinstance(metrics_val, list) and plot_val:
            y_val = metrics_val[i]
            ax.scatter(x_val, y_val, label="Validation Points",
                       color="blue", s=10)

        if interp_min is not None and interp_max is not None:
            ax.axvline(x=interp_min, color="red", linestyle="--")
            ax.axvline(x=interp_max, color="red", linestyle="--")

        if titles is not None:
            ax.set_title(titles[i])

        ax.set_ylabel(ylabels[i])
        ax.grid(True)

    axs[-1].set_xlabel(r"$x$")
    axs[-1].legend()

    label_subplots(axs)
    plt.tight_layout()

    if save_dir:
        save_plot(save_dir, fname)
    if block:
        plt.show()
    plt.close()


# Single Task Plots: Calibration Plots===============================================================================================

def metric_heatmap(metric, grid, x, y, x_min, x_max, save_name, title=None, save_dir=None, block=False):
    """
    Metric heatmap (used for the pdf and nll heatmap plots)
    """

    metric = metric.T
    y_plt = y[(y<=grid.max()) & (y >= grid.min())]
    x_plt = x[(y<=grid.max()) & (y >= grid.min())]
    
    plt.figure(figsize=iclr_figsize(layout="single"))
    plt.imshow(metric, aspect='auto', origin='lower',
               extent=[x.min(), x.max(), grid.min(), grid.max()],
               cmap='viridis')
    plt.colorbar()
    plt.plot(x_plt,y_plt,c="red")
    plt.axvline(x=x_min, color='black', linestyle='--')
    plt.axvline(x=x_max, color='black', linestyle='--')
    if title is not None:
        plt.title(title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, save_name)
    if block:
        plt.show()
    plt.close() 

def plot_pit_two_panel(
    x,
    y_true,
    preds,                      # (S,N) or (S,N,1) over TEST points only
    interp_region,              # (xmin, xmax)
    randomized=True,
    trend=True,                 # draw running median + 10–90% band
    trend_window=None,          # None -> auto
    save_dir=None,
    fname="pit_two_panel",
    block=False,
    set_title=None              # optional: str for both or (str,str) per panel
):
    """
    Two-panel PIT vs x using TEST points only.
      Left  = test interpolation (x in [xmin, xmax])
      Right = test extrapolation (x outside [xmin, xmax])
    Returns dict with summary stats for each panel.
    """
    x = np.squeeze(np.asarray(x))
    y_true = np.squeeze(np.asarray(y_true))
    N = x.shape[0]
    if y_true.shape[0] != N:
        raise ValueError("x and y_true must have the same length for test points.")

    xmin, xmax = float(interp_region[0]), float(interp_region[1])
    mask_interp = (x >= xmin) & (x <= xmax)
    idx_interp  = np.where(mask_interp)[0]
    idx_extrap  = np.where(~mask_interp)[0]

    # PIT for all (test) points
    u_all = _compute_pit_from_mc(y_true, preds, randomized=randomized)
    stats = {}

    def sorted_subset(idx):
        if len(idx) == 0:
            return np.array([]), np.array([]), np.array([], dtype=int)
        order = np.argsort(x[idx])
        return x[idx][order], u_all[idx][order], idx[order]

    panels = [
        ("PIT vs x — Test Interpolation", idx_interp,  "red"),
        ("PIT vs x — Test Extrapolation", idx_extrap,  "blue"),
    ]

    # Allow custom titles
    if isinstance(set_title, (list, tuple)) and len(set_title) == 2:
        panels = [
            (set_title[0], idx_interp, "red"),
            (set_title[1], idx_extrap, "blue"),
        ]
    elif isinstance(set_title, str):
        panels = [
            (set_title, idx_interp, "red"),
            (set_title, idx_extrap, "blue"),
        ]

    fig, axes = plt.subplots(1, 2, figsize=iclr_figsize(layout="double"))
    for k, (title, idx, color) in enumerate(panels):
        ax = axes[k]

        xs, us, _ = sorted_subset(idx)

        # refs
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.9)
        ax.axhline(0.1, color='gray', linestyle=':',  linewidth=1, alpha=0.6)
        ax.axhline(0.9, color='gray', linestyle=':',  linewidth=1, alpha=0.6)

        # shade interpolation region
        ax.axvspan(xmin, xmax, color='C1', alpha=0.08, lw=0)

        # scatter (test only)
        ax.scatter(xs, us, s=14, alpha=0.85, c=color,
                   label=f"{'Interpolation' if k==0 else 'Extrapolation'} (N={len(us)})")

        # trend
        if trend and len(us) >= 5:
            q50, q10, q90 = _rolling_quantiles(xs, us, window=trend_window)
            ax.plot(xs, q50, lw=2, color=color, alpha=0.9, label="running median")
            ax.fill_between(xs, q10, q90, color=color, alpha=0.15, linewidth=0,
                            label="10–90% band")
        # axes & labels
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$x$")
        if k == 0:
            ax.set_ylabel(r"$u = \hat F_x(y)$")
        # ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        # summary stats
        if len(us) > 0:
            med  = float(np.median(us))
            q10g = float(np.quantile(us, 0.10))
            q90g = float(np.quantile(us, 0.90))
            pval, ks = _uniformity_p(us)
            stats_key = "interp" if k == 0 else "extrap"
            stats[stats_key] = {
                "N": int(len(us)),
                "median": med,
                "median_delta": med - 0.5,
                "q10": q10g,
                "q90": q90g,
                "band_10_90": q90g - q10g,   # ideal ~0.8
                "ks_pvalue": pval,
                "ks_stat": ks,
            }
        else:
            stats["interp" if k == 0 else "extrap"] = {
                "N": 0, "median": np.nan, "median_delta": np.nan,
                "q10": np.nan, "q90": np.nan, "band_10_90": np.nan,
                "ks_pvalue": None, "ks_stat": None
            }

    label_subplots(axes)
    plt.tight_layout()

    if save_dir:
        save_plot(save_dir, fname)
    if block:
        plt.show()
    plt.close()


def plot_y_vs_x_2x2(
    x, y, grid, xlabel, ylabel, fname, db_scale, region_interp,
    plot_train=True, x_train=None, y_train=None, 
    plot_val=True, grid_val=None, x_val=None, y_val=None,  
    set_title=None, save_dir=None, block=False
):
    """
    2x2 panels (metrics-vs-metrics):
      [0,0] Test — Interpolation (points where grid in region)
      [0,1] Test — Extrapolation (points where grid outside region)
      [1,0] Training (if provided)
      [1,1] Validation (if provided)

    x, y       : metric arrays for TEST points (same length as grid)
    grid       : input coordinates for TEST, used ONLY to split interp/extrap
    region_interp = (xmin, xmax)
    x_train,y_train : metric arrays for TRAIN (plotted as-is)
    x_val,y_val     : metric arrays for VAL (plotted as-is)
    """
    # --- inputs & split by region (using grid only) ---
    x = np.squeeze(np.asarray(x))
    y = np.squeeze(np.asarray(y))
    grid = np.squeeze(np.asarray(grid))
    if x.shape[0] != y.shape[0] or x.shape[0] != grid.shape[0]:
        raise ValueError("x, y, and grid must have the same length for TEST points.")

    xmin, xmax = float(min(region_interp)), float(max(region_interp))
    mask_interp = (grid >= xmin) & (grid <= xmax)
    mask_extrap = ~mask_interp

    # panel data: metrics-vs-metrics; test is split via grid
    panels = [
        ("Test — Interpolation", x[mask_interp], y[mask_interp], "tab:blue"),
        ("Test — Extrapolation", x[mask_extrap], y[mask_extrap], "tab:orange"),
        ("Training", (None if not plot_train or x_train is None or y_train is None 
                      else np.squeeze(np.asarray(x_train))),
                     (None if not plot_train or x_train is None or y_train is None 
                      else np.squeeze(np.asarray(y_train))),
                     "tab:green"),
        ("Validation", (None if not plot_val or x_val is None or y_val is None 
                        else np.squeeze(np.asarray(x_val))),
                       (None if not plot_val or x_val is None or y_val is None 
                        else np.squeeze(np.asarray(y_val))),
                       "tab:red"),
    ]

    # global limits from available metric data
    xs_for_lim, ys_for_lim = [], []
    for _, xi, yi, _ in panels:
        if xi is not None and yi is not None and len(xi) > 0:
            xs_for_lim.append(xi); ys_for_lim.append(yi)
    if len(xs_for_lim) == 0:
        raise ValueError("No points to plot. Check your inputs/masks.")

    x_min, x_max = float(np.min(np.concatenate(xs_for_lim))), float(np.max(np.concatenate(xs_for_lim)))
    y_min, y_max = float(np.min(np.concatenate(ys_for_lim))), float(np.max(np.concatenate(ys_for_lim)))
    if x_min == x_max: x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max: y_min, y_max = y_min - 1.0, y_max + 1.0

    # --- plotting ---
    fig, axes = plt.subplots(2, 2, figsize=iclr_figsize(layout="2x2"), sharex=True, sharey=True)

    for ax, (title, xi, yi, color) in zip(axes.flat, panels):
        if xi is not None and yi is not None and len(xi) > 0:
            ax.scatter(xi, yi, s=10, alpha=0.5, c=color)
            nlab = f" (N={len(xi)})"
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes, alpha=0.7)
            nlab = " (N=0)"
        if set_title != None:
            ax.set_title(title + nlab)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # labels & suptitle
    xlab = xlabel + r" $(dB)$" if db_scale and xlabel != r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$" else xlabel
    ylab = ylabel + r" $(dB)$" if db_scale and ylabel != r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$" else ylabel
    fig.supxlabel(xlab)
    fig.supylabel(ylab)

    label_subplots(axes)
    if set_title != None:
        fig.suptitle(f"{ylabel} vs {xlabel}", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir:
        save_plot(save_dir, fname)
    if block:
        plt.show()
    plt.close()



def plot_y_vs_x(x, y, grid, xlabel, ylabel, fname, db_scale, region_interp,
    plot_train=True, x_train=None, y_train=None, 
    plot_val=True, grid_val=None, x_val=None, y_val=None,  
    title=None, save_dir=None, block=False):

    x_min = min(region_interp)
    x_max = max(region_interp)

    ind_test_interp = np.where((grid >= x_min) & (grid <= x_max))[0]
    ind_test_extrap = np.array([n for n in range(grid.shape[0]) if n not in ind_test_interp])

    plt.figure(figsize=iclr_figsize(layout="single"))

    plt.scatter(x[ind_test_interp], y[ind_test_interp], alpha=0.5, s=10, c="red", label="Test Interpolation Points")
    plt.scatter(x[ind_test_extrap], y[ind_test_extrap], alpha=0.5, s=10, c="blue", label="Test Extrapolation Points")

    if isinstance(x_train, np.ndarray) and isinstance(y_train, np.ndarray) and plot_train:
        plt.scatter(x_train, y_train, alpha=0.5, s=10, c="green", label="Training Points")

    if isinstance(x_val, np.ndarray) and isinstance(y_val, np.ndarray) and plot_val:
        ind_val_interp = np.where((grid_val >= x_min) & (grid_val <= x_max))[0]
        ind_val_extrap = np.array([n for n in range(grid_val.shape[0]) if n not in ind_val_interp])
        plt.scatter(x_val[ind_val_interp], y_val[ind_val_interp], alpha=0.5, s=10, c="orange", label="Validation Interpolation Points")
        plt.scatter(x_val[ind_val_extrap], y_val[ind_val_extrap], alpha=0.5, s=10, c="yellow", label="Validation Extrapolaion Points")

    if title != None:
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







# Overlay Plots================================================================================================================

def plot_single_task_overlay(
    path,
    region_interp,
    model_types,
    stoch_models,
    stoch_metrics,
    model_colors,
    save_path=None,
    show_figs=False,
    use_db_scale=True,
    plot_types=[],
    set_title=False
    ):
    # Load path
    load_path = path if isinstance(path, list) else [path]
    # Save path
    save_path = os.path.join(path, 'overlay_plots')
    # Re-load data
    # loaders, metrics, losses, summary, x_train, y_train, x_test, y_test, ind_train, ind_test, ind_interp, ind_extrap, save_paths = single_task_overlay_loader([load_path])
    loaders, metrics_test, metrics_train, metrics_val, losses, summary, x_train, y_train, x_val, y_val, x_test, y_test, _, _, save_paths = single_task_overlay_loader(load_path)
    
    # If plot_types is empty plot everything
    plot_all = True if len(plot_types) == 0 else False

    for save_path in save_paths:
        x_test = x_test[save_path]
        y_test = y_test[save_path]
        x_train = x_train[save_path]
        y_train = y_train[save_path]
        x_val = x_val[save_path]
        y_val = y_val[save_path]

        # Sort the input training, validation, and test input points
        x, index_maps, sort_idx = sort_and_track_indices(x_train, x_val, x_test)
        
        # Train, validation, and test indices
        ind_train = index_maps[0]
        ind_val = index_maps[1]
        ind_test = index_maps[2]

        # Interpolation and extrapolation indices
        ind_interp = np.where((x >= min(region_interp)) & (x <= max(region_interp)))[0]
        ind_extrap = [n for n in range(x.shape[0]) if n not in ind_interp]

        ind_test_interp = np.intersect1d(ind_interp, ind_test)
        ind_test_extrap = np.setdiff1d(ind_test, ind_test_interp)

        ind_val_interp = np.intersect1d(ind_interp, ind_val)
        ind_val_extrap = np.setdiff1d(ind_val, ind_val_interp)

        # Concatenate all the metrics in each dictionary then sort them according to input value
        metric_dict = comb_metric_dict(metrics_train[save_path], metrics_val[save_path], metrics_test[save_path], sort_idx)

        indices_map = {
                    "Test": np.asarray(ind_test),
                    "Train": np.asarray(ind_train),
                    "Validation": np.asarray(ind_val),
                    "Validation Interpolation": np.asarray(ind_val_interp),
                    "Validation Extrapolation": np.asarray(ind_val_extrap),
                    "Test Interpolation": np.asarray(ind_test_interp),
                    "Test Extrapolation": np.asarray(ind_test_extrap),
                    "Interpolation": np.asarray(ind_interp),
                    "Extrapolation": np.asarray(ind_extrap)
                    }

        x_min = min(region_interp)
        x_max = max(region_interp)

        # --- Metric Plots ---
        for metric_label in {"mean", "var", "bias", "mse", "nlpd_kde", "nlpd_hist", "crps"}:
            if f"{metric_label}_vs_x" not in plot_types and plot_all == False:
                continue
            stoch = metric_label in stoch_metrics
            models = sorted(stoch_models.intersection(model_types)) if stoch else sorted(model_types)

            fig, ax = plt.subplots(figsize=iclr_figsize(layout="single"))
            metric_min, metric_max = None, None

            for model in models:
                metric = metrics_test[save_path][model][metric_label]
                if metric_label in {"var", "mse", "crps"} and use_db_scale:
                    metric = _dB(metric)
                    plot_label = "$\sigma_{\hat{y}}^2$ (dB)" if metric_label == "var" else r"$MSE$ $(dB)$" if metric_label == "mse" else r"$CRPS$ $(dB)$"
                elif metric_label in {"nlpd_kde", "nlpd_hist"}:
                    plot_label = r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$"
                else:
                    plot_label = r"$\mu_{\hat{y}}$" if metric_label == "mean" else "$Bias$" if metric_label == "bias" else metric_label
                
                ax.plot(x_test, metric, label=model.replace('_','-'), color=model_colors.get(model, None))
                metric_min = np.min(metric) if metric_min is None else min(metric_min, np.min(metric))
                metric_max = np.max(metric) if metric_max is None else max(metric_max, np.max(metric))

            if metric_label == "mean":
                metric_min = min(metric_min, np.min(y_test))
                metric_max = max(metric_max, np.max(y_test))
                ax.plot(x_test, y_test, label="Truth", linestyle='--', color='black')

            ax.axvline(x=x_min, color='red', linestyle='--', label='Train Region')
            ax.axvline(x=x_max, color='red', linestyle='--')
            ax.set_ylim([metric_min, metric_max])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(plot_label)
            if set_title:
                ax.set_title(plot_label + r" vs $x$")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            if save_path:
                save_plot(os.path.join(save_path, 'metric_vs_x'), f"{metric_label}_vs_x")
            if show_figs:
                plt.show()
            plt.close()

        # --- Loss Plots ---
        label_map = {"mses": r"$\mathrm{MSE}$",
                    "losses": r"$\mathcal{L}$",
                    "kls": r"$\mathrm{KL}$",
                    "betas": r"$\beta$"}

        # Collect the union of labels across models for this run
        all_labels = set()
        for model, mdict in losses[save_path].items():
            all_labels.update(mdict.keys())

        for label in sorted(all_labels):
            if f"{label}_vs_epoch" not in plot_types and plot_all == False:
                continue
            fig, ax = plt.subplots(figsize=iclr_figsize(layout="single"))

            plotted_any = False
            y_min, y_max = None, None

            for model, mdict in losses[save_path].items():
                # Skip labels that don't apply to deterministic models
                if (model not in stoch_models and label in {"kls", "betas"}) \
                or (model == "DeepEnsembleNet" and label in {"kls", "betas"}):
                    continue

                if label not in mdict:
                    continue

                value = np.asarray(mdict[label], dtype=float)
                if value.ndim == 0:
                    value = value.reshape(1)

                if not np.isfinite(value).any():
                    continue

                epochs = np.arange(1, len(value) + 1)
                ax.plot(epochs, value, label=model.replace('_','-'), linewidth=1.6,
                        color=model_colors.get(model, None))

                vmin, vmax = float(np.nanmin(value)), float(np.nanmax(value))
                y_min = vmin if y_min is None else min(y_min, vmin)
                y_max = vmax if y_max is None else max(y_max, vmax)
                plotted_any = True

            if not plotted_any:
                plt.close(fig)
                continue

            # Nice padding on y-lims
            span = (y_max - y_min) if y_max > y_min else max(1.0, abs(y_max))
            pad = 0.05 * span
            ax.set_ylim(y_min - pad, y_max + pad)

            ax.set_xlabel(r"$Epoch$")
            ax.set_ylabel(label_map.get(label, label))
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)
            fig.tight_layout()

            if save_path:
                out_dir = os.path.join(save_path, "loss_vs_epoch")
                os.makedirs(out_dir, exist_ok=True)
                save_plot(out_dir, f"{label}_vs_epoch")
            if show_figs:
                plt.show()
            plt.close()

        # --- Scatter Plots ---
        def scatter_plot(x, y, xlabel, ylabel, fname, model_colors, title=None, set_title=False):
            title = title = f"{ylabel} vs {xlabel}".replace(" (dB)", "")
            fig = plt.figure(figsize=iclr_figsize(layout="single"))
            for model in sorted(stoch_models.intersection(model_types)):
                plt.scatter(x[model], y[model], s=10, alpha=0.5, label=model.replace('_','-'), color=model_colors.get(model, None))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if set_title:
                plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if save_path:
                save_plot(os.path.join(save_path, 'calibration_scatter_plots'), fname)
            if show_figs:
                plt.show()
            plt.close()


        # Collect data for scatter plots
        var = {m: metric_dict[m]["var"] for m in stoch_models.intersection(model_types)}
        mse = {m: metric_dict[m]["mse"] for m in stoch_models.intersection(model_types)}
        bias = {m: metric_dict[m]["bias"] for m in stoch_models.intersection(model_types)}
        nlpd_kde = {m: metric_dict[m]["nlpd_kde"] for m in stoch_models.intersection(model_types)}
        nlpd_hist = {m: metric_dict[m]["nlpd_hist"] for m in stoch_models.intersection(model_types)}
        crps = {m: metric_dict[m]["crps"] for m in stoch_models.intersection(model_types)}
        # dB Scatter Plots
        var_db = {m: _dB(v) for m, v in var.items()}
        mse_db = {m: _dB(mv) for m, mv in mse.items()}
        bias_sq_db = {m: _dB(b**2) for m, b in bias.items()}
        crps_db = {m: _dB(c) for m, c in crps.items()}

        var_str = r"$\sigma_{\hat{y}}^2$"
        bias_sq_str = r"$Bias^2$"
        bias_str = r"$Bias$"
        mse_str = r"$MSE$"
        nlpd_str = r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$"
        crps_str = r"$CRPS$"

        metric_key = ["var_db", "mse_db", "bias_sq_db", "nlpd_kde", "nlpd_hist", "crps_db"]
        db_metrics = ["var_db", "mse_db", "bias_sq_db", "crps_db"]
        metric_dict = {
            "mse_db": (mse_str, mse_db),
            "bias_sq_db": (bias_sq_str, bias_sq_db),
            "var_db": (var_str, var_db),
            "nlpd_hist": (nlpd_str, nlpd_hist),
            "nlpd_kde": (nlpd_str, nlpd_kde),
            "crps_db": (crps_str, crps_db)
        }
        for combo in list(itertools.combinations(metric_key, r=2)):
            key_x = combo[0]
            key_y = combo[1]
            if key_x in {"nlpd_hist", "nlpd_kde"} and key_y in {"nlpd_hist", "nlpd_kde"}:
                continue
            xlabel = metric_dict[key_x][0]
            metric_x = metric_dict[key_x][1]

            ylabel = metric_dict[key_y][0]
            metric_y = metric_dict[key_y][1]
            
            title = ylabel + " vs " + xlabel
            xlabel = xlabel + r" $(dB)$" if key_x in db_metrics else xlabel 
            ylabel = ylabel + r" $(dB)$" if key_y in db_metrics else ylabel 

            fname = key_y + "_vs_" + key_x
            fname_2x2 = fname + "_2x2"

            if fname_2x2 in plot_types or plot_all == True:
                overlay_scatter_2x2(
                    x_dict_db=metric_x,
                    y_dict_db=metric_y,
                    indices_map=indices_map,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    model_colors=model_colors,
                    save_dir=os.path.join(save_path, 'calibration_scatter_plots') if save_path else None,
                    fname=fname_2x2,
                    show_figs=show_figs
                )

            if fname in plot_types or plot_all == True:
                metric_test_x = {m: metric_x[m][ind_test] for m in metric_x.keys()}
                metric_test_y = {m: metric_y[m][ind_test] for m in metric_y.keys()}

                scatter_plot(x=metric_test_x, y=metric_test_y, xlabel=xlabel, ylabel=ylabel, fname=fname, model_colors=model_colors)

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
    model_colors, save_dir=None, fname=None, show_figs=True, set_title=None 
    ):
    fig, axes = plt.subplots(2, 2, figsize=iclr_figsize(layout="2x2"), sharex=True, sharey=True)
    ax_map = {
        "Test": axes[0,0],
        "Train": axes[0,1],
        "Test Interpolation": axes[1,0],
        "Test Extrapolation": axes[1,1],
    }

    # ax_map = {
    # "Test": axes[0],
    # "Train": axes[1],
    # "Validation": axes[2],
    # }

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
                       s=10, alpha=0.5, color=model_colors.get(m, None), label=m)
        if set_title != None:
            ax.set_title(panel)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Single legend
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', color=model_colors.get(m, 'k')) 
               for m in sorted(x_dict_db.keys())]
    labels = list(sorted(x_dict_db.keys()))
    labels = [lb.replace('_','-') for lb in labels]
    axes.flat[0].legend(handles, labels, loc='best', fontsize=9)

    if set_title != None:
        fig.suptitle(title, fontsize=14, y=0.98)
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    label_subplots(axes)
    fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    if save_dir and fname:
        save_plot(save_dir, fname)
    if show_figs:
        plt.show()
    plt.close()


#####################################################

def plot_coverage_two_panel(
    preds_by_model,   # dict: {model_name: preds (S,N) or (S,N,1)}
    y_true,           # (N,)
    ind_interp, ind_extrap, ind_test,   # idx arrays or boolean masks
    levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    set_title=None,
    model_colors=None,                  # dict model->color
    save_dir=None, fname="coverage_two_panel",
    block=False, show_ci=True
):
    """
    Two-panel nominal vs empirical coverage overlay across models.
      Left  : test ∩ interpolation
      Right : test ∩ extrapolation
    """
    y = np.squeeze(np.asarray(y_true))
    N = y.shape[0]

    def _to_index_local(idx_like, N):
        arr = np.asarray(idx_like)
        if arr.dtype == bool:
            assert arr.shape[0] == N, "boolean mask must match N"
            return np.flatnonzero(arr)
        return arr.astype(int)

    ii = _to_index_local(ind_interp, N)
    ie = _to_index_local(ind_extrap, N)
    it = _to_index_local(ind_test,   N)

    it_i = np.intersect1d(it, ii)   # test ∩ interp
    it_e = np.intersect1d(it, ie)   # test ∩ extrap

    # Container for stats you can dump into a table/caption
    stats = {"interp": {}, "extrap": {}}

    panels = [("Interpolation", it_i), ("Extrapolation", it_e)]
    fig, axes = plt.subplots(1, 2, figsize=iclr_figsize(layout="double"), sharex=True, sharey=True)

    for ax, (panel_name, idx) in zip(axes, panels):
        # diagonal (ideal)
        ax.plot([levels[0], levels[-1]], [levels[0], levels[-1]], "k--", lw=1, label="Ideal")
        for mname, P in preds_by_model.items():
            P = np.squeeze(np.asarray(P))
            assert P.ndim == 2, f"{mname}: preds must be (S,N) or (S,N,1)"
            S, Np = P.shape
            assert Np == N, f"{mname}: N mismatch"

            # subset to panel points
            if idx.size == 0:
                continue
            P_sub = P[:, idx]
            y_sub = y[idx]

            cov = coverage_curve_from_samples(P_sub, y_sub, levels=levels)
            emp = cov["empirical"]                # (K,)
            K = len(emp)
            color = model_colors.get(mname, None) if model_colors else None
            ax.plot(levels, emp, marker='o', lw=2, label=mname, color=color)

            # binomial CI (normal approx). Optional but handy.
            if show_ci:
                n = float(len(y_sub))
                se = np.sqrt(np.maximum(emp*(1-emp)/np.maximum(n,1.0), 1e-12))
                ax.fill_between(levels, emp - 1.96*se, emp + 1.96*se,
                                alpha=0.12, linewidth=0, color=color)

            # stash stats
            stats_key = "interp" if panel_name == "Interpolation" else "extrap"
            stats[stats_key][mname] = {
                "N": int(len(y_sub)),
                "levels": np.array(levels, dtype=float),
                "empirical": np.array(emp, dtype=float)
            }
        if set_title is not None:
            ax.set_title(f"Coverage vs Nominal — {panel_name}")
        ax.set_xlabel("Nominal coverage (α)")
        ax.set_ylabel("Empirical coverage")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(levels[0], levels[-1])
        ax.set_ylim(levels[0], levels[-1])

    # one legend
    handles, labels = axes[0].get_legend_handles_labels()
    if len(handles) == 0:   # models added on the right axis only
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
        fig.subplots_adjust(bottom=0.2)

    label_subplots(axes)
    fig.tight_layout()
    if save_dir:
        save_plot(save_dir, fname)
    if block:
        plt.show()
    plt.close(fig)

def _to_index(idx_like, N):
    idx_like = np.asarray(idx_like)
    if idx_like.dtype == bool:
        assert idx_like.shape[0] == N, "boolean mask must match N"
        return np.flatnonzero(idx_like)
    return idx_like.astype(int)

def _compute_pit_from_mc(y_true, preds, randomized=True, rng=None):
    y = np.squeeze(np.asarray(y_true))      # (N,)
    Y = np.squeeze(np.asarray(preds))       # -> (S,N)
    assert Y.ndim == 2, "preds must be (S,N) or (S,N,1)"
    S, N = Y.shape
    assert y.shape[0] == N, "y_true and preds length mismatch"

    less = np.sum(Y <  y[None, :], axis=0)
    leq  = np.sum(Y <= y[None, :], axis=0)

    if randomized:
        if rng is None:
            rng = np.random.default_rng()
        lo = less / S
        hi = leq  / S
        u = lo + (hi - lo) * rng.random(N)   # randomized PIT in jump
    else:
        u = (less + leq) / (2.0 * S)         # midpoint PIT

    return np.clip(u, 0.0, 1.0)

def _rolling_quantiles(xs, us, window=None):
    n = len(xs)
    if window is None:
        window = max(5, n // 8)
    half = max(1, window // 2)
    q50 = np.empty(n); q10 = np.empty(n); q90 = np.empty(n)
    for i in range(n):
        lo = max(0, i - half); hi = min(n, i + half + 1)
        u_slice = us[lo:hi]
        q50[i] = np.median(u_slice)
        q10[i] = np.quantile(u_slice, 0.10)
        q90[i] = np.quantile(u_slice, 0.90)
    return q50, q10, q90

def _uniformity_p(u):
    try:
        from scipy.stats import kstest
        stat, p = kstest(u, 'uniform', args=(0,1))
        return float(p), float(stat)
    except Exception:
        return None, None

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


    # Plot reliability: y=emp, x=nom  (plus a y=x diagonal)
    # Plot sharpness vs coverage: y=sharp, x=nom
    # Plot PIT histogram: histogram of pit_vals, compare to uniform
