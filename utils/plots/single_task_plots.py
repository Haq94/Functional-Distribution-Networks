import os
import matplotlib.pyplot as plt
import numpy as np
import itertools

from utils.metrics import metrics
from utils.saver.general_saver import save_plot
from utils.loader.single_task_loader import single_task_overlay_loader

# Single Plots=========================================================================================

def plot_coverage_two_panel(
    preds_by_model,   # dict: {model_name: preds (S,N) or (S,N,1)}
    y_true,           # (N,)
    ind_interp, ind_extrap, ind_test,   # idx arrays or boolean masks
    levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

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

    fig.tight_layout()
    if save_dir:
        save_plot(save_dir, fname)
    if block:
        plt.show()
    plt.close(fig)

def plot_pit_two_panel(
    x,
    y_true,
    preds,                      # (S,N) or (S,N,1)
    ind_interp,                 # indices or boolean mask (N,)
    ind_extrap,                 # indices or boolean mask (N,)
    ind_test,                   # indices or boolean mask (N,)
    ind_train,                  # indices or boolean mask (N,)
    interp_min=None,            # optional shading
    interp_max=None,
    randomized=True,
    trend=True,                 # draw running median + 10–90% band
    trend_window=None,          # None -> auto
    save_dir=None,
    fname="pit_two_panel",
    block=False,
):
    """
    Two-panel PIT vs x: left = test interpolation, right = test extrapolation.
    Returns dict with summary stats for each panel.
    """
    x = np.squeeze(np.asarray(x))
    y_true = np.squeeze(np.asarray(y_true))
    N = x.shape[0]

    ii = _to_index(ind_interp, N)
    ie = _to_index(ind_extrap, N)
    it = _to_index(ind_test,   N)
    ir = _to_index(ind_train,  N)

    it_i = np.intersect1d(it, ii)   # test ∩ interp
    it_e = np.intersect1d(it, ie)   # test ∩ extrap

    # PIT for all points (so we can also show training for context)
    u_all = _compute_pit_from_mc(y_true, preds, randomized=randomized)
    stats = {}

    # convenience sorter
    def sorted_subset(idx):
        order = np.argsort(x[idx])
        return x[idx][order], u_all[idx][order], idx[order]

    panels = [
        ("PIT vs x — Test Interpolation", it_i, "red"),
        ("PIT vs x — Test Extrapolation", it_e, "blue"),
    ]

    plt.figure(figsize=(12, 5))
    for k, (title, idx, color) in enumerate(panels):
        ax = plt.subplot(1, 2, k+1)

        xs, us, idx_sorted = sorted_subset(idx)
        x_tr, u_tr = x[ir], u_all[ir]

        # refs
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.9)
        ax.axhline(0.1, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.axhline(0.9, color='gray', linestyle=':', linewidth=1, alpha=0.6)

        if interp_min is not None and interp_max is not None:
            ax.axvspan(interp_min, interp_max, color='C1', alpha=0.08, lw=0)

        # scatter
        ax.scatter(xs, us, s=14, alpha=0.85, c=color,
                   label=f"{'Interp' if k==0 else 'Extrap'} Test (N={len(us)})")
        ax.scatter(x_tr, u_tr, s=12, alpha=0.5, facecolors='none', edgecolors='green',
                   label="Training points")

        # trend
        if trend and len(us) >= 5:
            q50, q10, q90 = _rolling_quantiles(xs, us, window=trend_window)
            ax.plot(xs, q50, lw=2, color=color, alpha=0.9, label="running median")
            ax.fill_between(xs, q10, q90, color=color, alpha=0.15, linewidth=0,
                            label="10–90% band")
        else:
            q50 = np.median(us); q10 = np.quantile(us,0.10); q90 = np.quantile(us,0.90)

        # axes & labels
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u = \hat F_x(y)$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

        # summary stats for this panel
        med = float(np.median(us)) if len(us)>0 else np.nan
        q10g = float(np.quantile(us, 0.10)) if len(us)>0 else np.nan
        q90g = float(np.quantile(us, 0.90)) if len(us)>0 else np.nan
        band = q90g - q10g if len(us)>0 else np.nan
        pval, ks = _uniformity_p(us)
        stats["interp" if k==0 else "extrap"] = {
            "N": int(len(us)),
            "median": med,
            "median_delta": med - 0.5,   # bias (ideal 0)
            "q10": q10g,
            "q90": q90g,
            "band_10_90": band,          # ideal ≈ 0.8
            "ks_pvalue": pval,           # None if SciPy not installed
            "ks_stat": ks,
        }

    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, fname)
    if block:
        plt.show()
    plt.close()





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
    axs[0].scatter(x[ind_c], mean[ind_c], label='Training Points', color='red', s=10)
    axs[0].set_title(r"$\mu$ vs $x$")
    axs[0].axvline(x=interp_min, color='red', linestyle='--')
    axs[0].axvline(x=interp_max, color='red', linestyle='--')
    axs[0].set_ylabel(r"$\mu$")
    axs[0].grid(True)

    axs[1].plot(x, 10 * np.log10(var))
    axs[1].scatter(x[ind_c], 10 * np.log10(var[ind_c]), label='Training Points', color='red', s=10)
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
    axs[0].scatter(x[ind_c], bias[ind_c], label='Training Points', color='red', s=10)
    axs[0].set_title(r"$Bias$ vs $x$")
    axs[0].axvline(x=interp_min, color='red', linestyle='--')
    axs[0].axvline(x=interp_max, color='red', linestyle='--')
    axs[0].grid(True)

    axs[1].plot(x, 10 * np.log10(mse))
    axs[1].scatter(x[ind_c], 10 * np.log10(mse[ind_c]), label='Training Points', color='red', s=10)
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

def plot_nlpd(x, nlpd_kde, nlpd_hist, ind_c, interp_min, interp_max, save_dir=None, block=False):
    plt.figure(figsize=(10, 6))
    plt.plot(x, nlpd_kde, color="purple")
    plt.scatter(x[ind_c], nlpd_kde[ind_c], label="Training Points", color="red", s=10)
    plt.axvline(x=interp_min, color='red', linestyle='--')
    plt.axvline(x=interp_max, color='red', linestyle='--')
    plt.title(r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$ vs $x$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "nlpd_kde")
    if block:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(x, nlpd_hist, color="purple")
    plt.scatter(x[ind_c], nlpd_hist[ind_c], label="Training Points", color="red", s=10)
    plt.axvline(x=interp_min, color='red', linestyle='--')
    plt.axvline(x=interp_max, color='red', linestyle='--')
    plt.title(r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$ vs $x$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "nlpd_hist")
    if block:
        plt.show()
    plt.close()

def plot_crps(x, crps, ind_c, interp_min, interp_max, save_dir=None, block=False):
    plt.figure(figsize=(10, 6))
    plt.plot(x, crps, color="purple")
    plt.scatter(x[ind_c], crps[ind_c], label="Training Points", color="red", s=10)
    plt.axvline(x=interp_min, color='red', linestyle='--')
    plt.axvline(x=interp_max, color='red', linestyle='--')
    plt.title(r"$CRPS$ vs $x$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$CRPS$ $(dB)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "crps")
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

def pdf_and_nll_heatmap(pdf_kde, grid_kde, pdf_hist, grid_hist, preds, x, y, x_min, x_max, save_dir=None, block=False):
    """
    Waterfall-style histogram of pdf and nll
    """
    # n_x = preds.shape[1]
    # hist_matrix = []

    # for i in range(n_x):
    #     hist, _ = np.histogram(preds[:, i], bins=bins, range=(preds.min(), preds.max()), density=True)
    #     hist_matrix.append(hist)

    # hist_matrix = np.array(hist_matrix).T  # shape: [bins, n_x]

    pdf_kde = pdf_kde.T
    pdf_hist = pdf_hist.T

    y_kde = y[(y<=grid_kde.max()) & (y >= grid_kde.min())]
    x_kde = x[(y<=grid_kde.max()) & (y >= grid_kde.min())]

    y_hist = y[(y<=grid_hist.max()) & (y >= grid_hist.min())]
    x_hist = x[(y<=grid_hist.max()) & (y >= grid_hist.min())]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(pdf_kde, aspect='auto', origin='lower',
               extent=[x.min(), x.max(), grid_kde.min(), grid_kde.max()],
               cmap='viridis')
    plt.colorbar()
    plt.plot(x_kde,y_kde,c="red")
    plt.axvline(x=x_min, color='black', linestyle='--')
    plt.axvline(x=x_max, color='black', linestyle='--')
    plt.title(r"$p(y|x)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "pdf_kde_heatmap")
    if block:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.imshow(pdf_hist, aspect='auto', origin='lower',
               extent=[x.min(), x.max(), grid_hist.min(), grid_hist.max()],
               cmap='viridis')
    plt.colorbar()
    plt.plot(x_hist,y_hist,c="red")
    plt.axvline(x=x_min, color='black', linestyle='--')
    plt.axvline(x=x_max, color='black', linestyle='--')
    plt.title(r"$p(y|x)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "pdf_hist_heatmap")
    if block:
        plt.show()
    plt.close()

    nll = -np.log(pdf_kde + 1e-12)
    plt.figure(figsize=(12, 6))
    plt.imshow(nll, aspect='auto', origin='lower',
               extent=[x.min(), x.max(), grid_kde.min(), grid_kde.max()],
               cmap='viridis')
    plt.colorbar()
    plt.plot(x_kde,y_kde,c="red")
    plt.axvline(x=x_min, color='black', linestyle='--')
    plt.axvline(x=x_max, color='black', linestyle='--')
    plt.title(r"$- \log\left( p(y \mid x) \right)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "nll_kde_heatmap")
    if block:
        plt.show()
    plt.close()

    nll = -np.log(pdf_hist + 1e-12)
    plt.figure(figsize=(12, 6))
    plt.imshow(nll, aspect='auto', origin='lower',
               extent=[x.min(), x.max(), grid_hist.min(), grid_hist.max()],
               cmap='viridis')
    plt.colorbar()
    plt.plot(x_hist,y_hist,c="red")
    plt.axvline(x=x_min, color='black', linestyle='--')
    plt.axvline(x=x_max, color='black', linestyle='--')
    plt.title(r"$- \log\left( p(y \mid x) \right)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()
    if save_dir:
        save_plot(save_dir, "nll_hist_heatmap")
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

    # Save folders
    calibration_save_path = os.path.join(save_dir,'calibration')
    losses_save_path = os.path.join(save_dir,'losses')
    metric_path = os.path.join(save_dir,'metrics_vs_x')

    # Compute metrics if they're not already computed
    if metric_outputs is None:
        metric_outputs = metrics(preds_np, y_test_np)
    # Unpack variables
    # for key, value in metric_outputs.items():
    #     exec(f"{key} = metric_outputs['{key}']")
    # mean, var, std, res_prec, res_acc, bias, mse, bias_var_diff, nlpd_kde, grid_kde, nlpd_hist, grid_hist, pdf_kde, pdf_hist, crps = metric_outputs.values()
    mean      = metric_outputs["mean"]
    var       = metric_outputs["var"]
    std       = metric_outputs["std"]
    res_prec  = metric_outputs["res_prec"]
    res_acc   = metric_outputs["res_acc"]
    bias      = metric_outputs["bias"]
    mse       = metric_outputs["mse"]
    nlpd_kde  = metric_outputs["nlpd_kde"]
    grid_kde  = metric_outputs["grid_kde"]
    nlpd_hist = metric_outputs["nlpd_hist"]
    grid_hist = metric_outputs["grid_hist"]
    pdf_kde   = metric_outputs["pdf_kde"]
    pdf_hist  = metric_outputs["pdf_hist"]
    crps      = metric_outputs["crps"]

    plot_loss_curve(trainer.losses, trainer.mses, trainer.kls, trainer.betas, desc=desc, 
                save_dir=losses_save_path, block=False)

    if "residuals" in capabilities:
        plot_residual_scatter(x_test_np, res_prec, res_acc, bias, x_min, x_max, save_dir=metric_path, block=block)
    if "mean" in capabilities:
        plot_mean_prediction(x_test_np, y_test_np, mean, std, preds, x_train_np, y_train_np, x_min, x_max, save_dir=metric_path, block=block, zoom=False)
        # plot_mean_prediction(x_test_np, y_test_np, mean, std, preds, x_train_np, y_train_np, x_min, x_max, save_dir=save_dir, block=block, zoom=True)
    if "variance" in capabilities:

        plot_pit_two_panel(x_test_np, y_test_np, preds, ind_interp, ind_extrap, ind_test, ind_train, save_dir=save_dir, block=block)

        plot_variance(x_test_np, var, mean, ind_train, x_min, x_max, save_dir=metric_path, block=block)

        # pdf_and_nll_heatmap(preds, x_test_np, y_test_np, x_min, x_max, bins=50, save_dir=save_dir, block=block)
        pdf_and_nll_heatmap(pdf_kde, grid_kde, pdf_hist, grid_hist, preds, x_test_np, y_test_np, x_min, x_max, save_dir=calibration_save_path, block=block)

        var_str = r"$\sigma_{\hat{y}}^2$"
        bias_sq_str = r"$Bias^2$"
        bias_str = r"$Bias$"
        mse_str = r"$MSE$"
        nlpd_str = r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$"
        crps_str = r"$CRPS$"

        ###########################################
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

            plot_y_vs_x(x=metric_x, y=metric_y, xlabel=xlabel, ylabel=ylabel, fname=fname, ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=calibration_save_path, block=block)
            plot_y_vs_x_2x2(x=metric_x, y=metric_y, xlabel=xlabel, ylabel=ylabel, fname=fname_2x2, ind_interp=ind_interp, ind_extrap=ind_extrap, ind_test=ind_test, ind_train=ind_train, db_scale=True, save_dir=calibration_save_path, block=block)
        
    if "bias" in capabilities:
        plot_bias_mse(x_test_np, bias, mse, ind_train, x_min, x_max, save_dir=metric_path, block=block)
    if "nlpd" in capabilities:
        plot_nlpd(x_test_np, nlpd_kde, nlpd_hist, ind_train, x_min, x_max, save_dir=metric_path, block=block)
        plot_crps(x_test_np, _dB(crps), ind_train, x_min, x_max, save_dir=metric_path, block=block)


# Overlay Plots================================================================================================================

def plot_single_task_overlay(
    seed,
    date_time,
    model_types,
    stoch_models,
    stoch_metrics,
    model_colors,
    save_dir=None,
    show_figs=False,
    use_db_scale=True
):
    # Re-load data
    loaders, metrics, losses, summary, x_train, y_train, x_test, y_test, ind_train, ind_test, ind_interp, ind_extrap, seed_date_time_list = single_task_overlay_loader([seed], date_time)
    
    for seed_date_time in seed_date_time_list:
        x = x_test[seed_date_time]
        y = y_test[seed_date_time]
        
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
        for metric_label in {"mean", "var", "bias", "mse", "nlpd_kde", "nlpd_hist", "crps"}:
            stoch = metric_label in stoch_metrics
            models = sorted(stoch_models.intersection(model_types)) if stoch else sorted(model_types)

            fig, ax = plt.subplots(figsize=(10, 8))
            metric_min, metric_max = None, None

            for model in models:
                metric = metrics[metric_label][seed_date_time][model]
                if metric_label in {"var", "mse", "crps"} and use_db_scale:
                    metric = _dB(metric)
                    plot_label = "$\sigma_{\hat{y}}^2$ (dB)" if metric_label == "var" else r"$MSE$ $(dB)$" if metric_label == "mse" else r"$CRPS$ $(dB)$"
                elif metric_label in {"nlpd_kde", "nlpd_hist"}:
                    plot_label = r"$- \log\left( p(y_{\text{truth}} \mid x) \right)$"
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
            ax.set_title(plot_label + r" vs $x$")
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
                if (model not in stoch_models and label in {"kls", "losses", "betas"}) or (model == "DeepEnsembleNet" and label in {"kls", "betas"}):
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
        nlpd_kde = {m: metrics["nlpd_kde"][seed_date_time][m] for m in stoch_models.intersection(model_types)}
        nlpd_hist = {m: metrics["nlpd_hist"][seed_date_time][m] for m in stoch_models.intersection(model_types)}
        crps = {m: metrics["crps"][seed_date_time][m] for m in stoch_models.intersection(model_types)}
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

        ###########################################
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

            overlay_scatter_2x2(
                x_dict_db=metric_x,
                y_dict_db=metric_y,
                indices_map=indices_map,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                model_colors=model_colors,
                save_dir=os.path.join(save_dir, 'calibration_scatter_plots') if save_dir else None,
                fname=fname_2x2,
                show_figs=show_figs
            )

            scatter_plot(x=metric_x, y=metric_y, xlabel=xlabel, ylabel=ylabel, fname=fname, model_colors=model_colors)

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
