import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_mse_vs_beta(data, model_type, warmup_epochs, scheduler_list=None):
    """
    For fixed warmup_epochs, plot MSE vs beta_max for each beta scheduler.

    Args:
        data: output of `load_all_scheduler_metrics`
        model_type: e.g., "IC_FDNet"
        warmup_epochs: int
        scheduler_list: optionally filter which schedulers to include
    """
    mse_data = defaultdict(list)
    beta_vals = set()

    for seed, model_dict in data.items():
        runs = model_dict.get(model_type, {})
        for run_name, metrics in runs.items():
            if f"warmup{warmup_epochs}" not in run_name:
                continue
            scheduler = run_name.split("_")[0]
            if scheduler_list and scheduler not in scheduler_list:
                continue
            beta_str = [s for s in run_name.split("_") if s.startswith("beta")][0]
            beta_val = float(beta_str.replace("beta", ""))
            beta_vals.add(beta_val)
            mse_data[scheduler].append((beta_val, metrics["final_mse"]))

    beta_vals = sorted(beta_vals)
    for scheduler, vals in mse_data.items():
        # average across seeds
        betas, mses = zip(*vals)
        beta_unique = sorted(set(betas))
        mean_mse = [np.mean([m for b, m in vals if np.isclose(b, bu)]) for bu in beta_unique]
        plt.plot(beta_unique, mean_mse, label=scheduler)

    plt.xlabel("Beta Max")
    plt.ylabel("Final MSE")
    plt.title(f"MSE vs Beta Max (Warmup {warmup_epochs}) - {model_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def plot_scheduler_metric_curves(data, model_type, seed, save_path=None):
    """
    For a given model and seed, overlay Loss, MSE, KL, Beta vs Epoch.
    Each line = one (scheduler, beta_max, warmup_epoch) config.

    Args:
        data (dict): Loaded metrics dictionary from loader.
        model_type (str)
        seed (int)
        save_path (str or None): Optional path to save plot.
    """
    metrics_keys = ["losses_per_epoch", "mse_per_epoch", "kls_per_epoch", "beta_per_epoch"]
    metric_names = ["Loss", "MSE", "KL Divergence", "Beta"]
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True)
    fig.suptitle(f"{model_type} (Seed {seed}) – Metrics vs Epoch", fontsize=16)

    runs = data[seed][model_type]

    for run_name, run_data in runs.items():
        label = f"{run_data['beta_scheduler']}, β={run_data['beta_max']}, w={run_data['warmup_epochs']}"
        epochs = run_data["epoch"]

        for idx, (key, title) in enumerate(zip(metrics_keys, metric_names)):
            y = run_data.get(key)
            if y is not None:
                axes[idx].plot(epochs, y, label=label, alpha=0.8)

    # Axis formatting
    for ax, title in zip(axes, metric_names):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True)

    axes[0].set_ylabel("Value")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, title="Config")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    plt.show()

