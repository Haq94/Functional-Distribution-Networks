import torch
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle

def save_analysis_arrays(metric_outputs, save_dir):
    """
    Save analysis arrays (mean, var, std, etc.) to .npy files.

    Args:
        metrics_output: tuple from `metrics(...)` function
        save_dir (str): directory to save .npy files into
    """
    os.makedirs(save_dir, exist_ok=True)
    keys = [
        "mean", "var", "std", "residual_precision", "residual_accuracy",
        "bias", "mse", "bias_var_diff", "nll"
    ]
    for key, array in zip(keys, metric_outputs):
        np.save(os.path.join(save_dir, f"{key}.npy"), array)

def save_plot(save_dir, plot_name, dpi=300, formats=("pkl", "png", "pdf"), fig=None):
    """
    Save a matplotlib figure in multiple formats.

    Args:
        save_dir (str): Directory to save the plot files.
        plot_name (str): Base filename (without extension).
        dpi (int): Resolution for raster formats like PNG.
        formats (tuple): Tuple of formats to save (e.g., "pkl", "png", "pdf").
        fig (matplotlib.figure.Figure or None): The figure to save. If None, uses current figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    base_path = os.path.join(save_dir, plot_name)
    fig = fig or plt.gcf()

    for fmt in formats:
        full_path = f"{base_path}.{fmt}"
        if fmt == "pkl":
            with open(full_path, "wb") as f:
                pickle.dump(fig, f)
        else:
            fig.savefig(full_path, dpi=dpi, format=fmt, bbox_inches="tight")

def save_experiment_outputs(metric_outputs, model, trainer, summary, x_train, y_train, x_test, y_test, save_dir):
    # Create directory
    os.makedirs(os.path.join(save_dir, "analysis"), exist_ok=True)

    # Save metrics
    save_analysis_arrays(metric_outputs, os.path.join(save_dir, "analysis"))
    np.savez(os.path.join(save_dir, "analysis", "loss_curve_data.npz"),
                        losses=trainer.losses,
                        mses=trainer.mses,
                        kls=trainer.kls,
                        betas=trainer.betas)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save test and training data
    np.savez(os.path.join(save_dir, "analysis", "data.npz"),
             x_train=x_train.cpu().numpy(),
             y_train=y_train.cpu().numpy(),
             x_test=x_test.cpu().numpy(),
             y_test=y_test.cpu().numpy())

    # Save summary
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

# OLD CODE =======================================================================================================

# def save_results(exp_id, model_type, seed, mean, std, y_true, nll, rmse, results_dir="Results"):
#     """
#     Saves predictions and metrics as .npy and .json under results_dir/exp_id/
#     """
#     exp_path = os.path.join(results_dir, exp_id)
#     os.makedirs(exp_path, exist_ok=True)

#     # Save arrays
#     np.save(os.path.join(exp_path, "mean.npy"), mean)
#     np.save(os.path.join(exp_path, "std.npy"), std)
#     np.save(os.path.join(exp_path, "y_true.npy"), y_true)

#     # Save metrics
#     metrics = {
#         "model": model_type,
#         "seed": seed,
#         "nll": float(nll),
#         "rmse": float(rmse)
#     }
#     with open(os.path.join(exp_path, "metrics.json"), "w") as f:
#         json.dump(metrics, f, indent=2)