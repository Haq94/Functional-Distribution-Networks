import os
import numpy as np
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

