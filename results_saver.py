import os
import numpy as np
import json

def save_analysis_arrays(metrics_output, save_dir):
    """
    Save analysis arrays (mean, var, residuals, etc.) to .npy files.

    Args:
        metrics_output: tuple returned by metrics(...)
        save_dir: path to folder to save .npy files
    """
    keys = [
        "mean", "var", "std", "res_prec", "res_acc", "bias", "mse", "bias_var_diff", "nll"
    ]
    os.makedirs(save_dir, exist_ok=True)
    for key, arr in zip(keys, metrics_output):
        np.save(os.path.join(save_dir, f"{key}.npy"), arr)


def save_results(exp_id, model_type, seed, mean, std, y_true, nll, rmse, results_dir="Results"):
    """
    Saves predictions and metrics as .npy and .json under results_dir/exp_id/
    """
    exp_path = os.path.join(results_dir, exp_id)
    os.makedirs(exp_path, exist_ok=True)

    # Save arrays
    np.save(os.path.join(exp_path, "mean.npy"), mean)
    np.save(os.path.join(exp_path, "std.npy"), std)
    np.save(os.path.join(exp_path, "y_true.npy"), y_true)

    # Save metrics
    metrics = {
        "model": model_type,
        "seed": seed,
        "nll": float(nll),
        "rmse": float(rmse)
    }
    with open(os.path.join(exp_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
