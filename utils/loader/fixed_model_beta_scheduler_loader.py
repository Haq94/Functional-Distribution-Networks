import os
import numpy as np

def load_scheduler_metrics(root_dir="results/fixed_model_beta_scheduler_experiment"):
    """
    Load all metrics.npz files from beta scheduler experiment.
    Returns:
        A nested dict:
        data[seed][model_type][scheduler_param_str] = dict of metrics
    """
    data = {}
    for seed_dir in os.listdir(root_dir):
        seed_path = os.path.join(root_dir, seed_dir)
        if not os.path.isdir(seed_path): continue
        seed = int(seed_dir.replace("seed_", ""))
        data[seed] = {}

        for model_dir in os.listdir(seed_path):
            model_path = os.path.join(seed_path, model_dir)
            if not os.path.isdir(model_path): continue
            data[seed][model_dir] = {}

            for param_dir in os.listdir(model_path):
                run_path = os.path.join(model_path, param_dir)
                metrics_file = os.path.join(run_path, "metrics.npz")
                if not os.path.exists(metrics_file): continue

                metrics = dict(np.load(metrics_file, allow_pickle=True))
                data[seed][model_dir][param_dir] = metrics

    return data

def get_metric(metrics_dicts, 
               seed, 
               model_type, 
               beta_scheduler, 
               beta_max, 
               warmup_epochs, 
               metric_name):
    """
    Retrieve a specific metric from the nested metrics dictionary.

    Args:
        metrics_dict (dict): Nested dictionary storing all experiment metrics.
        seed (int)
        model_type (str): e.g., 'IC_FDNet'
        beta_scheduler (str): e.g., 'linear'
        beta_max (float)
        warmup_epochs (int)
        metric_name (str): one of ["losses_per_epoch", "mse_per_epoch", "kls_per_epoch", "beta_per_epoch", ...]

    Returns:
        List or array corresponding to the metric, or None if not found.
    """
    try:
        return metrics_dicts[seed][model_type][beta_scheduler][beta_max][warmup_epochs][metric_name]
    except KeyError as e:
        print(f"[Warning] Missing key in metrics_dict: {e}")
        return None

