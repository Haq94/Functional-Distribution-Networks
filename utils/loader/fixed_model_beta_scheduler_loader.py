import os
import numpy as np

def load_all_scheduler_metrics(root_dir="results/fixed_model_beta_scheduler_experiment"):
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
