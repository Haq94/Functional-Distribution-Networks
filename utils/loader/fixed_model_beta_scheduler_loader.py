import os
import glob
import numpy as np

def fixed_model_beta_scheduler_loader(results_root="results/fixed_model_beta_scheduler_experiment"):
    results = []
    for model_dir in glob.glob(os.path.join(results_root, "*")):
        for run_dir in glob.glob(os.path.join(model_dir, "*")):
            metrics_path = os.path.join(run_dir, "metrics.npz")
            if os.path.exists(metrics_path):
                data = dict(np.load(metrics_path, allow_pickle=True))
                data["run_dir"] = run_dir
                results.append(data)
    return results
