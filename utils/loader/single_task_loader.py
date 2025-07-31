import os
import json
import numpy as np

from utils.general import get_latest_run_dir, extract_seed_from_dir, extract_timestamp_from_dir

class SingleTaskExperimentLoader:
    def __init__(self, model_type, seed, date_time, base_dir="results//single_task_experiment"):
        """
        Utility to load metrics, losses, summaries, and input/output data.

        Args:
            model_type (str): Name of the model used (e.g., 'IC_FDNet').
            seed (int): Seed for the experiment.
            date_time (str): Timestamp string in format 'YYYY-MM-DD_HH-MM-SS'.
            base_dir (str): Root directory where results are saved.
        """
        self.model_type = model_type
        self.seed = seed
        self.date_time = date_time
        self.base_dir = base_dir

        self.run_name = f"{model_type}_seed{seed}_{date_time}"
        self.run_path = os.path.join(base_dir, model_type, self.run_name)
        self.analysis_path = os.path.join(self.run_path, "analysis")

    def load_summary(self):
        path = os.path.join(self.run_path, "metrics.json")
        with open(path, "r") as f:
            return json.load(f)

    def load_loss_curve(self):
        path = os.path.join(self.analysis_path, "loss_curve_data.npz")
        data = np.load(path)
        return {
            "losses": data["losses"],
            "mses": data["mses"],
            "kls": data["kls"],
            "betas": data["betas"]
        }

    def load_metrics(self):
        """
        Load individual metric arrays saved as .npy files.

        Returns:
            dict: Dictionary with metric names as keys and numpy arrays as values.
        """
        metrics = {}
        metric_names = [
            "mean", "var", "std", "residual_precision", "residual_accuracy",
            "bias", "mse", "bias_var_diff", "nll"
        ]
        for name in metric_names:
            path = os.path.join(self.analysis_path, f"{name}.npy")
            if os.path.exists(path):
                metrics[name if "residual" not in name else name.replace("residual_", "res_")] = np.load(path)
            else:
                print(f"Warning: {name}.npy not found in {self.analysis_path}")
        return metrics


    def load_data(self):
        path = os.path.join(self.analysis_path, "data.npz")
        data = np.load(path)
        return {
            "x_train": data["x_train"],
            "y_train": data["y_train"],
            "x_test": data["x_test"],
            "y_test": data["y_test"]
        }
    
    @staticmethod
    def get_latest_loader(model_type, base_dir="results//single_task_experiment"):
        run_dir = get_latest_run_dir(model_type, base_dir=base_dir)
        if run_dir:
            seed = extract_seed_from_dir(run_dir)  # optional utility
            date_time = extract_timestamp_from_dir(run_dir)  # optional utility
            return SingleTaskExperimentLoader(model_type, seed, date_time, base_dir)
        else:
            return None

if __name__ == '__main__':
    # Imports
    from general import get_latest_run_dir

    # Model and metrics
    model_type = 'IC_FDNet'
    metric = 'residual_scatter'

    # Get latest run directory
    run_dir = get_latest_run_dir(model_type)

    # Create instance
    seed = extract_seed_from_dir(run_dir)
    date_time = extract_timestamp_from_dir(run_dir)
    loader = SingleTaskExperimentLoader(model_type, seed, date_time, base_dir="results//single_task_experiment")

    # Load metrics
    metrics = loader.load_metrics()

    # Load summary
    summary = loader.load_summary()

    # Loss curve data
    loss_curve_data = loader.load_loss_curve()

    # Load data
    data = loader.load_data()

    # Test "get_latest_loader"
    loader = SingleTaskExperimentLoader.get_latest_loader(model_type)




    

