import re
import os
import json
import numpy as np

from utils.general import get_latest_run_dir, extract_seed_from_dir, extract_timestamp_from_dir, get_all_experiment_runs, get_seed_time_pairs_for_models

class SingleTaskExperimentLoader:
    def __init__(self, model_type, seed, date_time, base_dir="results/single_task_experiment"):
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

        # Reformat date for folder name
        self.date_time_folder = date_time.replace("-", "_")
        self.seed_folder = f"seed{seed}"
        
        # New path structure
        self.run_path = os.path.join(base_dir, self.date_time_folder, self.seed_folder, model_type)
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
        metrics = {}
        metric_names = [
            "mean", "var", "std", "residual_precision", "residual_accuracy",
            "bias", "mse", "bias_var_diff", "nlpd"
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
    
    def load_ind(self):
        path = os.path.join(self.analysis_path, "ind.npz")
        ind = np.load(path)
        return {
            "ind_train": ind["ind_train"],
            "ind_test": ind["ind_test"],
            "ind_interp": ind["ind_interp"],
            "ind_extrap": ind["ind_extrap"]
        }

    @staticmethod
    def get_latest_loader(model_type, base_dir="results/single_task_experiment"):
        # Update this if your latest-run detection uses old naming
        run_dir = get_latest_run_dir(model_type, base_dir=base_dir)
        if run_dir:
            seed = extract_seed_from_dir(run_dir)
            date_time = extract_timestamp_from_dir(run_dir)
            return SingleTaskExperimentLoader(model_type, seed, date_time, base_dir)
        else:
            return None



def get_all_experiment_runs(base_dir="results/single_task_experiment"):
    """
    Scan the new results directory structure and return all (model_type, seed, date_time) tuples.

    Structure:
    results/single_task_experiment/<date_time>/seed<seed>/<model_type>/

    Returns:
        List of tuples: [(model_type, seed, date_time), ...]
    """
    runs = []
    seed_pattern = re.compile(r'^seed(\d+)$')

    if not os.path.exists(base_dir):
        return []

    for date_time_dir in os.listdir(base_dir):
        date_time_path = os.path.join(base_dir, date_time_dir)
        if not os.path.isdir(date_time_path):
            continue

        # Re-convert underscores to dashes for standard datetime format
        date_time = date_time_dir.replace('_', '-')

        for seed_dir in os.listdir(date_time_path):
            seed_match = seed_pattern.match(seed_dir)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))
            seed_path = os.path.join(date_time_path, seed_dir)

            for model_type in os.listdir(seed_path):
                model_path = os.path.join(seed_path, model_type)
                if os.path.isdir(model_path):
                    runs.append((model_type, seed, date_time))

    return sorted(runs, key=lambda x: (x[0], x[1], x[2]))



def single_task_overlay_loader(seeds, date_times):
    """
    Load metrics, losses, summaries, and data across models for given seeds and date_times.

    Args:
        seeds (list of int): Seeds to include.
        date_times (list of str): Date-time strings in 'YYYY-MM-DD_HH-MM-SS' format.

    Returns:
        loaders, metrics, losses, summary, x_train, y_train, x_test, y_test, seed_date_time_list
    """
    # List of models to load
    model_types = ['LP_FDNet', 'HyperNet', 'IC_FDNet', 'BayesNet', 'GaussHyperNet', 'MLPNet', 'DeepEnsembleNet']

    # Load all available runs and filter
    runs_list = get_all_experiment_runs(base_dir="results/single_task_experiment")
    seed_date_time_all = get_seed_time_pairs_for_models(runs=runs_list, model_type_list=model_types)
    seed_date_time_list = [(s, t) for s, t in seed_date_time_all if s in seeds and t in date_times.replace('_','-')]

    # Initialize data containers
    loaders = {}
    x_train, y_train, x_test, y_test = {}, {}, {}, {}
    ind_test, ind_train, ind_interp, ind_extrap = {}, {}, {}, {}

    metrics = {name: {} for name in [
        "mean", "var", "std", "res_precision", "res_accuracy",
        "bias", "mse", "bias_var_diff", "nlpd"
    ]}

    losses = {name: {} for name in ["losses", "mses", "kls", "betas"]}

    summary = {name: {} for name in [
        'desc', 'model', 'seed', 'rmse', 'mean_nlpd', 'training_time', 'timestamp',
        'epochs', 'beta_param_dict', 'x_min', 'x_max', 'region_interp', 'frac_train'
    ]}

    for seed, date_time in seed_date_time_list:
        for model in model_types:
            run_tag = f"{model}_seed{seed}_{date_time}"
            try:
                loader = SingleTaskExperimentLoader(model, seed, date_time)

                # Load metrics
                metric_dict = loader.load_metrics()
                for k in metrics:
                    metrics[k].setdefault((seed, date_time), {})[model] = metric_dict.get(k)

                # Load loss curve
                loss_dict = loader.load_loss_curve()
                for k in losses:
                    losses[k].setdefault((seed, date_time), {})[model] = loss_dict.get(k)

                # Load summary
                summary_dict = loader.load_summary()
                for k in summary:
                    summary[k].setdefault((seed, date_time), {})[model] = summary_dict.get(k)

                # Store loader
                loaders.setdefault((seed, date_time), {})[model] = loader

            except FileNotFoundError:
                print(f"Skipping missing result: {run_tag}")
            except Exception as e:
                print(f"Error loading {run_tag}: {e}")
            finally:
                print(f"Checked: {run_tag}")

        # Load data once per (seed, date_time)
        try:
            loader = loaders.get((seed, date_time), {}).get(model_types[0])  # Any model should have data
            if loader:
                data = loader.load_data()
                x_train[(seed, date_time)] = data["x_train"]
                y_train[(seed, date_time)] = data["y_train"]
                x_test[(seed, date_time)] = data["x_test"]
                y_test[(seed, date_time)] = data["y_test"]

                ind = loader.load_ind()
                ind_train[(seed, date_time)] = ind["ind_train"]
                ind_test[(seed, date_time)] = ind["ind_test"]
                ind_interp[(seed, date_time)] = ind["ind_interp"]
                ind_extrap[(seed, date_time)] = ind["ind_extrap"]

        except Exception as e:
            print(f"Failed to load data for {seed}_{date_time}: {e}")

    print("Loaded runs:", list(loaders.keys()))

    return loaders, metrics, losses, summary, x_train, y_train, x_test, y_test, ind_train, ind_test, ind_interp, ind_extrap, seed_date_time_list

