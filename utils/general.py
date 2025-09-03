import os
import re
import numpy as np

def get_latest_run_dir(model_type, base_dir="results/single_task_experiment"):
    """
    Return the latest run directory path for a given model_type based on modification time.

    New structure:
    results/single_task_experiment/<date_time>/seed<seed>/<model_type>/
    """
    run_dirs = []

    if not os.path.exists(base_dir):
        return None

    for date_time_dir in os.listdir(base_dir):
        date_time_path = os.path.join(base_dir, date_time_dir)
        if not os.path.isdir(date_time_path):
            continue

        for seed_dir in os.listdir(date_time_path):
            seed_path = os.path.join(date_time_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue

            model_path = os.path.join(seed_path, model_type)
            if os.path.isdir(model_path):
                run_dirs.append(model_path)

    return max(run_dirs, key=os.path.getmtime) if run_dirs else None


def extract_timestamp_from_dir(run_dir):
    """
    Extracts the timestamp string (underscored) from the full path to the model directory.

    Returns format: 'YYYY-MM-DD_HH-MM-SS'
    """
    parts = run_dir.split(os.sep)
    try:
        # Convert 'YYYY_MM_DD_HH_MM_SS' → 'YYYY-MM-DD_HH-MM-SS'
        date_time_str = parts[-3].replace('_', '-')
        return date_time_str
    except IndexError:
        raise ValueError(f"Could not extract timestamp from directory: {run_dir}")


def extract_seed_from_dir(run_dir):
    """
    Extracts the seed (int) from the full path to the model directory.
    """
    parts = run_dir.split(os.sep)
    try:
        seed_str = parts[-2]
        if seed_str.startswith("seed"):
            return int(seed_str[len("seed"):])
        else:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract seed from directory: {run_dir}")


def get_all_experiment_runs(base_dir="results/single_task_experiment"):
    """
    Returns all (model_type, seed, date_time) tuples under new directory structure.
    """
    runs = []
    if not os.path.exists(base_dir):
        return []

    for date_time_dir in os.listdir(base_dir):
        date_time_path = os.path.join(base_dir, date_time_dir)
        if not os.path.isdir(date_time_path):
            continue

        date_time = date_time_dir.replace('_', '-')

        for seed_dir in os.listdir(date_time_path):
            seed_match = re.match(r"seed(\d+)", seed_dir)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))
            seed_path = os.path.join(date_time_path, seed_dir)

            for model_type in os.listdir(seed_path):
                model_path = os.path.join(seed_path, model_type)
                if os.path.isdir(model_path):
                    runs.append((model_type, seed, date_time))

    return sorted(runs, key=lambda x: (x[0], x[1], x[2]))


def get_seed_time_pairs_for_models(runs, model_type_list):
    """
    Extract unique (seed, date_time) pairs for the specified model types.
    """
    seen = set()
    result = []
    for model_type, seed, date_time in runs:
        if model_type in model_type_list:
            pair = (seed, date_time)
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
    return result

def set_determinism(seed: int = 0, cuda: bool = True):
    import os, random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Turn OFF speed-optimal algo selection; turn ON deterministic kernels
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Stronger guard (may error if an op has no deterministic impl):
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Optional: hash determinism for some libs
        os.environ.setdefault("PYTHONHASHSEED", str(seed))

def count_parameters(model, trainable_only=True):
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model (nn.Module): the model
        trainable_only (bool): if True, count only parameters with requires_grad=True

    Returns:
        int: number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
    
def build_model_dict(model_type, **kwargs):
    model_dict = {}
    # model_dict['model_type'] = model_type
    if model_type in {'IC_FDNet', 'LP_FDNet', 'HyperNet'}:
        model_dict['hidden_dim'] = kwargs.get('hidden_dim', 32)
        model_dict['hyper_hidden_dim'] = kwargs.get('hyper_hidden_dim', 32)
    elif model_type == 'BayesNet':
        model_dict['hidden_dim'] = kwargs.get('hidden_dim', 32)
    elif model_type == 'GaussHyperNet':
        model_dict['hidden_dim'] = kwargs.get('hidden_dim', 32)
        model_dict['hyper_hidden_dim'] = kwargs.get('hyper_hidden_dim', 32)
        model_dict['latent_dim'] = kwargs.get('latent_dim', 10)
    elif model_type in {'MLPNet', 'MLPDropoutNet'}:
        model_dict['hidden_dim'] = kwargs.get('hidden_dim', 32)
        model_dict['dropout_rate'] = kwargs.get('dropout_rate', 0.1)
    elif model_type == 'DeepEnsembleNet':
        model_dict['hidden_dim'] = kwargs.get('hidden_dim', 32)
        model_dict['dropout_rate'] = kwargs.get('dropout_rate', 0.1)
        model_dict['num_models'] = kwargs.get('num_models', 5)
        model_dict['ensemble_seed_list'] = kwargs.get('ensemble_seed_list', [n for n in range(model_dict['num_models'])])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_dict


