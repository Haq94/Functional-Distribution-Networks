import os
import re

def get_latest_run_dir(model_name, base_dir="results//single_task_experiment"):
    subdirs = [
        os.path.join(base_dir, model_name, d)
        for d in os.listdir(os.path.join(base_dir, model_name))
        if os.path.isdir(os.path.join(base_dir, model_name, d))
    ]
    return max(subdirs, key=os.path.getmtime) if subdirs else None

def extract_timestamp_from_dir(run_dir):
    """
    Extracts the timestamp from a run directory name formatted as: ModelType_seedX_YYYY-MM-DD_HH-MM-SS

    Args:
        run_dir (str): Directory name or full path

    Returns:
        str: Timestamp string in format 'YYYY-MM-DD_HH-MM-SS'
    """
    base = os.path.basename(run_dir)
    parts = base.split("_")
    if len(parts) >= 4:
        return "_".join(parts[-2:])  # ['YYYY-MM-DD', 'HH-MM-SS']
    else:
        raise ValueError(f"Could not extract timestamp from directory: {run_dir}")

def extract_seed_from_dir(run_dir):
    """
    Extracts the seed from a run directory name formatted as: ModelType_seedX_YYYY-MM-DD_HH-MM-SS

    Args:
        run_dir (str): Directory name or full path

    Returns:
        int: Extracted seed
    """
    base = os.path.basename(run_dir)
    try:
        seed_str = base.split("_seed")[1].split("_")[0]
        return int(seed_str)
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract seed from directory: {run_dir}")

def get_all_experiment_runs(base_dir="results"):
    """
    Scan the results directory and return all (model_type, seed, date_time) tuples found.

    Args:
        base_dir (str): Base results directory, default "results".

    Returns:
        List of tuples: [(model_type, seed, date_time), ...]
    """
    runs = []
    pattern = re.compile(r'^(.*)_seed(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$')

    if not os.path.exists(base_dir):
        return []

    for model_type in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_type)
        if not os.path.isdir(model_path):
            continue

        for run_dir in os.listdir(model_path):
            match = pattern.match(run_dir)
            if match:
                mt, seed, dt = match.groups()
                runs.append((mt, int(seed), dt))

    return sorted(runs, key=lambda x: (x[0], x[1], x[2]))

def get_seed_time_pairs_for_models(runs, model_type_list):
    """
    Extract unique (seed, date_time) pairs for the specified model types.

    Args:
        runs (list of tuples): List of (model_type, seed, date_time).
        model_type_list (list of str): Model types to include.

    Returns:
        list of tuples: Unique (seed, date_time) pairs.
    """
    seen = set()
    result = []
    for model_type, seed, date_time in runs:
        pair = (seed, date_time)
        if model_type in model_type_list and pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result

