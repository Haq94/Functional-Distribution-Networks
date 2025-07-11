import os

def get_latest_run_dir(model_name, base_dir="results"):
    subdirs = [
        os.path.join(base_dir, model_name, d)
        for d in os.listdir(os.path.join(base_dir, model_name))
        if os.path.isdir(os.path.join(base_dir, model_name, d))
    ]
    return max(subdirs, key=os.path.getmtime) if subdirs else None

