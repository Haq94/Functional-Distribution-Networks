"""
Configuration for single-task toy regression experiments (paper spec).

This defines a single baseline_params dict that main.py expects.
"""

# ~1k parameter configs per model (matches paper / previous code comments).
model_dict = {
    "IC_FDNet":       {"hidden_dim": 23, "hyper_hidden_dim": 6},       # ≈1004 params
    "LP_FDNet":       {"hidden_dim": 24, "hyper_hidden_dim": 5},       # ≈1011
    # "HyperNet":       {"hidden_dim": 25, "hyper_hidden_dim": 9},       # ≈1012
    "BayesNet":       {"hidden_dim": 166},                             # ≈998
    "GaussHyperNet":  {"hidden_dim": 24, "hyper_hidden_dim": 5, "latent_dim": 9},  # ≈994
    # "MLPNet":         {"hidden_dim": 333, "dropout_rate": 0.1},        # ≈1000
    "MLPDropoutNet":  {"hidden_dim": 333, "dropout_rate": 0.1},        # ≈1000
    "DeepEnsembleNet": {
        "hidden_dim": 33,
        "dropout_rate": 0.1,
        "num_models": 10,
        "ensemble_seed_list": list(range(10)),
    },  # ≈1000 total
}

# Training schedule (matches the paper)
epochs = 400
ensemble_epochs = 40

# Monte Carlo samples
MC_train = 1
MC_val = 100
MC_test = 100

# Input / region specification for toy task regression
region = (-10.0, 10.0)          # full x-range used for plots
region_interp = (-1.0, 1.0)     # "ID" interpolation region

n_train = 1024
n_test = 2001
n_val_interp = 256
n_val_extrap = 256

# Checkpoint selection configuration:
#   - stochastic models: minimize CRPS in interpolation region
#   - deterministic models: minimize MSE in interpolation region
checkpoint_dict = {
    "stoch": {
        "metric_str": "crps",
        "region_interp": region_interp,
        "min_or_max": "min",
        "interp_or_extrap": "interp",
    },
    "det": {
        "metric_str": "mse",
        "region_interp": region_interp,
        "min_or_max": "min",
        "interp_or_extrap": "interp",
    },
}

# Beta schedule presets (used by main.get_beta_param_dict)
linear_beta_scheduler = {
    "beta_scheduler": "linear",
    "warmup_epochs": epochs // 2,
    "beta_max": 1.0,
}
cosine_beta_scheduler = {
    "beta_scheduler": "cosine",
    "warmup_epochs": epochs // 2,
    "beta_max": 0.01,
}
sigmoid_beta_scheduler = {
    "beta_scheduler": "sigmoid",
    "warmup_epochs": epochs // 2,
    "beta_max": 1.0,
}
unity_beta_scheduler = {
    "beta_scheduler": "constant",
    "warmup_epochs": 0,
    "beta_max": 1.0,
}
zero_beta_scheduler = {
    "beta_scheduler": "constant",
    "warmup_epochs": 0,
    "beta_max": 0.0,
}

# Default plot dictionary; SingleTaskExperiment will further interpret this.
plot_dict = {
    "Single": [],
    "Overlay": [],
}

# Single dict that main.py expects
baseline_params = {
    "model_dict": model_dict,
    "epochs": epochs,
    "ensemble_epochs": ensemble_epochs,
    "MC_train": MC_train,
    "MC_val": MC_val,
    "MC_test": MC_test,
    "checkpoint_dict": checkpoint_dict,
    "linear_beta_scheduler": linear_beta_scheduler,
    "cosine_beta_scheduler": cosine_beta_scheduler,
    "sigmoid_beta_scheduler": sigmoid_beta_scheduler,
    "unity_beta_scheduler": unity_beta_scheduler,
    "zero_beta_scheduler": zero_beta_scheduler,
    "region": region,
    "region_interp": region_interp,
    "n_train": n_train,
    "n_test": n_test,
    "n_val_interp": n_val_interp,
    "n_val_extrap": n_val_extrap,
    "plot_dict": plot_dict,
}
