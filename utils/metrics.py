import numpy as np

def compute_rmse(mean, y_true):
    """
    Root Mean Squared Error
    """
    return np.sqrt(np.mean((mean - y_true) ** 2))

def compute_nll(mean, y_true, std, eps=1e-6):
    """
    Negative Log Likelihood for Gaussian predictive distribution
    NLL = 0.5 * log(2πσ^2) + (y - μ)^2 / (2σ^2)
    """
    std = np.maximum(std, eps)
    nll = 0.5 * np.log(2 * np.pi * std**2) + ((y_true - mean) ** 2) / (2 * std**2)
    return np.mean(nll)
