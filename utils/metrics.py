import numpy as np
from datetime import datetime

def metrics(preds, y, eps=1e-6):
    """
    Compute various per-sample metrics including NLL.
    
    Args:
        preds (numpy.ndarray): (n_samples, n_points, n_features) multiple stochastic predictions
        y     (torch.Tensor) : (n_points, n_features)            true target output
        eps   (float)        : ()                                small constant for numerical stability

    Returns:
        mean:       (n_points,) mean prediction
        var:        (n_points,) predictive variance
        std:        (n_points,) predictive std deviation
        res_prec:   (n_points, n_samples) deviation from predictive mean
        res_acc:    (n_points, n_samples) deviation from ground truth
        bias:       (n_points,) predictive bias
        mse:        (n_points,) mean squared error
        bias_var_diff: (n_points,) |mse - (bias² + var)|
        nlpd:        (n_points,) negative log predictive distribution (NLPD) per sample
    """
    
    # Convert to numpy arrays
    y = y.cpu().numpy().squeeze().astype(np.float64)
    preds = preds.astype(np.float64)

    # Mean, variance, std: shape = (n_points, 1)
    mean = preds.mean(0)             # (n_points, 1)
    var = preds.var(0) + eps         # (n_points, 1)
    std = np.sqrt(var)                  # (n_points, 1)

    # Residuals
    res_prec = preds.squeeze().T - mean  # (n_points, n_samples)
    res_acc = preds.squeeze().T - y.reshape(-1, 1)

    # Bias and MSE
    bias = mean - y.reshape(-1, 1)              # (n_points, 1)
    mse = (res_acc ** 2).mean(1).reshape(-1, 1)    # (n_points, 1)

    # Bias-variance decomposition error
    bias_var_diff = np.abs(mse - (var + bias**2))  # (n_points, 1)

    # Gaussian NLL per sample (n_points,)
    nlpd = per_x_nlpd_from_samples_hist(preds=preds, y=y, bins=50, y_min=preds.min(), y_max=preds.max())

    return (
        mean.squeeze(),
        var.squeeze(),
        std.squeeze(),
        res_prec,
        res_acc,
        bias.squeeze(),
        mse.squeeze(),
        bias_var_diff.squeeze(),
        nlpd.squeeze()
    )

def per_x_nlpd_from_samples_hist(preds, y, bins=50, y_min=None, y_max=None, eps=1e-12):
    """
    preds: (S, N, 1) MC samples
    y    : (N,)   ground-truth
    returns: (N,) per-x NLPD using histogram density lookup
    """
    S, N, _ = preds.shape
    if y_min is None: y_min = float(preds.min())
    if y_max is None: y_max = float(preds.max())
    edges = np.linspace(y_min, y_max, bins + 1)
    dy = edges[1] - edges[0]

    nlpd = np.empty(N, dtype=float)
    for i in range(N):
        hist, _ = np.histogram(preds[:, i], bins=edges, density=True)  # density=True => integrates to 1
        # find bin for y[i]
        b = np.searchsorted(edges, y[i], side='right') - 1
        b = np.clip(b, 0, bins - 1)
        p = hist[b]  # density at that bin
        nlpd[i] = -np.log(p + eps)
    return nlpd


def get_summary(metric_outputs, y_t, model, desc, seed, training_time, epochs, beta_param_dict, x, region_interp, frac_train):
    """
    Generate summary statistics from model predictions and ground truth.

    Args:
        metric_outputs (tuple): Output from the `metrics` function (mean, std, nll, etc.)
        y_t (torch.Tensor): Ground truth target values
        model (torch.nn.Module): Model instance
        desc (str): Task/function description
        seed (int): Random seed used in experiment
        training_time (float): Total training time in seconds
        epochs (int): Total number of training epochs
        beta_param_dict (dict): Dictionary with beta scheduler information
        x (np.ndarray): Full x domain used in evaluation
        region_interp (tuple): Interpolation region bounds
        frac_train (float): Fraction of interpolation region used for training

    Returns:
        dict: Summary containing RMSE, NLL, and experiment metadata
    """
    mean_pred = metric_outputs[0]         # Predicted mean
    nlpd_per_sample = metric_outputs[-1]   # Per-sample NLL

    y_t_np = y_t.detach().cpu().numpy().squeeze()
    rmse = float(np.sqrt(np.mean((mean_pred - y_t_np) ** 2)))
    mean_nlpd = float(np.mean(nlpd_per_sample))

    return {
        "desc": desc,
        "model": model.__class__.__name__,
        "seed": seed,
        "rmse": rmse,
        "mean_nlpd": mean_nlpd,
        "training_time": training_time,
        "timestamp": datetime.now().isoformat(),
        "epochs": epochs,
        "beta_param_dict": beta_param_dict,
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "region_interp": region_interp,
        "frac_train": frac_train
    }


# OLD CODE =======================================================================================================

# def compute_rmse(mean, y_true):
#     """
#     Root Mean Squared Error
#     """
#     return np.sqrt(np.mean((mean - y_true) ** 2))

# def compute_nll(mean, y_true, std, eps=1e-6):
#     """
#     Negative Log Likelihood for Gaussian predictive distribution
#     NLL = 0.5 * log(2πσ^2) + (y - μ)^2 / (2σ^2)
#     """
#     std = np.maximum(std, eps)
#     nll = 0.5 * np.log(2 * np.pi * std**2) + ((y_true - mean) ** 2) / (2 * std**2)
#     return np.mean(nll)

# def metrics(preds, y, eps=1e-6):
#     """
#     Compute various per-sample metrics.
    
#     Args:
#         preds (numpy.ndarray): (n_samples, n_points, n_features) multiple stochastic predictions
#         y     (torch.Tensor) : (n_points, n_features)            true target output
#         eps   (float)        : ()                                small constant for numerical stability

#     Returns:
#         nll: (B,) tensor of per-sample NLLs
#     """
#     # Convert all arrays from torch tensors to numpy arrays
#     y_np = y.cpu().numpy().squeeze().astype(np.float64)
#     preds_np = preds.astype(np.float64)

#     # Mean, Variance, and Standard Deviation: shape = (n_points, 1)
#     mean = preds_np.mean(0)                
#     var = preds_np.var(0)              
#     std = preds_np.std(0)  

#     # Residual Precision: shape = (n_points, n_samples)           
#     res_prec = preds_np.squeeze().T - mean
#     # Residual Accuracy: shape = (n_points, n_samples)
#     res_acc = preds_np.squeeze().T - y_np.reshape(-1,1)

#     # Bias and MSE: shape = (n_points, 1)
#     bias = mean - y_np.reshape(-1,1)
#     mse = (res_acc**2).mean(1).reshape(-1,1)

#     # Bias-Variance Trade Off Difference: shape = (n_points, 1)
#     bias_var_diff = abs(mse - (var + bias**2))

#     return mean.squeeze(), var.squeeze(), std.squeeze(), res_prec, res_acc, bias.squeeze(), mse.squeeze(), bias_var_diff.squeeze()

