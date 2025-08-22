import numpy as np
from datetime import datetime

def metrics(preds, y, eps=1e-6, nlpd_type='kde'):
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

    # Gaussian NLPD per sample (n_points,)
    nlpd_kde = per_x_nlpd_from_samples_kde(samples=preds, y_true=y)
    nlpd_hist = per_x_nlpd_from_samples_hist(samples=preds, y_true=y, bins=50, y_min=preds.min(), y_max=preds.max())

    # PDF (n_points, bins)
    grid_kde, pdf_kde = per_x_pdf_kde(samples=preds, y_min=preds.min(), y_max=preds.max())
    _, grid_hist, pdf_hist = per_x_pdf_hist(samples=preds,y_min=preds.min(), y_max=preds.max())

    # CRPS
    crps = energy_score_from_samples_np(samples=preds, y_true=y)

    # Metric dict

    metric_dict = {
        'mean': mean.squeeze(),
        'var': var.squeeze(),
        'std': std.squeeze(),
        'res_prec': res_prec,
        'res_acc': res_acc,
        'bias': bias.squeeze(),
        'mse': mse.squeeze(),
        'bias_var_diff': bias_var_diff.squeeze(),
        'nlpd_kde': nlpd_kde.squeeze(),
        'grid_kde': grid_kde,
        'nlpd_hist': nlpd_hist.squeeze(),
        'grid_hist': grid_hist,
        'pdf_kde': pdf_kde,
        'pdf_hist': pdf_hist,
        'crps': crps
    }

    return metric_dict

def energy_score_from_samples_np(samples, y_true, beta=1.0):
    """
    Multivariate generalization of CRPS (beta=1 → CRPS in 1D with L2 norm).

    Args
    ----
    samples : np.ndarray, shape (S, N, D) or (S, N)  (D>=1)
    y_true  : np.ndarray, shape (N, D) or (N,)       (matches D)
    beta    : float in (0,2]
    reduction : 'mean' | 'none'

    Returns
    -------
    es : float or np.ndarray (N,)
    """
    s = np.asarray(samples)
    if s.ndim == 2:
        s = s[..., None]  # (S,N,1)
    S, N, D = s.shape

    y = np.asarray(y_true)
    if y.ndim == 1:
        y = y[:, None]    # (N,1)
    assert y.shape == (N, D)

    # term1: E||X - y||^beta
    diff1 = np.linalg.norm(s - y[None, :, :], axis=-1)   # (S,N)
    term1 = np.mean(diff1**beta, axis=0)                 # (N,)

    # term2: E||X - X'||^beta   (O(S^2); fine for moderate S)
    diff2 = s[:, None, :, :] - s[None, :, :, :]          # (S,S,N,D)
    norm2 = np.linalg.norm(diff2, axis=-1)               # (S,S,N)
    term2 = np.mean(norm2**beta, axis=(0, 1))            # (N,)

    es = term1 - 0.5 * term2                             # (N,)
    return es

def per_x_pdf_kde(samples, M=200, y_min=None, y_max=None, min_band=1e-6):
    """
    KDE-based per-x PDF on a common grid.
    samples: (S,N[,1]) predictive samples
    returns: grid (M,), pdf (N,M)
    """
    s = np.asarray(np.squeeze(samples), dtype=np.float64)  # (S,N)
    S, N = s.shape
    if y_min is None: y_min = float(s.min())
    if y_max is None: y_max = float(s.max())
    grid = np.linspace(y_min, y_max, M)

    std = s.std(axis=0, ddof=1) + 1e-12
    h = 1.06 * std * (S ** (-1/5))
    h = np.maximum(h, min_band)

    pdf = np.empty((N, M), dtype=np.float64)
    inv_norm = 1.0 / np.sqrt(2*np.pi)
    for i in range(N):
        z = (grid[None, :] - s[:, i][:, None]) / h[i]     # (S,M)
        phi = np.exp(-0.5 * z*z) * (inv_norm / h[i])      # (S,M)
        pdf[i, :] = phi.mean(axis=0)                      # (M,)
    return grid, pdf


def per_x_pdf_hist(samples, bins=50, y_min=None, y_max=None, alpha=0.5):
    """
    Histogram-based per-x PDF (piecewise-constant).
    samples: (S,N[,1])
    returns: edges (bins+1,), centers (bins,), pdf (N,bins)
    """
    s = np.asarray(np.squeeze(samples), dtype=np.float64)  # (S,N)
    S, N = s.shape
    if y_min is None: y_min = float(s.min())
    if y_max is None: y_max = float(s.max())
    edges = np.linspace(y_min, y_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dy = edges[1] - edges[0]

    pdf = np.empty((N, bins), dtype=np.float64)
    for i in range(N):
        counts, _ = np.histogram(s[:, i], bins=edges, density=False)
        counts = counts.astype(np.float64) + alpha
        total = counts.sum()
        pdf[i, :] = counts / (total * dy)   # integrates to 1
    return edges, centers, pdf


def per_x_nlpd_from_samples_kde(samples, y_true, min_band=1e-6):
    """
    KDE-based per-x NLPD.
    samples: (S,N) torch/np — S predictive samples per x_i
    y_true : (N,) torch/np
    returns: (N,) array of -log p(y_i | KDE_i)
    """
    s = np.asarray(np.squeeze(samples), dtype=np.float64)
    S, N = s.shape
    y = np.asarray(y_true,  dtype=np.float64)

    # Silverman bandwidth per x_i; clamp for stability
    std = s.std(axis=0, ddof=1) + 1e-12
    h = 1.06 * std * (S ** (-1/5))
    h = np.maximum(h, min_band)

    # log p(y_i) = logsumexp_j [ -0.5 ((y_i - s_ji)/h_i)^2 - log(sqrt(2π) h_i) ] - log S
    z = (y[None, :] - s) / h[None, :]
    log_phi = -0.5 * z*z - np.log(np.sqrt(2*np.pi) * h[None, :])  # (S,N)
    m = log_phi.max(axis=0)                                        # (N,)
    log_p = m + np.log(np.exp(log_phi - m).mean(axis=0)) - np.log(S)
    return -log_p  # (N,)


def per_x_nlpd_from_samples_hist(samples, y_true, bins=50, y_min=None, y_max=None,
                                 alpha=0.5, eps=1e-12):
    """
    Histogram-based per-x NLPD with Laplace/Jeffreys smoothing.
    samples: (S,N) torch/np
    y_true : (N,) torch/np
    alpha  : pseudo-count per bin (alpha=0.5 is Jeffreys; alpha=1 is Laplace)
    returns: (N,) array of -log p(y_i) where p is piecewise-constant histogram density.
    """
    S, N, _ = samples.shape
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y_true,  dtype=np.float64)

    # Global edges so bins are comparable across x (simple & consistent)
    if y_min is None: y_min = float(s.min())
    if y_max is None: y_max = float(s.max())
    edges = np.linspace(y_min, y_max, bins + 1)
    dy = edges[1] - edges[0]

    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        # counts (no density yet) to allow pseudo-count smoothing
        counts, _ = np.histogram(s[:, i], bins=edges, density=False)
        counts = counts.astype(np.float64) + alpha           # smooth
        total = counts.sum()
        density = counts / (total * dy)                      # integrate to 1

        # bin index for y_i
        b = np.searchsorted(edges, y[i], side='right') - 1
        b = np.clip(b, 0, bins - 1)
        p = density[b]
        out[i] = -np.log(p + eps)
    return out  # (N,)

# def per_x_nlpd_from_samples_hist(preds, y, bins=50, y_min=None, y_max=None, eps=1e-12):
#     """
#     preds: (S, N, 1) MC samples
#     y    : (N,)   ground-truth
#     returns: (N,) per-x NLPD using histogram density lookup
#     """
#     S, N, _ = preds.shape
#     if y_min is None: y_min = float(preds.min())
#     if y_max is None: y_max = float(preds.max())
#     edges = np.linspace(y_min, y_max, bins + 1)
#     dy = edges[1] - edges[0]

#     nlpd = np.empty(N, dtype=float)
#     for i in range(N):
#         hist, _ = np.histogram(preds[:, i], bins=edges, density=True)  # density=True => integrates to 1
#         # find bin for y[i]
#         b = np.searchsorted(edges, y[i], side='right') - 1
#         b = np.clip(b, 0, bins - 1)
#         p = hist[b]  # density at that bin
#         nlpd[i] = -np.log(p + eps)
#     return nlpd


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
    mean_pred = metric_outputs['mean']        # Predicted mean
    nlpd_per_sample = metric_outputs['nlpd_hist']   # Per-sample NLL

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

