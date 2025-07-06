import numpy as np

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
        nll:        (n_points,) negative log likelihood per sample
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
    nll = 0.5 * (np.log(2 * np.pi * var.squeeze()) + ((y - mean.squeeze()) ** 2) / var.squeeze())

    return (
        mean.squeeze(),
        var.squeeze(),
        std.squeeze(),
        res_prec,
        res_acc,
        bias.squeeze(),
        mse.squeeze(),
        bias_var_diff.squeeze(),
        nll.squeeze()
    )

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

