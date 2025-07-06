import torch

def compute_kl_divergence(mu, log_sigma, prior_mu=0.0, prior_sigma=1.0, reduction='mean'):
    """
    Compute total KL divergence between N(mu, sigma^2) and N(prior_mu, prior_sigma^2)

    Args:
        mu (Tensor): shape (B, D), posterior mean
        log_sigma (Tensor): shape (B, D), log std dev of posterior
        prior_mu (float): prior mean (scalar or tensor broadcastable to mu)
        prior_sigma (float): prior std dev
        reduction (str): 'mean' (default), 'sum', or 'none'

    Returns:
        kl (Tensor): scalar KL loss (or per-sample vector if reduction='none')
    """
    sigma = torch.exp(log_sigma)
    sigma_sq = sigma ** 2
    prior_sigma_sq = prior_sigma ** 2

    kl = (
        torch.log(prior_sigma / sigma)
        + (sigma_sq + (mu - prior_mu) ** 2) / (2 * prior_sigma_sq)
        - 0.5
    )  # shape: (B, D)

    if kl.dim() > 1:
        kl = kl.sum(dim=1)  # shape: (B,)

    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl  # shape: (B,)

# OLD CODE========================================================================================================

# import torch

# def compute_kl_divergence(mu, log_sigma, prior_mu=0.0, prior_sigma=1.0):
#     """
#     Compute KL divergence between N(mu, sigma^2) and N(prior_mu, prior_sigma^2)
    
#     Args:
#         mu (torch.Tensor): Mean of the approximate posterior
#         log_sigma (torch.Tensor): Log of standard deviation (not log variance)
#         prior_mu (float): Mean of the prior
#         prior_sigma (float): Std dev of the prior
        
#     Returns:
#         kl (torch.Tensor): Scalar KL divergence
#     """
#     sigma = torch.exp(log_sigma)
#     kl = (
#         torch.log(prior_sigma / sigma)
#         + (sigma**2 + (mu - prior_mu) ** 2) / (2 * prior_sigma**2)
#         - 0.5
#     )
#     return kl.sum()
