import torch

def compute_kl_divergence(mu, log_sigma, prior_mu=0.0, prior_sigma=1.0):
    """
    Compute KL divergence between N(mu, sigma^2) and N(prior_mu, prior_sigma^2)
    
    Args:
        mu (torch.Tensor): Mean of the approximate posterior
        log_sigma (torch.Tensor): Log of standard deviation (not log variance)
        prior_mu (float): Mean of the prior
        prior_sigma (float): Std dev of the prior
        
    Returns:
        kl (torch.Tensor): Scalar KL divergence
    """
    sigma = torch.exp(log_sigma)
    kl = (
        torch.log(prior_sigma / sigma)
        + (sigma**2 + (mu - prior_mu) ** 2) / (2 * prior_sigma**2)
        - 0.5
    )
    return kl.sum()
