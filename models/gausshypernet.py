import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.kl_divergence import compute_kl_divergence

class GaussianHyperLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_hidden_dim, latent_dim=10, prior_std=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_std = prior_std

        self.weight_dim = input_dim * output_dim
        self.bias_dim = output_dim

        # The hypernetwork generates μ and log σ for weights and biases
        self.hypernet = nn.Sequential(
            nn.Linear(latent_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, 2 * (self.weight_dim + self.bias_dim))  # [w_mu, w_logσ, b_mu, b_logσ]
        )

        self.latent = nn.Parameter(torch.randn(latent_dim))  # Fixed latent vector

    def forward(self, x, return_kl=False, sample=True):
        h = self.hypernet(self.latent)

        split_sizes = [self.weight_dim, self.weight_dim, self.bias_dim, self.bias_dim]
        w_mu, w_log_sigma, b_mu, b_log_sigma = torch.split(h, split_sizes, dim=-1)

        if sample:
            w_sigma = torch.exp(w_log_sigma)
            b_sigma = torch.exp(b_log_sigma)

            W = w_mu + w_sigma * torch.randn_like(w_mu)
            b = b_mu + b_sigma * torch.randn_like(b_mu)
        else:
            W = w_mu
            b = b_mu

        W = W.view(self.output_dim, self.input_dim)
        b = b.view(self.output_dim)

        out = F.linear(x, W, b)

        if return_kl:
            kl_w = compute_kl_divergence(w_mu, w_log_sigma)
            kl_b = compute_kl_divergence(b_mu, b_log_sigma)
            return out, kl_w + kl_b
        else:
            return out

class GaussianHyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper_hidden_dim, latent_dim=10, prior_std=1.0):
        super().__init__()
        self.layer1 = GaussianHyperLayer(input_dim, hidden_dim, hyper_hidden_dim, latent_dim, prior_std)
        self.layer2 = GaussianHyperLayer(hidden_dim, output_dim, hyper_hidden_dim, latent_dim, prior_std)

    def forward(self, x, return_kl=False, sample=True):
        x, kl1 = self.layer1(x, return_kl=return_kl, sample=sample) if return_kl else (self.layer1(x, sample=sample), 0.0)
        x = F.relu(x)
        x, kl2 = self.layer2(x, return_kl=return_kl, sample=sample) if return_kl else (self.layer2(x, sample=sample), 0.0)

        if return_kl:
            return x, kl1 + kl2
        return x
