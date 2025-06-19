import torch
import torch.nn as nn
from utils.kl_divergence import compute_kl_divergence

class FDNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_param_dim = input_dim * output_dim
        self.bias_param_dim = output_dim

        total_output_dim = 2 * (self.weight_param_dim + self.bias_param_dim)

        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, total_output_dim)
        )

    def forward(self, x, return_kl=False, sample=True):
        """
        Args:
            x: Input tensor of shape [B, input_dim]
            return_kl: Whether to return KL divergence term
            sample: Whether to sample weights and biases (True during training or uncertainty estimation)
        Returns:
            Prediction tensor (and optionally, KL divergence)
        """
        device = next(self.hypernet.parameters()).device  
        x = x.to(device)

        B = x.shape[0]
        h = self.hypernet(x)

        split_sizes = [self.weight_param_dim, self.weight_param_dim,
                       self.bias_param_dim, self.bias_param_dim]
        
        if sample:
            w_mu, w_log_sigma, b_mu, b_log_sigma = torch.split(h, split_sizes, dim=-1)

            # Reparameterization
            w_sigma = torch.exp(w_log_sigma)
            b_sigma = torch.exp(b_log_sigma)

            W = w_mu + w_sigma * torch.randn_like(w_mu, device=w_mu.device)
            b = b_mu + b_sigma * torch.randn_like(b_mu, device=b_mu.device)
        else:
            w_mu, _, b_mu, _ = torch.split(h, split_sizes, dim=-1)

            W = w_mu
            b = b_mu

        W = W.view(B, self.output_dim, self.input_dim)
        b = b.view(B, self.output_dim, 1)

        if return_kl:
            kl_w = compute_kl_divergence(w_mu, w_log_sigma)
            kl_b = compute_kl_divergence(b_mu, b_log_sigma)
            return W, b, kl_w + kl_b
        else:
            return W, b

class FDNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper_hidden_dim):
        super().__init__()
        self.fdn1 = FDNLayer(input_dim, hidden_dim, hyper_hidden_dim)
        self.fdn2 = FDNLayer(hidden_dim, output_dim, hyper_hidden_dim)

    def forward(self, x, return_kl=False, sample=True):
        """
        Args:
            x: Input tensor of shape [B, input_dim]
            return_kl: Whether to return KL divergence term
            sample: Whether to sample weights and biases (True during training or uncertainty estimation)
        Returns:
            Prediction tensor (and optionally, KL divergence)
        """
        device = next(self.fdn1.hypernet.parameters()).device  # Or fdn2
        x = x.to(device)

        W1, b1, kl1 = self.fdn1(x, return_kl=return_kl, sample=sample) if return_kl else (*self.fdn1(x, sample=sample), 0.0)
        x = torch.bmm(W1, x.unsqueeze(-1)) + b1
        x = torch.relu(x)

        W2, b2, kl2 = self.fdn2(x.squeeze(-1), return_kl=return_kl, sample=sample) if return_kl else (*self.fdn2(x.squeeze(-1), sample=sample), 0.0)
        x = torch.bmm(W2, x) + b2

        if return_kl:
            return x.squeeze(-1), kl1 + kl2
        return x.squeeze(-1)


# class FDNLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, hyper_hidden_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight_param_dim = input_dim * output_dim
#         self.bias_param_dim = output_dim

#         # Total outputs: [mu_W, log_sigma_W, mu_b, log_sigma_b]
#         total_output_dim = 2 * (self.weight_param_dim + self.bias_param_dim)

#         self.hypernet = nn.Sequential(
#             nn.Linear(input_dim, hyper_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hyper_hidden_dim, total_output_dim)
#         )

#     def forward(self, x):
#         B = x.shape[0]  # batch size
#         h = self.hypernet(x)  # shape: [B, total_output_dim]

#         # Split into mean/log_sigma for weights and biases
#         split_sizes = [self.weight_param_dim, self.weight_param_dim,
#                        self.bias_param_dim, self.bias_param_dim]
#         w_mu, w_log_sigma, b_mu, b_log_sigma = torch.split(h, split_sizes, dim=-1)

#         # Sample weights and biases using reparameterization
#         w_sigma = torch.exp(w_log_sigma)
#         b_sigma = torch.exp(b_log_sigma)

#         W = w_mu + w_sigma * torch.randn_like(w_mu)  # shape: [B, out_dim * in_dim]
#         b = b_mu + b_sigma * torch.randn_like(b_mu)  # shape: [B, out_dim]

#         # Reshape for matrix multiplication
#         W = W.view(B, self.output_dim, self.input_dim)
#         b = b.view(B, self.output_dim, 1)

#         return W, b


# class FDNNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, hyper_hidden_dim):
#         super().__init__()
#         self.fdn1 = FDNLayer(input_dim, hidden_dim, hyper_hidden_dim)
#         self.fdn2 = FDNLayer(hidden_dim, output_dim, hyper_hidden_dim)

#     def forward(self, x):
#         # x shape: [B, input_dim]

#         W1, b1 = self.fdn1(x)
#         x = torch.bmm(W1, x.unsqueeze(-1)) + b1
#         x = torch.relu(x)
#         W2, b2 = self.fdn2(x.squeeze(-1))
#         x = torch.bmm(W2, x) + b2
#         return x.squeeze(-1)

