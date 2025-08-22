import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.kl_divergence import compute_kl_divergence

class BayesLayer(nn.Module):
    def __init__(self, input_dim, output_dim, prior_std=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Mean and log variance of weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(0, 0.1))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim).fill_(-3.0))


        self.bias_mu = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.1))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(output_dim).fill_(-3.0))

        self.prior_std = prior_std

    def forward(self, x, return_kl=False, sample=True):
        """
        Args:
            x: Input tensor of shape [B, input_dim]
            return_kl: Whether to return KL divergence term
            sample: Whether to sample weights and biases (True during training or uncertainty estimation)
        Returns:
            Prediction tensor (and optionally, KL divergence)
        """
        device = next(self.parameters()).device  
        x = x.to(device)
        
        if sample:
            weight_sigma = torch.exp(self.weight_log_sigma)
            bias_sigma = torch.exp(self.bias_log_sigma)

            eps_w = torch.randn_like(weight_sigma)
            eps_b = torch.randn_like(bias_sigma)

            weight = self.weight_mu + eps_w * weight_sigma
            bias = self.bias_mu + eps_b * bias_sigma
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        x = F.linear(x, weight, bias)

        if return_kl:
            kl_w = compute_kl_divergence(self.weight_mu, self.weight_log_sigma, reduction='sum')
            kl_b = compute_kl_divergence(self.bias_mu, self.bias_log_sigma, reduction='sum')
            return x, kl_w + kl_b
        else:
            return x


class BayesNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prior_std=1.0):
        super().__init__()
        self.fc1 = BayesLayer(input_dim, hidden_dim, prior_std)
        self.fc2 = BayesLayer(hidden_dim, output_dim, prior_std)
        
    def forward(self, x, return_kl=False, sample=True):
        """
        Args:
            x: Input tensor of shape [B, input_dim]
            return_kl: Whether to return KL divergence term
            sample: Whether to sample weights and biases (True during training or uncertainty estimation)
        Returns:
            Prediction tensor (and optionally, KL divergence)
        """
        device = next(self.fc1.parameters()).device  
        x = x.to(device)

        x, kl1 = self.fc1(x, return_kl=return_kl, sample=sample) if return_kl else (self.fc1(x, sample=sample), 0.0)
        x = F.relu(x)
        x, kl2 = self.fc2(x, return_kl=return_kl, sample=sample) if return_kl else (self.fc2(x, sample=sample), 0.0)
        
        if return_kl:
            return x, kl1 + kl2
        return x



##################################################################################################################


# class BayesLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, prior_std=1.0):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         # Mean and log variance of weights and biases
#         self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim).normal_(0, 0.1))
#         self.weight_log_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim).fill_(-3.0))

#         self.bias_mu = nn.Parameter(torch.Tensor(output_dim).normal_(0, 0.1))
#         self.bias_log_sigma = nn.Parameter(torch.Tensor(output_dim).fill_(-3.0))

#         self.prior_std = prior_std

#     def forward(self, x, return_kl=False):
#         if self.training:
#             weight_sigma = torch.exp(self.weight_log_sigma)
#             bias_sigma = torch.exp(self.bias_log_sigma)

#             eps_w = torch.randn_like(weight_sigma)
#             eps_b = torch.randn_like(bias_sigma)

#             weight = self.weight_mu + eps_w * weight_sigma
#             bias = self.bias_mu + eps_b * bias_sigma
#         else:
#             weight = self.weight_mu
#             bias = self.bias_mu

#         x = F.linear(x, weight, bias)

#         if return_kl:
#             kl_w = compute_kl_divergence(self.weight_mu, self.weight_log_sigma)
#             kl_b = compute_kl_divergence(self.bias_mu, self.bias_log_sigma)
#             return x, kl_w + kl_b
#         else:
#             return x

# class BayesNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, prior_std=1.0):
#         super().__init__()
#         self.fc1 = BayesLayer(input_dim, hidden_dim, prior_std)
#         self.fc2 = BayesLayer(hidden_dim, output_dim, prior_std)

#     def forward(self, x, return_kl=False):
#         if self.training:
#             x, kl1 = self.fc1(x, return_kl=True) if return_kl else (*self.fc1(x), 0.0)
#             x = F.relu(x)
#             x, kl2 = self.fc2(x, return_kl=True) if return_kl else (*self.fc2(x), 0.0)
#             if return_kl:
#                 return x.squeeze(-1), kl1 + kl2
#             return x.squeeze(-1)
#         else:
#             x = self.fc1(x)
#             x = F.relu(x)
#             x = self.fc2(x)
#             return x.squeeze(-1)
