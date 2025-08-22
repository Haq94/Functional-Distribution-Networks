# Models/hypernet.py

import torch
import torch.nn as nn

class HyperLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Total number of parameters in target layer: weights + biases
        self.weight_dim = output_dim * input_dim
        self.bias_dim = output_dim
        self.total_dim = self.weight_dim + self.bias_dim

        # Input-conditioned hypernetwork
        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, self.total_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, input_dim)
        Returns:
            W: (B, output_dim, input_dim)
            b: (B, output_dim, 1)
        """
        B = x.shape[0]
        params = self.hypernet(x)  # (B, total_dim)
        W_flat = params[:, :self.weight_dim]
        b_flat = params[:, self.weight_dim:]

        W = W_flat.view(B, self.output_dim, self.input_dim)
        b = b_flat.view(B, self.output_dim, 1)
        return W, b

class HyperNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper_hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hyper1 = HyperLayer(input_dim, hidden_dim, hyper_hidden_dim)
        self.hyper2 = HyperLayer(hidden_dim, output_dim, hyper_hidden_dim)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim)
        Returns:
            y: (B, output_dim)
        """

        # # If x is not 3D, try to reshape it appropriately
        # if x.dim() == 2:
        #     x = x.unsqueeze(-1)
        # elif x.dim() == 1:
        #     x = x.unsqueeze(0).unsqueeze(-1)
        # elif x.dim() != 3:
        #     raise ValueError(f"Unexpected shape for x: {x.shape}")

        W1, b1 = self.hyper1(x)
        x = torch.bmm(W1, x.unsqueeze(-1)) + b1  # (B, H, 1)
        x = torch.relu(x)
        W2, b2 = self.hyper2(x.squeeze(-1))
        x = torch.bmm(W2, x) + b2  # (B, O, 1)
        return x.squeeze(-1)  # (B, O)
