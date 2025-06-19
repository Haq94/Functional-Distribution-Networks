# fdn_viz.py
import torch
from torch import nn
from torchviz import make_dot

# --- Your FDNLayer with weights + biases ---
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

    def forward(self, x):
        B = x.shape[0]
        h = self.hypernet(x)
        w_mu, w_log_sigma, b_mu, b_log_sigma = torch.split(
            h, [self.weight_param_dim, self.weight_param_dim, self.bias_param_dim, self.bias_param_dim], dim=-1)

        W = w_mu + torch.exp(w_log_sigma) * torch.randn_like(w_mu)
        b = b_mu + torch.exp(b_log_sigma) * torch.randn_like(b_mu)
        W = W.view(B, self.output_dim, self.input_dim)
        b = b.view(B, self.output_dim, 1)
        return W, b

# --- Your FDNNetwork ---
class FDNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hyper_hidden_dim):
        super().__init__()
        self.fdn1 = FDNLayer(input_dim, hidden_dim, hyper_hidden_dim)
        self.fdn2 = FDNLayer(hidden_dim, output_dim, hyper_hidden_dim)

    def forward(self, x):
        W1, b1 = self.fdn1(x)
        x = torch.bmm(W1, x.unsqueeze(-1)) + b1
        x = torch.relu(x)

        W2, b2 = self.fdn2(x.squeeze(-1))
        x = torch.bmm(W2, x) + b2
        return x.squeeze(-1)

# --- Run one batch and visualize ---
if __name__ == '__main__':
    input_dim = 4
    hidden_dim = 8
    output_dim = 1
    hyper_hidden_dim = 16
    batch_size = 2

    model = FDNNetwork(input_dim, hidden_dim, output_dim, hyper_hidden_dim)

    x = torch.randn(batch_size, input_dim, requires_grad=True)
    y_true = torch.randn(batch_size, output_dim)

    y_pred = model(x)
    loss = nn.MSELoss()(y_pred, y_true)

    # Visualize the graph
    dot = make_dot(loss, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render("fdn_computational_graph")

    print("Graph rendered to fdn_computational_graph.png")
