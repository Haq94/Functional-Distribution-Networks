import torch.nn as nn
import torch.nn.functional as F

class MLPDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x, mc_dropout=False):
        # Force dropout active if mc_dropout=True
        if mc_dropout:
            self.train()
        else:
            self.eval()
        return self.net(x)
