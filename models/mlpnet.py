import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class DeepEnsembleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_models=5):
        super().__init__()
        self.models = nn.ModuleList([
            MLPNetwork(input_dim, hidden_dim, output_dim)
            for _ in range(num_models)
        ])

    def forward(self, x):
        """
        Returns: (mean_prediction, std_prediction)
        """
        preds = torch.stack([model(x) for model in self.models], dim=0)  # [E, B]
        mean = preds.mean(dim=0)  # [B]
        std = preds.std(dim=0)    # [B]
        return mean

def train_deep_ensemble(ensemble, x_train, y_train, num_epochs=500, lr=1e-3):
    criterion = nn.MSELoss()
    for idx, model in enumerate(ensemble.models):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            model.train()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    # Example dimensions
    input_dim = 10
    hidden_dim = 64
    output_dim = 1
    num_ensemble = 5

    # Data
    x_train = torch.randn(32, input_dim)
    y_train = torch.sin(x_train[:, 0])  # Example target

    # Train and evaluate
    ensemble = DeepEnsembleNetwork(input_dim, hidden_dim, output_dim, num_models=num_ensemble)
    train_deep_ensemble(ensemble, x_train, y_train)

    ensemble.eval()
    with torch.no_grad():
        mean_pred, std_pred = ensemble(x_train)

