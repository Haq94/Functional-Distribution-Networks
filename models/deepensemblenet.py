import torch
import torch.nn as nn

class DeepEnsembleNetwork(nn.Module):
    def __init__(self, network_class, num_models=5, seed_list=None, *net_args, **net_kwargs):
        """
        Args:
            network_class: Class of the network to ensemble (e.g., DeterministicMLPNetwork)
            num_models: Number of ensemble members
            seed_list: Optional list of seeds, one per model (length == num_models)
            *net_args: Positional args for each network constructor
            **net_kwargs: Keyword args for each network constructor
        """
        super().__init__()

        if seed_list is not None and len(seed_list) != num_models:
            print('seed_list is no the same length as num_models. Appending seed_list')
            


        if seed_list is not None and len(seed_list) != num_models:
            raise ValueError("seed_list must be the same length as num_models")

        self.models = nn.ModuleList()
        for i in range(num_models):
            if seed_list is not None:
                torch.manual_seed(seed_list[i])
            self.models.append(network_class(*net_args, **net_kwargs))


    def forward(self, x, mc_samples=1):
        preds = []
        for model in self.models:
            if mc_samples>1:
                # Monte Carlo sampling if model has stochasticity (e.g., Dropout)
                single_model_preds = [model(x) for _ in range(mc_samples)]
                preds.extend(single_model_preds)
            else:
                preds.append(model(x))
        
        preds = torch.stack(preds, dim=0)  # [E * S, B]

        return preds

        # return preds.mean(dim=0), preds.std(dim=0)


if __name__=='__main__':
    from models.mlpnet import DeterministicMLPNetwork
    # Example dimensions
    input_dim = 10
    hidden_dim = 64
    output_dim = 1
    num_models = 5

    mc_samples = 3
    num_epochs = 100
    lr=0.01

    # Data
    x_train = torch.randn(32, input_dim)
    y_train = torch.sin(x_train[:, 0])  # Example target
    seed_list = [0, 1, 2, 3, 4]
    ensemble = DeepEnsembleNetwork(network_class=DeterministicMLPNetwork, num_models=num_models, seed_list=seed_list, input_dim=10, hidden_dim=64, output_dim=1, dropout_rate=0.1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
    for epoch in range(num_epochs):
        ensemble.train()
        pred, _ = ensemble(x_train, mc_samples=mc_samples)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # MC Dropout enabled during eval
    ensemble.eval()
    with torch.no_grad():
        mean, std = ensemble(x_train, mc_samples=mc_samples)