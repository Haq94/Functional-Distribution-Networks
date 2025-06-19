import torch
from models.fdnet import FDNNetwork
import matplotlib.pyplot as plt

input_dim = 1
hidden_dim = 16
output_dim = 1
hyper_hidden_dim = 32
B = 4

model = FDNNetwork(input_dim, hidden_dim, output_dim, hyper_hidden_dim)
x = torch.linspace(-2, 2, B).unsqueeze(1)
y_samples = [model(x).detach().numpy() for _ in range(10)]

for i in range(B):
    plt.plot([s[i, 0] for s in y_samples], label=f"x={x[i].item():.2f}")
plt.title("Variation Across Samples Per Input")
plt.xlabel("Sample Index")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
