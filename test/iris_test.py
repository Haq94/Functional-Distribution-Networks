import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---- Load and Prepare Iris Data ---- #
def prepare_iris_data(test_size=0.2):
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test

# ---- Uncertainty Visualization ---- #
def plot_uncertainty(model, X_test, y_test, num_samples=30):
    model.eval()
    device = next(model.parameters()).device

    softmax_outputs = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(X_test.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            softmax_outputs.append(probs)

    softmax_outputs = np.stack(softmax_outputs, axis=0)  # [samples, batch, classes]
    mean_probs = np.mean(softmax_outputs, axis=0)
    std_probs = np.std(softmax_outputs, axis=0)

    # Plot mean ± std bars for a few test points
    num_points = min(10, len(X_test))
    x_ticks = [f"Sample {i}" for i in range(num_points)]

    for class_idx in range(mean_probs.shape[1]):
        plt.errorbar(
            x=np.arange(num_points),
            y=mean_probs[:num_points, class_idx],
            yerr=std_probs[:num_points, class_idx],
            label=f"Class {class_idx}",
            capsize=4,
            marker='o'
        )

    plt.xticks(np.arange(num_points), x_ticks, rotation=45)
    plt.xlabel("Test Samples")
    plt.ylabel("Predicted Probability")
    plt.title("Predictive Uncertainty (Mean ± Std of Softmax)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# ---- Train FDN Classifier ---- #
def train_fdn_on_iris(model, epochs=500, lr=1e-3, print_every=50, plot=True, seed=None):

    X_train, X_test, y_train, y_test = prepare_iris_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        # ---- Training ---- #
        model.train()
        optimizer.zero_grad()
        logits = model(X_train.to(device))
        loss = criterion(logits, y_train.to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # ---- Validation ---- #
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test.to(device))
            test_loss = criterion(test_logits, y_test.to(device)).item()
            test_losses.append(test_loss)
            if epoch % print_every == 0:
                preds = torch.argmax(test_logits, dim=1)
                acc = accuracy_score(y_test.cpu(), preds.cpu())
                print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f}")

    # ---- Final Metrics ---- #
    print("\nFinal Evaluation:")
    test_logits = model(X_test.to(device))
    test_preds = torch.argmax(test_logits, dim=1)
    y_true = y_test.cpu().numpy()
    y_pred = test_preds.cpu().numpy()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=load_iris().target_names))
    report = classification_report(
        y_true, y_pred,
        target_names=load_iris().target_names,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=load_iris().target_names,
                yticklabels=load_iris().target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


    # ---- Precision and Recall Bar Plot ---- #

    # Extract precision and recall
    labels = load_iris().target_names
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, precision, width, label='Precision')
    plt.bar(x + width/2, recall, width, label='Recall')
    plt.xticks(x, labels)
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Per-Class Precision and Recall")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    # ---- Learning Curves ---- #
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Also show predictive uncertainty
        plot_uncertainty(model, X_test, y_test)


# ---- Main ---- #
if __name__ == "__main__":
    from models.fdnet import IC_FDNetwork, LP_FDNetwork  # adjust import path as needed
    from models.hypernet import HyperNetwork
    from models.bayesnet import BayesNetwork
    from models.gausshypernet import GaussianHyperNetwork
    input_dim = 4
    hidden_dim = 32
    output_dim = 3  # 3 Iris classes
    hyper_hidden_dim = 64

    seed = None
    if seed is not None:
        set_seed(seed=seed)

    network = 'LP_FDNet'

    if network == 'IC_FDNet':
        model = IC_FDNetwork(input_dim, hidden_dim, output_dim, hyper_hidden_dim)
    elif network == 'LP_FDNet':
        model = LP_FDNetwork(input_dim, hidden_dim, output_dim, hyper_hidden_dim)
    elif network == 'HyperNet':
        model = HyperNetwork(input_dim, hidden_dim, output_dim, hyper_hidden_dim)
    elif network == 'BayesNet':
        model = BayesNetwork(input_dim, hidden_dim, output_dim)
    elif network == 'GaussHyperNet':
        model = GaussianHyperNetwork(input_dim, hidden_dim, output_dim, hyper_hidden_dim)

    train_fdn_on_iris(model, seed=seed)


