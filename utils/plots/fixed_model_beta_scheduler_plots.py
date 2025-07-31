import matplotlib.pyplot as plt

def plot_loss_vs_epoch(results, model_type):

    schedulers = set([r['beta_scheduler'] for r in results if r['model_type'] == model_type])
    for scheduler in schedulers:
        runs = [r for r in results if r['model_type'] == model_type and r['beta_scheduler'] == scheduler]
        all_losses = np.stack([r['losses_per_epoch'] for r in runs])
        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)
        plt.plot(mean_loss, label=scheduler)
        plt.fill_between(np.arange(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_type} - Loss vs Epoch")
    plt.legend()
    plt.show()

def plot_final_mse_by_scheduler(results, model_type):
    import matplotlib.pyplot as plt

    schedulers = sorted(set(r['beta_scheduler'] for r in results if r['model_type'] == model_type))
    data = [[r['final_mse'] for r in results if r['model_type'] == model_type and r['beta_scheduler'] == s] for s in schedulers]
    plt.boxplot(data, labels=schedulers)
    plt.ylabel("Final MSE")
    plt.title(f"{model_type} - Final MSE per Scheduler")
    plt.show()

def plot_mse_heatmap(results, model_type, scheduler):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    table = defaultdict(dict)
    for r in results:
        if r['model_type'] == model_type and r['beta_scheduler'] == scheduler:
            b, w = r['beta_max'], r['warmup_epochs']
            table[w][b] = table[w].get(b, []) + [r['final_mse']]

    warmups = sorted(table.keys())
    betas = sorted({b for w in table for b in table[w]})

    heatmap_data = np.array([
        [np.mean(table[w].get(b, [np.nan])) for b in betas]
        for w in warmups
    ])

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, xticklabels=np.round(betas, 2), yticklabels=warmups, cmap="viridis")
    plt.xlabel("beta_max")
    plt.ylabel("warmup_epochs")
    plt.title(f"{model_type} | {scheduler} - Final MSE Heatmap")
    plt.show()
