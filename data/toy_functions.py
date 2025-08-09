import random
import numpy as np
import torch

def sample_function(seed=None):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    fn_type = np.random.choice(["sine", "quadratic", "step"])
    
    if fn_type == "sine":
        amp = np.random.uniform(0.5, 2.0)
        freq = np.random.uniform(1.0, 3.0)
        return lambda x: amp * np.sin(2*np.pi*freq * x), f"sine (amp={amp:.2f}, freq={freq:.2f})"

    elif fn_type == "quadratic":
        a = np.random.uniform(-1.0, 1.0)
        b = np.random.uniform(-1.0, 1.0)
        return lambda x: a * x**2 + b, f"quad (a={a:.2f}, b={b:.2f})"

    else: 
        return lambda x: np.where(x > 0, 1.0, -1.0), "step"

def generate_meta_task(n_context=10, n_target=10, 
                       x_range=None, 
                       x_context_range=(-3, -1), 
                       x_target_range=(1, 3),
                       seed=None):
    f, desc = sample_function(seed=seed)

    if x_range is not None:
        # Sample context + target together (used for joint tasks)
        x_all = np.random.uniform(*x_range, size=(n_context + n_target, 1))
        y_all = f(x_all)
        x_context = torch.tensor(x_all[:n_context], dtype=torch.float32)
        y_context = torch.tensor(y_all[:n_context], dtype=torch.float32)
        x_target = torch.tensor(x_all[n_context:], dtype=torch.float32)
        y_target = torch.tensor(y_all[n_context:], dtype=torch.float32)
    else:
        # Sample context and target separately from disjoint domains
        x_context = np.random.uniform(*x_context_range, size=(n_context, 1))
        x_target = np.random.uniform(*x_target_range, size=(n_target, 1))
        y_context = f(x_context)
        y_target = f(x_target)

        x_context = torch.tensor(x_context, dtype=torch.float32)
        y_context = torch.tensor(y_context, dtype=torch.float32)
        x_target = torch.tensor(x_target, dtype=torch.float32)
        y_target = torch.tensor(y_target, dtype=torch.float32)

    # return x_context.squeeze(), y_context.squeeze(), x_target.squeeze(), y_target.squeeze(), desc
    return x_context, y_context, x_target, y_target, desc

def generate_grid(input_type=None, input_seed=0, x_min=-10, x_max=10, region_interp=(-1,1), n_interp=10, n_extrap=100):
    if input_type == "uniform_random":
        np.random.seed(input_seed)
        x_l = np.random.uniform(low=x_min,high=region_interp[0],size=round(n_extrap/2)) 
        x_c = np.random.uniform(low=region_interp[0],high=region_interp[1],size=n_interp)
        x_r = np.random.uniform(low=region_interp[1],high=x_max,size=n_extrap-round(n_extrap/2))
    else:
        x_l = np.linspace(start=x_min,stop=region_interp[0],num=round(n_extrap/2))
        x_c = np.linspace(start=region_interp[0],stop=region_interp[1],num=n_interp+2)
        x_r = np.linspace(start=region_interp[1],stop=x_max,num=n_extrap-round(n_extrap/2))
    return np.sort(np.unique(np.concatenate([x_l, x_c, x_r])))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Use disjoint context/target domains for testing extrapolation
    x_c, y_c, x_t, y_t, desc = generate_meta_task(n_context=10, n_target=10)

    # Combine and sort for visualization
    x_plot = torch.cat([x_c, x_t]).squeeze().numpy()
    y_plot = torch.cat([y_c, y_t]).squeeze().numpy()
    idx = np.argsort(x_plot)

    plt.plot(x_plot[idx], y_plot[idx], label="ground truth")
    plt.scatter(x_c, y_c, color="red", label="context")
    plt.title(f"Sampled Function: {desc}")
    plt.legend()
    plt.grid(True)
    plt.show()


