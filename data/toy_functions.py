import random
import numpy as np
import torch

def sample_function(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    fn_type = np.random.choice(["sine", "quadratic", "step"])
    
    if fn_type == "sine":
        amp = np.random.uniform(0.5, 2.0)
        freq = np.random.uniform(1.0/0.6, 1.0/0.3)
        # freq = 1.0/0.6
        return lambda x: amp * np.sin(2*np.pi*freq * x), f"sine (amp={amp:.2f}, freq={freq:.2f})"

    elif fn_type == "quadratic":
        a = np.random.uniform(-1.0, 1.0)
        b = np.random.uniform(-1.0, 1.0)
        return lambda x: a * x**2 + b, f"quad (a={a:.2f}, b={b:.2f})"

    else: 
        return lambda x: np.where(x > 0, 1.0, -1.0), "step"

def generate_meta_task(
    n_train=10, n_val=5, n_test=10,
    x_range=None,
    x_train_range=(-3, -1),
    x_val_range=(-2, -1),
    x_test_range=(1, 3),
    seed=None,
    dtype=torch.float64,   # match your .double() training
    device=None
):
    rng = np.random.default_rng(seed)
    f, desc = sample_function(seed=seed)  # keep your existing behavior

    def _np_to_torch(a):
        # ensure 2D (n,1), then to desired dtype/device
        a = np.asarray(a).reshape(-1, 1)
        return torch.as_tensor(a, dtype=dtype, device=device)

    if x_range is not None:
        # Sample context + target together (joint tasks)
        x_all = rng.uniform(*x_range, size=(n_train + n_test + n_val, 1))
        y_all = f(x_all)
        x_train = _np_to_torch(x_all[:n_train])
        y_train = _np_to_torch(y_all[:n_train])
        x_val   = _np_to_torch(x_all[n_train:n_train+n_val])
        y_val   = _np_to_torch(y_all[n_train:n_train+n_val])
        x_test  = _np_to_torch(x_all[n_train+n_val:])
        y_test  = _np_to_torch(y_all[n_train+n_val:])
    else:
        # Disjoint domains
        x_train_np = rng.uniform(*x_train_range, size=(n_train, 1))
        x_val_np   = rng.uniform(*x_val_range,   size=(n_val,   1))
        x_test_np  = rng.uniform(*x_test_range,  size=(n_test,  1))

        y_train_np = f(x_train_np)
        y_val_np   = f(x_val_np)
        y_test_np  = f(x_test_np)

        x_train = _np_to_torch(x_train_np)
        y_train = _np_to_torch(y_train_np)
        x_val   = _np_to_torch(x_val_np)
        y_val   = _np_to_torch(y_val_np)
        x_test  = _np_to_torch(x_test_np)
        y_test  = _np_to_torch(y_test_np)

    return x_train, y_train, x_val, y_val, x_test, y_test, desc

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

import numpy as np

def generate_splits(x_min=-10, x_max=10, region_interp=(-1,1), 
                    n_train=30, n_test=10, n_val_interp=5, n_val_extrap=5, seed=None):
    """
    Generate training, validation, and test points.

    Args:
        x_min, x_max (float): Bounds of the full interval.
        region_interp (tuple): (a, b) defining interpolation region.
        n_train (int): Number of training points (uniform random across full interval).
        n_test (int): Number of test points (uniform grid across full interval).
        n_val_interp (int): Number of validation points in interpolation region.
        n_val_extrap (int): Number of validation points outside interpolation region.
        seed (int, optional): Random seed.

    Returns:
        dict with keys 'train', 'val_interp', 'val_extrap', 'test'.
    """
    rng = np.random.default_rng(seed)

    # Interp region
    a, b = region_interp

    # Training: uniform samples across full interval
    x_train = np.sort(rng.uniform(a, b, size=(n_train, 1)),axis=0)

    # Test: uniform grid across full interval
    x_test = np.linspace(x_min, x_max, n_test).reshape(-1, 1)

    # Validation - interp region
    x_val_interp = rng.uniform(a, b, size=(n_val_interp, 1))

    # Validation - extrap region (split across left and right)
    n_left = n_val_extrap // 2
    n_right = n_val_extrap - n_left
    x_val_left = rng.uniform(x_min, a, size=(n_left, 1)) if n_left > 0 else np.empty((0,1))
    x_val_right = rng.uniform(b, x_max, size=(n_right, 1)) if n_right > 0 else np.empty((0,1))
    x_val_extrap = np.vstack([x_val_left, x_val_right])
    x_val = np.sort(np.concatenate([x_val_extrap, x_val_interp]),axis=0)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "region": (x_min, x_max),
        "region_interp": region_interp
    }



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Use disjoint context/target domains for testing extrapolation
    x_c, y_c, x_t, y_t, desc = generate_meta_task(n_train=10, n_test=10)

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


