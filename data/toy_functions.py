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

def generate_single_task_splits(
    x_min: float,
    x_max: float,
    region_interp: tuple,
    n_train: int = 50,
    n_test: int = 200,
    n_val: int = 20,
    val_frac_interp: float = 0.5,
    seed: int = None
):
    """
    Generate 1D regression train/val/test splits.

    Args:
        x_min, x_max (float): Global domain bounds for test grid.
        region_interp (tuple): (interp_min, interp_max), the interpolation region.
        n_train (int): Number of training points sampled uniformly inside interpolation region.
        n_test (int): Number of test points on uniform grid from x_min to x_max.
        n_val (int): Number of validation points.
        val_frac_interp (float): Fraction of val points from interpolation region (rest from extrapolation).
        seed (int or None): Random seed for reproducibility.

    Returns:
        dict with:
            - x_train: (n_train,) training inputs
            - x_val: (n_val,) validation inputs
            - x_test: (n_test,) uniform test grid
            - region_interp: (interp_min, interp_max)
            - ind_interp: indices of x_test inside interpolation region
            - ind_extrap: indices of x_test outside interpolation region
    """
    rng = np.random.default_rng(seed)

    interp_min, interp_max = region_interp

    # --- Train ---
    x_train = rng.uniform(interp_min, interp_max, size=n_train)

    # --- Test (full grid) ---
    x_test = np.linspace(x_min, x_max, n_test)

    # --- Interp / Extrap masks on test grid ---
    ind_interp = (x_test >= interp_min) & (x_test <= interp_max)
    ind_extrap = ~ind_interp

    # --- Validation ---
    n_val_interp = int(round(val_frac_interp * n_val))
    n_val_extrap = n_val - n_val_interp

    x_val_interp = rng.uniform(interp_min, interp_max, size=n_val_interp) if n_val_interp > 0 else np.array([])
    x_val_extrap = rng.uniform(x_min, x_max, size=n_val_extrap)
    # Ensure extrap only outside interp region
    if n_val_extrap > 0:
        mask = (x_val_extrap < interp_min) | (x_val_extrap > interp_max)
        while not np.all(mask):  # resample until all are out of interp
            resample = rng.uniform(x_min, x_max, size=mask.sum())
            x_val_extrap[~mask] = resample
            mask = (x_val_extrap < interp_min) | (x_val_extrap > interp_max)

    x_val = np.concatenate([x_val_interp, x_val_extrap]) if n_val > 0 else np.array([])

    # --- Bundle ---
    return {
        "x_train": np.sort(x_train),
        "x_val": np.sort(x_val),
        "x_test": x_test,
        "region_interp": (interp_min, interp_max),
        "ind_interp": np.where(ind_interp)[0],
        "ind_extrap": np.where(ind_extrap)[0],
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


