import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------------
# Generic helpers
# ------------------------------

def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0):
    """
    Simple random split of X, y into train/val/test.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    # Remaining goes to test
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx],  y[val_idx],
        X[test_idx], y[test_idx],
    )


def standardize_from_train(X_train, X_val, X_test,
                           y_train, y_val, y_test):
    """
    Z-score standardize features and targets using TRAIN stats only.
    Returns standardized arrays and (means, stds) for X, y.
    """
    # Features
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std == 0.0] = 1.0

    X_train_n = (X_train - x_mean) / x_std
    X_val_n   = (X_val   - x_mean) / x_std
    X_test_n  = (X_test  - x_mean) / x_std

    # Targets
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std == 0.0] = 1.0

    y_train_n = (y_train - y_mean) / y_std
    y_val_n   = (y_val   - y_mean) / y_std
    y_test_n  = (y_test  - y_mean) / y_std

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return X_train_n, X_val_n, X_test_n, y_train_n, y_val_n, y_test_n, stats

def feature_based_split(
    X,
    y,
    feature_idx: int,
    seed: int = 0,
    train_frac_id: float = 0.6,
    val_frac_id: float = 0.2,
    q_low: float = 0.2,
    q_high: float = 0.8,
):
    """
    Split X, y into train/val/test based on a designated feature:

      - Choose feature_idx (e.g., 0 for Airfoil frequency).
      - Define ID (interp) region as [q_low, q_high] quantiles of that feature.
      - Train/val use ONLY ID points.
      - Test contains remaining ID points + ALL OOD points.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, region_raw, region_interp_raw
    """
    rng = np.random.default_rng(seed)
    z = X[:, feature_idx]  # 1D feature in raw units

    z_min = float(z.min())
    z_max = float(z.max())
    q_lo, q_hi = np.quantile(z, [q_low, q_high])

    # ID / OOD masks
    id_mask = (z >= q_lo) & (z <= q_hi)
    ood_mask = ~id_mask

    id_idx = np.where(id_mask)[0]
    ood_idx = np.where(ood_mask)[0]

    rng.shuffle(id_idx)
    rng.shuffle(ood_idx)

    n_id = len(id_idx)
    n_train = int(train_frac_id * n_id)
    n_val = int(val_frac_id * n_id)
    n_test_id = n_id - n_train - n_val
    assert n_test_id >= 0, "train_frac_id + val_frac_id too large."

    train_idx = id_idx[:n_train]
    val_idx = id_idx[n_train:n_train + n_val]
    test_id_idx = id_idx[n_train + n_val:]

    # Test = remaining ID + all OOD
    test_idx = np.concatenate([test_id_idx, ood_idx])
    rng.shuffle(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx],   y[val_idx]
    X_test, y_test   = X[test_idx],  y[test_idx]

    region_raw = np.array([z_min, z_max], dtype=np.float64)
    region_interp_raw = np.array([q_lo, q_hi], dtype=np.float64)

    return X_train, y_train, X_val, y_val, X_test, y_test, region_raw, region_interp_raw


def save_npz(out_path,
             X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             desc,
             stats=None,
             region=None,
             region_interp=None,
             id_feature_idx=None):
    """
    Save arrays to .npz in the format expected by your npz loader:
    x_train, y_train, x_val, y_val, x_test, y_test, region, region_interp, desc

    region / region_interp should be on the SAME SCALE as the saved x_* arrays
    (i.e., after standardization). id_feature_idx tells which feature they refer to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if region is None:
        region = np.array([0.0, 1.0], dtype=np.float64)
    if region_interp is None:
        region_interp = np.array([0.0, 1.0], dtype=np.float64)

    save_dict = dict(
        x_train=X_train.astype(np.float64),
        y_train=y_train.astype(np.float64),
        x_val=X_val.astype(np.float64),
        y_val=y_val.astype(np.float64),
        x_test=X_test.astype(np.float64),
        y_test=y_test.astype(np.float64),
        region=region.astype(np.float64),
        region_interp=region_interp.astype(np.float64),
        desc=np.array(desc),
    )

    if stats is not None:
        # Store standardization stats so you can invert later if needed
        save_dict.update({
            "x_mean": stats["x_mean"],
            "x_std": stats["x_std"],
            "y_mean": stats["y_mean"],
            "y_std": stats["y_std"],
        })

    if id_feature_idx is not None:
        save_dict["id_feature_index"] = np.array([id_feature_idx], dtype=np.int64)

    np.savez(out_path, **save_dict)
    print(f"Saved: {out_path}")



# ------------------------------
# Dataset-specific loaders
# ------------------------------

def prep_airfoil(raw_dir, out_dir, seed=0):
    """
    UCI Airfoil Self-Noise: airfoil_self_noise.dat
    5 features, 1 target (last col).
    Feature 0 = frequency (Hz) is used for ID vs OOD.
    """
    raw_path = Path(raw_dir) / "airfoil_self_noise.dat"
    df = pd.read_csv(raw_path, delim_whitespace=True, header=None)

    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)

    feature_idx = 0  # frequency

    # 1) Split based on frequency
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw, \
        region_raw, region_interp_raw = feature_based_split(
            X, y, feature_idx=feature_idx, seed=seed,
            train_frac_id=0.6, val_frac_id=0.2, q_low=0.2, q_high=0.8
        )

    # 2) Standardize from TRAIN ONLY
    X_train, X_val, X_test, y_train, y_val, y_test, stats = standardize_from_train(
        X_train_raw, X_val_raw, X_test_raw,
        y_train_raw, y_val_raw, y_test_raw,
    )

    # 3) Map region / region_interp into normalized units of the chosen feature
    z_mean = stats["x_mean"][0, feature_idx]
    z_std  = stats["x_std"][0, feature_idx]
    region = (region_raw - z_mean) / z_std
    region_interp = (region_interp_raw - z_mean) / z_std

    desc = "UCI Airfoil Self-Noise: predict SPL from airfoil parameters"
    out_path = Path(out_dir) / f"airfoil_self_noise_feat_dim_{feature_idx}.npz"
    save_npz(
        out_path,
        X_train, y_train, X_val, y_val, X_test, y_test,
        desc,
        stats=stats,
        region=region,
        region_interp=region_interp,
        id_feature_idx=feature_idx,
    )


def prep_ccpp(raw_dir, out_dir, seed=0):
    """
    UCI Combined Cycle Power Plant: Folds5x2_pp.xlsx
    Columns: AT, V, AP, RH (features), PE (target).
    We use feature 0 (AT, ambient temperature) for ID vs OOD.
    """
    raw_path = Path(raw_dir) / "Folds5x2_pp.xlsx"
    df = pd.read_excel(raw_path, sheet_name=0)

    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)

    feature_idx = 0  # AT

    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw, \
        region_raw, region_interp_raw = feature_based_split(
            X, y, feature_idx=feature_idx, seed=seed,
            train_frac_id=0.6, val_frac_id=0.2, q_low=0.2, q_high=0.8
        )

    X_train, X_val, X_test, y_train, y_val, y_test, stats = standardize_from_train(
        X_train_raw, X_val_raw, X_test_raw,
        y_train_raw, y_val_raw, y_test_raw,
    )

    z_mean = stats["x_mean"][0, feature_idx]
    z_std  = stats["x_std"][0, feature_idx]
    region = (region_raw - z_mean) / z_std
    region_interp = (region_interp_raw - z_mean) / z_std

    desc = "UCI CCPP: predict net hourly electrical energy output (PE)"
    out_path = Path(out_dir) / f"ccpp_power_plant_feat_dim_{feature_idx}.npz"
    save_npz(
        out_path,
        X_train, y_train, X_val, y_val, X_test, y_test,
        desc,
        stats=stats,
        region=region,
        region_interp=region_interp,
        id_feature_idx=feature_idx,
    )


def prep_energy_efficiency(raw_dir, out_dir, seed=0, target="heating"):
    """
    UCI Energy Efficiency: ENB2012_data.xlsx
    8 features (X1...X8) and 2 targets (Y1=heating, Y2=cooling).
    We use feature 0 (X1 = Relative compactness) for ID vs OOD.
    """
    raw_path = Path(raw_dir) / "ENB2012_data.xlsx"
    df = pd.read_excel(raw_path)

    X = df.iloc[:, 0:8].to_numpy(dtype=np.float64)
    targets = df.iloc[:, 8:10].to_numpy(dtype=np.float64)

    feature_idx = 0  # X1 = Relative compactness

    if target == "heating":
        y = targets[:, 0:1]  # Y1
        desc = "UCI Energy Efficiency: predict Heating Load (Y1)"
        out_name = f"energy_efficiency_heating_feat_dim_{feature_idx}.npz"
    elif target == "cooling":
        y = targets[:, 1:2]  # Y2
        desc = "UCI Energy Efficiency: predict Cooling Load (Y2)"
        out_name = f"energy_efficiency_cooling_feat_dim_{feature_idx}.npz"
    else:
        raise ValueError("target must be 'heating' or 'cooling'")

    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw, \
        region_raw, region_interp_raw = feature_based_split(
            X, y, feature_idx=feature_idx, seed=seed,
            train_frac_id=0.6, val_frac_id=0.2, q_low=0.2, q_high=0.8
        )

    X_train, X_val, X_test, y_train, y_val, y_test, stats = standardize_from_train(
        X_train_raw, X_val_raw, X_test_raw,
        y_train_raw, y_val_raw, y_test_raw,
    )

    z_mean = stats["x_mean"][0, feature_idx]
    z_std  = stats["x_std"][0, feature_idx]
    region = (region_raw - z_mean) / z_std
    region_interp = (region_interp_raw - z_mean) / z_std

    out_path = Path(out_dir) / out_name
    save_npz(
        out_path,
        X_train, y_train, X_val, y_val, X_test, y_test,
        desc,
        stats=stats,
        region=region,
        region_interp=region_interp,
        id_feature_idx=feature_idx,
    )



# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prep UCI real datasets into .npz")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw UCI files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/uci_npz",
        help="Directory to write .npz files into",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for train/val/test split",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="heating",
        choices=["heating", "cooling", "both"],
        help="Which target to use for Energy Efficiency (Y1=heating, Y2=cooling).",
    )
    args = parser.parse_args()

    # Airfoil
    prep_airfoil(args.raw_dir, args.out_dir, seed=args.seed)

    # CCPP
    prep_ccpp(args.raw_dir, args.out_dir, seed=args.seed)

    # Energy Efficiency
    if args.target == "both":
        prep_energy_efficiency(args.raw_dir, args.out_dir, seed=args.seed, target="heating")
        prep_energy_efficiency(args.raw_dir, args.out_dir, seed=args.seed, target="cooling")
    else:
        prep_energy_efficiency(args.raw_dir, args.out_dir, seed=args.seed, target=args.target)
