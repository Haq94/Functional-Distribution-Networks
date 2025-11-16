#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr

from configs import baseline_params
from experiments.single_task_experiment import SingleTaskExperiment
from data.toy_functions import generate_splits
from utils.loader.single_task_loader import single_task_overlay_loader
from utils.plots.plot_helpers import iclr_figsize
from utils.saver.general_saver import save_plot


# -----------------------------------------------------------------------------------
# GLOBAL STYLE
# -----------------------------------------------------------------------------------
DPI = 150
plt.style.use("utils/plots/iclr.mplstyle")


# -----------------------------------------------------------------------------------
# ARGPARSE
# -----------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDN single-task paper script: train + analyze (toy or npz)."
    )

    # ---- high-level mode ----
    parser.add_argument(
        "--mode",
        choices=["train", "analyze", "train_and_analyze"],
        default="train",
        help="train: only train; analyze: only overlay analysis; "
             "train_and_analyze: train then run overlay analysis on resulting seeds.",
    )

    # ---- dataset options ----
    parser.add_argument(
        "--dataset-mode",
        choices=["toy", "npz"],
        default="toy",
        help="toy: generate toy 1D regression data (paper specs). "
             "npz: load pre-split data from an .npz file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to .npz file when --dataset-mode=npz.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="Seed for toy data split.",
    )

    # ---- experiment / training options ----
    parser.add_argument(
        "--exp-name",
        type=str,
        default="paper_repro",
        help="Name of this experiment (subfolder under results/single_task_experiment).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root folder for saving results.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="IC_FDNet,LP_FDNet,BayesNet,GaussHyperNet,MLPDropoutNet,DeepEnsembleNet",
        help=("Comma-separated list of models to run/analyze. "
              "Subset of: IC_FDNet,LP_FDNet,HyperNet,BayesNet,"
              "GaussHyperNet,MLPNet,MLPDropoutNet,DeepEnsembleNet."),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 8, 9],
        help="Training seeds (and default seeds to analyze if mode=train_and_analyze).",
    )

    parser.add_argument(
        "--beta-scheduler",
        choices=["linear", "cosine", "sigmoid", "unity", "zero"],
        default="cosine",
        help="KL beta schedule key (mapped into baseline_params).",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Disable per-run plotting inside SingleTaskExperiment.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving during training (debug only).",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append timestamp to experiment directory to avoid overwrites.",
    )

    # ---- overlay-analysis options (for mode=analyze / train_and_analyze) ----
    parser.add_argument(
        "--overlay-runs",
        type=str,
        nargs="+",
        default=None,
        help=("List of run directories to analyze. "
              "Each should be a *seed directory* that directly contains model folders, "
              "e.g. results/single_task_experiment/paper_repro/seed_7. "
              "If omitted and mode=train_and_analyze, we use the newly-trained seeds."),
    )
    parser.add_argument(
        "--overlay-name",
        type=str,
        default="quad",
        help="Short name for this overlay analysis (used as subfolder name for figs/tables).",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------------
# DATA HELPERS
# -----------------------------------------------------------------------------------

def build_input_data_dict_from_toy(cfg: dict, data_seed: int) -> dict:
    """Generate toy data using paper specs from baseline_params."""
    region = cfg["region"]
    region_interp = cfg["region_interp"]
    n_train = cfg["n_train"]
    n_test = cfg["n_test"]
    n_val_interp = cfg["n_val_interp"]
    n_val_extrap = cfg["n_val_extrap"]

    x_min, x_max = region
    input_data_dict = generate_splits(
        x_min=x_min,
        x_max=x_max,
        region_interp=region_interp,
        n_train=n_train,
        n_test=n_test,
        n_val_interp=n_val_interp,
        n_val_extrap=n_val_extrap,
        seed=data_seed,
    )
    return input_data_dict


def build_input_data_dict_from_npz(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"npz file not found: {path}")

    data = np.load(path)
    required = [
        "x_train", "y_train", "x_val", "y_val",
        "x_test", "y_test", "region", "region_interp",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"npz missing required keys: {missing}")

    return {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "x_val": data["x_val"],
        "y_val": data["y_val"],
        "x_test": data["x_test"],
        "y_test": data["y_test"],
        "region": tuple(data["region"]),
        "region_interp": tuple(data["region_interp"]),
    }


def make_save_path(results_root: str, exp_name: str, timestamp: bool) -> str:
    base_dir = Path(results_root) / "single_task_experiment"
    if timestamp:
        t_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        exp_dir = base_dir / f"{exp_name}_{t_str}"
    else:
        exp_dir = base_dir / exp_name
        if exp_dir.exists():
            t_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            print(f"[main] WARNING: {exp_dir} exists, appending timestamp.")
            exp_dir = base_dir / f"{exp_name}_{t_str}"

    exp_dir.mkdir(parents=True, exist_ok=True)
    return str(exp_dir)


def get_beta_param_dict(cfg: dict, name: str) -> dict:
    key_map = {
        "linear": "linear_beta_scheduler",
        "cosine": "cosine_beta_scheduler",
        "sigmoid": "signmoid_beta_scheduler",  # yes, spelled that way in configs
        "unity": "unity_beta_scheduler",
        "zero": "zero_beta_scheduler",
    }
    full_key = key_map[name]
    if full_key not in cfg:
        raise KeyError(f"{full_key} not found in baseline_params.")
    return cfg[full_key]


# -----------------------------------------------------------------------------------
# OVERLAY ANALYSIS (modularized delete.py)
# -----------------------------------------------------------------------------------

def lin_fit_mse_on_var(var, mse):
    var = np.asarray(var)
    mse = np.asarray(mse)
    A = np.vstack([var, np.ones_like(var)]).T
    b, a = np.linalg.lstsq(A, mse, rcond=None)[0]
    rce = float(np.mean((mse - var) ** 2))
    return float(a), float(b), rce


def binned_calibration(var, mse, nbins=20):
    var = np.asarray(var)
    mse = np.asarray(mse)
    qs = np.quantile(var, np.linspace(0, 1, nbins + 1))
    mids, emse, pvar = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (var >= lo) & ((var < hi) if hi < qs[-1] else (var <= hi))
        if np.any(sel):
            mids.append(np.median(var[sel]))
            emse.append(np.mean(mse[sel]))
            pvar.append(np.mean(var[sel]))
    return np.array(mids), np.array(emse), np.array(pvar)


def aurc(var, loss):
    var = np.asarray(var)
    loss = np.asarray(loss)
    order = np.argsort(var)  # lower var = higher confidence
    cum_mean = np.cumsum(loss[order]) / np.arange(1, len(loss) + 1)
    coverage = np.arange(1, len(loss) + 1) / len(loss)
    area = float(np.trapz(cum_mean, coverage))
    return area, coverage, cum_mean


def safe_mean(x):
    x = np.asarray(x)
    mask = np.isfinite(x)
    return float(x[mask].mean()) if mask.any() else np.nan


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return np.nan
    return spearmanr(x[mask], y[mask])[0]


def run_overlay_analysis(
    seed_dirs,
    out_root,
    overlay_name="quad",
    run_models=None,
    save_tables=True,
    save_plots=True,
):
    """
    Reproduce delete.py-style overlay analysis across a list of seed dirs.

    Each element of seed_dirs should look like:
      results/single_task_experiment/<exp_name>/seed_<seed>
    and must contain subfolders named by model type (IC_FDNet, ...).
    """
    if run_models is None:
        run_models = [
            "IC_FDNet",
            "LP_FDNet",
            "BayesNet",
            "GaussHyperNet",
            "MLPDropoutNet",
            "DeepEnsembleNet",
        ]

    model_colors = {
        "IC_FDNet": "#1f77b4",
        "LP_FDNet": "#ff7f0e",
        "BayesNet": "#2ca02c",
        "GaussHyperNet": "#ff0000",
        "DeepEnsembleNet": "#653593",
        "HyperNet": "#8c564b",
        "MLPNet": "#e377c2",
        "MLPDropoutNet": "#e2fb55",
    }

    def display_name(name: str) -> str:
        if name == "IC_FDNet":
            return "IC-FDNet"
        if name == "LP_FDNet":
            return "LP-FDNet"
        return name

    out_dir = os.path.join(out_root, overlay_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[overlay] analyzing seeds: {seed_dirs}")
    (
        loaders,
        metrics_test,
        metrics_train,
        metrics_val,
        losses,
        summary,
        x_train_all,
        y_train_all,
        x_val_all,
        y_val_all,
        x_test_all,
        y_test_all,
        region_all,
        region_interp_all,
        run_list,
    ) = single_task_overlay_loader(seed_dirs)

    # We'll aggregate over all provided seeds/runs
    # (delete.py used run index 0/1/2; here you just choose the seed_dirs explicitly)
    assert len(run_list) == len(seed_dirs)

    # Use first run's region info
    first_run = run_list[0]
    region = region_all[first_run]
    region_interp = region_interp_all[first_run]
    x_min, x_max = region_interp

    # build combined store per model
    store = {}
    models_present = []

    for model in run_models:
        xs = []
        mses = []
        vars_ = []
        nlpds = []
        crpss = []

        for run_path in run_list:
            mtest = metrics_test[run_path].get(model)
            if mtest is None:
                continue
            x_test = x_test_all[run_path]
            y_test = y_test_all[run_path]
            y_hat_mean = mtest["mean"]
            y_hat_var = mtest["var"]
            mse = (y_hat_mean - y_test.squeeze()) ** 2

            xs.append(x_test.squeeze())
            mses.append(mse)
            vars_.append(y_hat_var)

            if "nlpd_kde" in mtest:
                nlpds.append(mtest["nlpd_kde"])
            if "crps" in mtest:
                crpss.append(mtest["crps"])

        if not xs:
            continue

        x = np.concatenate(xs)
        mse = np.concatenate(mses)
        var = np.concatenate(vars_)
        nlpd = np.concatenate(nlpds) if nlpds else np.full_like(mse, np.nan)
        crps = np.concatenate(crpss) if crpss else np.full_like(mse, np.nan)

        store[model] = dict(
            x=x,
            mse=mse,
            var=var,
            nlpd=nlpd,
            crps=crps,
            label=display_name(model),
            color=model_colors.get(model, "#777777"),
        )
        models_present.append(model)

    if not models_present:
        print("[overlay] No models present in provided runs.")
        return

    # ---- ID / OOD masks ----
    for name in models_present:
        x = store[name]["x"]
        interp_mask = (x >= x_min) & (x <= x_max)
        extrap_mask = ~interp_mask
        store[name]["interp_mask"] = interp_mask
        store[name]["extrap_mask"] = extrap_mask

    # --------------------------------------------------------------------------------
    # SUMMARY TABLE (mean/median MSE, mean Var, correlations, lin fit, RCE, AURC)
    # --------------------------------------------------------------------------------
    results = []
    for name in models_present:
        arr = store[name]
        var = arr["var"]
        mse = arr["mse"]

        rho = safe_spearman(var, mse)
        tau = safe_spearman(var, mse)  # close enough; delete.py used kendalltau separately
        pr = pearsonr(var, mse)[0]

        a, b, rce = lin_fit_mse_on_var(var, mse)
        auc, cov, risk = aurc(var, mse)

        store[name].update(dict(rho=rho, tau=tau, pearson=pr,
                                a=a, b=b, rce=rce, aurc=auc, cov=cov, risk=risk))

        results.append([
            display_name(name),
            float(np.mean(mse)),
            float(np.median(mse)),
            float(np.mean(var)),
            rho, tau, pr, a, b, rce, auc,
        ])

    df_summary = None
    try:
        import pandas as pd

        cols = [
            "Model",
            "Mean MSE", "Median MSE", "Mean Var",
            "Spearman ρ", "Kendall τ", "Pearson r",
            "Intercept a", "Slope b", "RCE", "AURC",
        ]
        df_summary = pd.DataFrame(results, columns=cols).sort_values("AURC").reset_index(drop=True)
        print("\n=== Summary (lower AURC/RCE better; ideal slope≈1, intercept≈0) ===")
        print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

        if save_tables:
            df_summary.to_csv(os.path.join(out_dir, "summary_main.csv"), index=False)
            with open(os.path.join(out_dir, "summary_main.tex"), "w", encoding="utf-8") as f:
                f.write(df_summary.to_latex(index=False, escape=True))
    except Exception as e:
        print(f"[overlay] pandas summary failed ({e}); printing fallback.")
        for row in sorted(results, key=lambda r: r[-1]):
            print(row)

    # --------------------------------------------------------------------------------
    # ID vs OOD delta table (ΔMSE, ΔVar, ΔNLPD, ΔCRPS)
    # --------------------------------------------------------------------------------
    df_delta = None
    df_region = None
    try:
        import pandas as pd
        rows = []
        for name in models_present:
            arr = store[name]
            x = arr["x"]
            var = arr["var"]
            mse = arr["mse"]
            nlpd = arr["nlpd"]
            crps = arr["crps"]
            interp_mask = arr["interp_mask"]
            extrap_mask = arr["extrap_mask"]

            # sort by x just to be consistent
            order = np.argsort(x)
            interp_mask_sorted = interp_mask[order]
            extrap_mask_sorted = extrap_mask[order]
            mse_sorted = mse[order]
            var_sorted = var[order]
            nlpd_sorted = nlpd[order]
            crps_sorted = crps[order]

            has_nlpd = np.isfinite(nlpd_sorted).any()
            has_crps = np.isfinite(crps_sorted).any()

            mse_id = safe_mean(mse_sorted[interp_mask_sorted])
            mse_ood = safe_mean(mse_sorted[extrap_mask_sorted])
            var_id = safe_mean(var_sorted[interp_mask_sorted])
            var_ood = safe_mean(var_sorted[extrap_mask_sorted])
            nlpd_id = safe_mean(nlpd_sorted[interp_mask_sorted]) if has_nlpd else np.nan
            nlpd_ood = safe_mean(nlpd_sorted[extrap_mask_sorted]) if has_nlpd else np.nan
            crps_id = safe_mean(crps_sorted[interp_mask_sorted]) if has_crps else np.nan
            crps_ood = safe_mean(crps_sorted[extrap_mask_sorted]) if has_crps else np.nan

            rows.append([
                display_name(name),
                mse_id, mse_ood, mse_ood - mse_id,
                var_id, var_ood, var_ood - var_id,
                nlpd_id, nlpd_ood, nlpd_ood - nlpd_id,
                crps_id, crps_ood, crps_ood - crps_id,
            ])

        cols = [
            "Model",
            "MSE(ID)", "MSE(OOD)", "ΔMSE",
            "Var(ID)", "Var(OOD)", "ΔVar",
            "NLPD(ID)", "NLPD(OOD)", "ΔNLPD",
            "CRPS(ID)", "CRPS(OOD)", "ΔCRPS",
        ]
        df_delta = pd.DataFrame(rows, columns=cols)
        df_region = df_delta[
            ["Model", "MSE(ID)", "MSE(OOD)", "Var(ID)", "Var(OOD)",
             "NLPD(ID)", "NLPD(OOD)", "CRPS(ID)", "CRPS(OOD)"]
        ].copy()

        sort_keys = []
        if np.isfinite(df_delta["ΔNLPD"]).any():
            sort_keys.append("ΔNLPD")
        if np.isfinite(df_delta["ΔCRPS"]).any():
            sort_keys.append("ΔCRPS")
        sort_keys.append("ΔMSE")
        df_delta = df_delta.sort_values(sort_keys, na_position="last").reset_index(drop=True)

        print("\n=== Interp (ID) vs Extrap (OOD) summary (Δ = OOD − ID) ===")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df_delta.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

        if save_tables:
            df_delta.to_csv(os.path.join(out_dir, "delta_id_to_ood.csv"), index=False)
            df_region.to_csv(os.path.join(out_dir, "region_metrics_id_ood.csv"), index=False)
    except Exception as e:
        print(f"[overlay] delta table failed ({e})")

    # --------------------------------------------------------------------------------
    # PLOTS (same family as delete.py)
    # --------------------------------------------------------------------------------
    if not save_plots:
        return

    # (1) Variance vs x
    fig, ax = plt.subplots(1, 1, figsize=iclr_figsize(layout="single"), dpi=DPI)
    for name in models_present:
        ax.plot(store[name]["x"], store[name]["var"],
                lw=1.6, label=store[name]["label"],
                color=store[name]["color"])
    ax.set_ylabel("Variance")
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    save_plot(fig, os.path.join(out_dir, "var_vs_x"))

    # (2) MSE vs Var scatter + linear fits
    fig, ax = plt.subplots(1, 1, figsize=iclr_figsize(layout="single"), dpi=DPI)
    vmin = min(store[n]["var"].min() for n in models_present)
    vmax = max(store[n]["var"].max() for n in models_present)
    xline = np.linspace(vmin, vmax, 200)
    for name in models_present:
        arr = store[name]
        ax.scatter(arr["var"], arr["mse"], s=8, alpha=0.4, color=arr["color"])
        a, b = arr["a"], arr["b"]
        ax.plot(xline, a + b * xline, lw=1.6, color=arr["color"],
                label=f'{arr["label"]} (a={a:.2g}, b={b:.2g})')
    ax.set_xlabel("Predicted variance")
    ax.set_ylabel("Empirical MSE")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    save_plot(fig, os.path.join(out_dir, "mse_vs_var_scatter"))

    # (3) Calibration curve: binned MSE vs var
    fig, ax = plt.subplots(1, 1, figsize=iclr_figsize(layout="single"), dpi=DPI)
    ylim = max(store[n]["var"].max() for n in models_present)
    ax.plot([0, ylim], [0, ylim], "k--", lw=1, label="Ideal: y=x")
    for name in models_present:
        mid, emse, pvar = binned_calibration(store[name]["var"], store[name]["mse"], nbins=20)
        ax.plot(pvar, emse, "o-", ms=4, lw=1,
                color=store[name]["color"], label=store[name]["label"])
    ax.set_xlabel("Predicted variance (bin mean)")
    ax.set_ylabel("Empirical MSE (bin mean)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    save_plot(fig, os.path.join(out_dir, "calibration_curve"))

    # (4) Risk–Coverage
    fig, ax = plt.subplots(1, 1, figsize=iclr_figsize(layout="single"), dpi=DPI)
    for name in models_present:
        arr = store[name]
        ax.plot(arr["cov"], arr["risk"], lw=1.6, color=arr["color"],
                label=f'{arr["label"]} (AURC={arr["aurc"]:.3f})')
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (cumulative mean MSE)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    save_plot(fig, os.path.join(out_dir, "risk_coverage"))

    # (5) NLPD vs x (if available)
    have_nlpd = [n for n in models_present if np.isfinite(store[n]["nlpd"]).any()]
    if have_nlpd:
        fig, ax = plt.subplots(1, 1, figsize=iclr_figsize(layout="single"), dpi=DPI)
        for name in have_nlpd:
            arr = store[name]
            ax.plot(arr["x"], arr["nlpd"], lw=1.6,
                    label=arr["label"], color=arr["color"])
        ax.set_xlabel("x")
        ax.set_ylabel("NLPD (KDE)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, ncol=3, fontsize=9)
        fig.tight_layout()
        save_plot(fig, os.path.join(out_dir, "nlpd_vs_x"))

    # (6) Delta bar charts if we have df_delta
    if df_delta is not None:
        import pandas as pd
        labels = df_delta["Model"].values
        idx = np.arange(len(labels))
        colors = [model_colors.get(m.replace("-", "_"), "#777") for m in labels]

        metrics = [("ΔMSE", "delta_bars_MSE"),
                   ("ΔVar", "delta_bars_Var"),
                   ("ΔNLPD", "delta_bars_NLPD"),
                   ("ΔCRPS", "delta_bars_CRPS")]

        for col, fname in metrics:
            if col not in df_delta.columns:
                continue
            fig, ax = plt.subplots(1, 1, figsize=iclr_figsize(layout="single"), dpi=DPI)
            vals = df_delta[col].values
            ax.bar(idx, vals, color=colors)
            ax.set_xticks(idx)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel(col)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            save_plot(fig, os.path.join(out_dir, fname))


# -----------------------------------------------------------------------------------
# TRAINING DRIVER
# -----------------------------------------------------------------------------------

def run_training_and_get_seed_dirs(args: argparse.Namespace) -> (str, list):
    cfg = baseline_params

    # data
    if args.dataset-mode == "toy":
        print("[main] Using toy dataset (paper specs).")
        input_data_dict = build_input_data_dict_from_toy(cfg, args.data_seed)
    else:
        if args.data_path is None:
            raise ValueError("--data-path required for dataset-mode=npz")
        print(f"[main] Loading npz dataset from {args.data_path}")
        input_data_dict = build_input_data_dict_from_npz(args.data_path)

    # model list & seeds
    model_types = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = args.seeds

    print(f"[main] Training models: {model_types}")
    print(f"[main] Seeds: {seeds}")

    # model hyperparams ~1k params from baseline_params
    full_model_dict = cfg["model_dict"]
    model_dict = {m: full_model_dict[m] for m in model_types}

    epochs = cfg["epochs"]
    ensemble_epochs = cfg["ensemble_epochs"]
    MC_train = cfg["MC_train"]
    MC_val = cfg["MC_val"]
    MC_test = cfg["MC_test"]
    checkpoint_dicts = cfg["checkpoint_dict"]
    beta_param_dict = get_beta_param_dict(cfg, args.beta_scheduler)

    plot_dict = {
        "Single": [
            "loss_vs_epoch",
            "mean_vs_x",
            "mse_vs_x",
            "nlpd_kde_vs_x",
            "pit_two_panel",
            "mse_db_vs_var_db",
            "nll_kde_heatmap",
        ],
        "Overlay": [
            "mean_vs_x",
            "mses_vs_epoch",
            "nlpd_kde_vs_x",
            "crps_db_vs_nlpd_kde",
            "crps_db_vs_nlpd_kde_2x2",
            "mse_db_vs_var_db_2x2",
        ],
    }

    analysis = not args.no_analysis
    save_switch = not args.no_save

    save_path = make_save_path(args.results_root, args.exp_name, args.timestamp)
    print(f"[main] saving training results under: {save_path}")

    exp = SingleTaskExperiment(
        model_type=model_types,
        seeds=seeds,
        model_dict=model_dict,
        plot_dict=plot_dict,
    )

    exp.run_experiments(
        input_data_dict=input_data_dict,
        epochs=epochs,
        beta_param_dict=beta_param_dict,
        checkpoint_dicts=checkpoint_dicts,
        MC_train=MC_train,
        MC_val=MC_val,
        MC_test=MC_test,
        analysis=analysis,
        save_switch=save_switch,
        save_path=save_path,
        ensemble_epochs=ensemble_epochs,
    )

    # return list of *seed dirs* (each contains model folders)
    seed_dirs = [os.path.join(save_path, f"seed_{s}") for s in seeds]
    return save_path, seed_dirs


# -----------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.mode in {"train", "train_and_analyze"}:
        exp_root, new_seed_dirs = run_training_and_get_seed_dirs(args)
    else:
        exp_root, new_seed_dirs = None, None

    if args.mode in {"analyze", "train_and_analyze"}:
        # overlay seed dirs
        if args.overlay_runs is not None:
            seed_dirs = args.overlay_runs
        elif new_seed_dirs is not None:
            seed_dirs = new_seed_dirs
        else:
            raise ValueError("No seed dirs provided for overlay analysis.")

        overlay_root = exp_root if exp_root is not None else os.path.dirname(seed_dirs[0])
        run_overlay_analysis(
            seed_dirs=seed_dirs,
            out_root=overlay_root,
            overlay_name=args.overlay_name,
            run_models=[m.strip() for m in args.models.split(",") if m.strip()],
            save_tables=True,
            save_plots=True,
        )


if __name__ == "__main__":
    main()
