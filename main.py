#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr

from utils.configs import baseline_params
from experiments.single_task_experiment import SingleTaskExperiment
from data.toy_functions import generate_splits
from utils.loader.single_task_loader import single_task_overlay_loader
from utils.plots.plot_helpers import iclr_figsize, label_subplots
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

    parser.add_argument(
        "--analysis-type",
        choices=["overlay", "real"],
        default="overlay",
        help=(
            "overlay: 1D toy-style overlay analysis (y vs x, shaded ID/OOD). "
            "real: high-dimensional real-data analysis (aggregate metrics across seeds)."
        ),
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

    # ---- experiment naming / saving ----
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory to save experiments under.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="paper_repo",
        help="Name of this experiment (subfolder under results_root/single_task_experiment).",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append timestamp to experiment directory to avoid overwrites.",
    )

    # ---- models / seeds ----
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
        default=[24, 25, 26],
        help="Training seeds (and default seeds to analyze if mode=train_and_analyze).",
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
        default=None,
        help="Short name for this overlay analysis (used as subfolder name for figs/tables).",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------------
# DATA HELPERS
# -----------------------------------------------------------------------------------

def build_input_data_dict_from_toy(cfg: dict, data_seed: int) -> dict:
    """
    Generate toy *inputs* only. SingleTaskExperiment will generate y via sample_function.
    """
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

    # generate_splits should give us x’s (and usually region info). We do NOT
    # expect any y’s here – SingleTaskExperiment creates them.
    required = ["x_train", "x_val", "x_test"]
    for k in required:
        if k not in input_data_dict:
            raise KeyError(f"generate_splits missing key: {k}")

    return {
        "x_train": input_data_dict["x_train"],
        "x_val": input_data_dict["x_val"],
        "x_test": input_data_dict["x_test"],
        # These may already be in input_data_dict; we override with cfg for safety.
        "region": tuple(region),
        "region_interp": tuple(region_interp),
    }



def build_input_data_dict_from_npz(path: str) -> dict:
    """Load pre-split data from .npz with keys matching the toy dict structure."""
    data = np.load(path)
    required = [
        "x_train",
        "y_train",
        "x_val",
        "y_val",
        "x_test",
        "y_test",
        "region",
        "region_interp",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"npz at {path} missing required keys: {missing}")
    
    if "id_feature_index" in data:
        id_feature_index = int(np.asarray(data["id_feature_index"]).item())
    else:
        id_feature_index = None

    return {
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "x_val": data["x_val"],
        "y_val": data["y_val"],
        "x_test": data["x_test"],
        "y_test": data["y_test"],
        "region": tuple(data["region"]),
        "region_interp": tuple(data["region_interp"]),
        "id_feature_index": id_feature_index,
    }


def make_save_path(results_root: str, exp_name: str, timestamp: bool) -> str:
    base_dir = Path(results_root) / "single_task_experiment"
    if timestamp:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{exp_name}_{now}"
    save_path = base_dir / exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    return str(save_path)


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
    auc = float(np.trapz(cum_mean, coverage))
    return auc, coverage, cum_mean


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


def safe_kendall(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return np.nan
    return kendalltau(x[mask], y[mask])[0]

def infer_run_name_from_summary(run_summary: dict) -> str | None:
    """
    Infer a short task name (e.g. 'step', 'sine', 'quad') from the summary dict
    for a single run.

    run_summary is summary[run_key]: a dict mapping model_name -> summary_info.
    We assume the function description is stored as summary_info['desc'].
    """
    if not isinstance(run_summary, dict) or not run_summary:
        return None

    # Take the first model's summary (all models share the same underlying function)
    first_summary = next(iter(run_summary.values()))
    if not isinstance(first_summary, dict):
        return None

    desc = first_summary.get("desc", "")
    if not isinstance(desc, str):
        desc = str(desc)
    dl = desc.lower()

    # Try to map to the canonical toy names
    if "step" in dl:
        return "step"
    if "sine" in dl or "sin" in dl:
        return "sine"
    if "quad" in dl or "quadratic" in dl:
        return "quad"

    # Fallback: short slug from the description
    slug = re.sub(r"[^a-z0-9]+", "-", dl).strip("-")
    return slug[:30] or None



def run_overlay_analysis(
    seed_dir,
    out_root,
    overlay_name=None,
    run_models=None,
    save_tables=True,
    save_plots=True,
    feat_dim=None
):
    """
    Paper-style overlay analysis: **one task / run at a time**.

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
        "GaussHyperNet": "#d62728",
        "MLPDropoutNet": "#9467bd",
        "DeepEnsembleNet": "#8c564b",
        "MLPNet": "#17becf",
        "HyperNet": "#e377c2",
    }

    def display_name(name: str) -> str:
        if name == "IC_FDNet":
            return "IC-FDNet"
        if name == "LP_FDNet":
            return "LP-FDNet"
        return name

    out_dir = os.path.join(seed_dir, "paper_figs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[overlay] analyzing seeds: {seed_dir}")
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
    ) = single_task_overlay_loader([seed_dir])

    # ---- Select single run (paper-style overlay, one task at a time) ----
    if len(run_list) != 1:
        raise ValueError(
            f"run_overlay_analysis (paper mode) expects exactly one run/seed dir, "
            f"got {len(run_list)}. Call it separately per task."
        )
    run_key = run_list[0]

   # Infer overlay_name from summary.json if not provided
    if overlay_name is None:
        run_summary = summary.get(run_key, {})
        inferred = infer_run_name_from_summary(run_summary)
        overlay_name = inferred or "overlay"

    # Region info for this task
    region = region_all[run_key]
    region_interp = region_interp_all[run_key]
    x_min, x_max = region_interp

    # build per-model store for this single run
    store = {}
    models_present = []

    for model in run_models:
        mtest = metrics_test[run_key].get(model)
        if mtest is None:
            continue

        x_test = x_test_all[run_key]
        y_test = y_test_all[run_key]

        y_hat_mean = mtest["mean"]
        y_hat_var = mtest["var"]

        y_true = y_test.squeeze()
        mean_pred = y_hat_mean.squeeze()

        mse = mtest['mse']
        var = mtest['var']

        # Optional metrics
        if "nlpd_kde" in mtest:
            nlpd = mtest["nlpd_kde"]
        else:
            nlpd = np.full_like(var, np.nan, dtype=float)

        if "crps" in mtest:
            crps = mtest["crps"]
        else:
            crps = np.full_like(var, np.nan, dtype=float)

        store[model] = dict(
            x=x_test.squeeze(),
            y_true=y_true,
            mean=mean_pred,
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
    
    # ------------------------------------------------------------
    # Determine whether test inputs are 1D or multi-dimensional
    # ------------------------------------------------------------
    first_x = store[models_present[0]]["x"]

    if first_x.ndim == 1:
        is_1d_x = True
    elif first_x.ndim == 2 and first_x.shape[1] == 1:
        # Column vector -> squeeze to 1D
        is_1d_x = True
        for name in models_present:
            store[name]["x"] = store[name]["x"].reshape(-1)
        first_x = store[models_present[0]]["x"]
    else:
        is_1d_x = False
        print(
            f"[overlay] Detected multi-dimensional inputs x.shape={first_x.shape}; "
            "will skip mean/metric-vs-x panels and treat all points as 'ID' for "
            "ID/OOD summaries."
        )

    # ---- ID / OOD masks ----
    for name in models_present:
        x = store[name]["x"]
        x_feat_split = x[:, feat_dim].squeeze() if feat_dim is not None else x
        interp_mask = (x_feat_split >= x_min) & (x_feat_split <= x_max)
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
        tau = safe_kendall(var, mse)
        pr = pearsonr(var, mse)[0]

        a, b, rce = lin_fit_mse_on_var(var, mse)
        auc, cov, risk = aurc(var, mse)

        store[name].update(dict(rho=rho, tau=tau, pearson=pr,
                                a=a, b=b, rce=rce, aurc=auc, cov=cov, risk=risk))

        results.append([
            display_name(name),
            safe_mean(mse),
            np.median(mse),
            safe_mean(var),
            rho,
            tau,
            pr,
            a,
            b,
            rce,
            auc,
        ])

    cols = [
        "Model",
        "MSE(mean)",
        "MSE(median)",
        "Var(mean)",
        "Spearman ρ",
        "Kendall τ",
        "Pearson r",
        "a (intercept)",
        "b (slope)",
        "RCE",
        "AURC",
    ]
    df_summary = pd.DataFrame(results, columns=cols)
    print("\n=== Global summary (all points) ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    if save_tables:
        df_summary.to_csv(os.path.join(out_dir, "summary_main.csv"), index=False)
        try:
            with open(os.path.join(out_dir, "summary_main.tex"), "w") as f_tex:
                f_tex.write(df_summary.to_latex(index=False, float_format=lambda x: f"{x:.3g}"))
        except Exception as e:
            print(f"[overlay] LaTeX summary export failed ({e})")

    # --------------------------------------------------------------------------------
    # ID vs OOD per-model metrics + deltas
    # --------------------------------------------------------------------------------
    df_delta = None
    df_region = None
    try:
        rows = []
        for name in models_present:
            arr = store[name]
            x = arr["x"]
            var = arr["var"]
            mse = arr["mse"]
            crps = arr["crps"]
            interp_mask = arr["interp_mask"]
            extrap_mask = arr["extrap_mask"]

            def region_stats(mask, var, mse, crps):
                """
                Compute mean MSE, Var, CRPS over a boolean mask.
                Returns NaN if the mask selects no points.
                """
                if mask is None:
                    return np.nan, np.nan, np.nan

                mask = np.asarray(mask, dtype=bool)
                if mask.ndim > 1:
                    # Flatten masks that came from multi-d inputs, just in case.
                    mask = mask.reshape(-1)

                if mask.size == 0 or not np.any(mask):
                    # No points in this region
                    return np.nan, np.nan, np.nan

                mse_region = mse[mask]
                var_region = var[mask]

                mse_mean = float(np.mean(mse_region))
                var_mean = float(np.mean(var_region))

                if crps is None:
                    crps_mean = np.nan
                else:
                    crps_region = crps[mask]
                    crps_mean = float(np.mean(crps_region))

                return mse_mean, var_mean, crps_mean


            mse_id, var_id, crps_id = region_stats(interp_mask, var, mse, crps)
            mse_ood, var_ood, crps_ood = region_stats(extrap_mask, var, mse, crps)

            rows.append([
                display_name(name),
                mse_id, mse_ood, mse_ood - mse_id,
                var_id, var_ood, var_ood - var_id,
                crps_id, crps_ood, crps_ood - crps_id,
            ])

        cols = [
            "Model",
            "MSE(ID)", "MSE(OOD)", "ΔMSE",
            "Var(ID)", "Var(OOD)", "ΔVar",
            "CRPS(ID)", "CRPS(OOD)", "ΔCRPS",
        ]
        df_delta = pd.DataFrame(rows, columns=cols)
        df_region = df_delta[
            ["Model", "MSE(ID)", "MSE(OOD)",
             "Var(ID)", "Var(OOD)",
             "CRPS(ID)", "CRPS(OOD)"]
        ].copy()

        sort_keys = []
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
    # PLOTS 
    # --------------------------------------------------------------------------------
    if not save_plots:
        return

    # (1) MSE vs Var scatter + linear fits
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
    save_plot(out_dir, "mse_vs_var_scatter", dpi=DPI, fig=fig)

    # (2) Calibration curve: binned MSE vs var
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
    save_plot(out_dir, "calibration_curve", dpi=DPI, fig=fig)

    # (3) Risk–Coverage
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
    save_plot(out_dir, "risk_coverage", dpi=DPI, fig=fig)

    # (4) Predictive mean vs x (with ground truth and ID region shaded)
    # -----------------------------------------------------------------
    fig_mean, ax_mean = plt.subplots(
        1, 1,
        figsize=iclr_figsize(layout="single"),
        dpi=DPI,
    )

    # Use any one model just to get the x-grid and y_true
    first_model = models_present[0]
    x_plot = x_feat_split
    y_true_plot = store[first_model]["y_true"]

    # Shade interpolation (ID) region
    ax_mean.axvspan(x_min, x_max, color="0.9", alpha=0.4, zorder=0)

    # Ground truth curve
    order = np.argsort(x_plot)
    ax_mean.plot(
        x_plot[order],
        y_true_plot[order],
        "k--",
        lw=1.3,
        label="Ground truth",
    )

    # Model means
    for name in models_present:
        arr = store[name]
        x = x_feat_split
        mean_pred = arr["mean"]
        order = np.argsort(x)
        ax_mean.plot(
            x[order],
            mean_pred[order],
            lw=1.3,
            color=arr["color"],
            label=arr["label"],
        )

    ax_mean.set_ylabel("Mean")
    ax_mean.set_xlabel("x")
    ax_mean.grid(alpha=0.3)
    ax_mean.legend(frameon=False, ncol=2, fontsize=8)

    fig_mean.tight_layout()
    save_plot(out_dir, "mean_vs_x_with_truth", dpi=DPI, fig=fig_mean)

    # (5) MSE, variance, and CRPS vs x (1×3 row, ID region shaded)
    # -------------------------------------------------------------
    # Check whether we have any non-NaN CRPS at all
    have_crps = any(np.isfinite(store[n]["crps"]).any() for n in models_present)

    n_cols = 3 if have_crps else 2
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(9.0, 3.0),  # tweak if you want wider/narrower
        dpi=DPI,
        sharex=True,
    )
    if n_cols == 1:
        axes = [axes]

    # Shade interpolation (ID) region in all subplots
    for ax in axes:
        ax.axvspan(x_min, x_max, color="0.9", alpha=0.4, zorder=0)

    # Col 1: MSE vs x
    ax_mse = axes[0]
    for name in models_present:
        arr = store[name]
        x = x_feat_split
        mse = arr["mse"]
        order = np.argsort(x)
        ax_mse.plot(
            x[order],
            mse[order],
            lw=1.3,
            color=arr["color"],
            label=arr["label"],
        )
    ax_mse.set_ylabel("MSE")
    ax_mse.grid(alpha=0.3)

    # Col 2: variance vs x
    ax_var = axes[1]
    for name in models_present:
        arr = store[name]
        x = x_feat_split
        v = arr["var"]
        order = np.argsort(x)
        ax_var.plot(
            x[order],
            v[order],
            lw=1.3,
            color=arr["color"],
            label=arr["label"],
        )
    ax_var.set_ylabel("Variance")
    ax_var.grid(alpha=0.3)

    # Col 3 (optional): CRPS vs x
    if have_crps:
        ax_crps = axes[2]
        for name in models_present:
            arr = store[name]
            x = x_feat_split
            c = arr["crps"]
            if not np.isfinite(c).any():
                continue
            order = np.argsort(x)
            ax_crps.plot(
                x[order],
                c[order],
                lw=1.3,
                color=arr["color"],
                label=arr["label"],
            )
        ax_crps.set_ylabel("CRPS")
        ax_crps.grid(alpha=0.3)

    # Common x-label
    axes[-1].set_xlabel("x")

    # No (a),(b),(c) labels here
    fig.tight_layout()
    save_plot(out_dir, "mse_var_crps_vs_x", dpi=DPI, fig=fig)


    # (6) Delta bar charts: ΔMSE / ΔVar / ΔCRPS as 1×3 row (left-to-right)
    if df_delta is not None:
        labels = df_delta["Model"].values
        idx = np.arange(len(labels))
        colors = [model_colors.get(m.replace("-", "_"), "#777") for m in labels]

        metric_specs = [
            ("ΔMSE",  "ΔMSE (OOD − ID)"),
            ("ΔVar",  "ΔVar (OOD − ID)"),
            ("ΔCRPS", "ΔCRPS (OOD − ID)"),
        ]

        # 1 row, multiple columns (left-to-right)
        n_cols = len(metric_specs)
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(9.0, 3.0),  # tweak width/height as needed
            dpi=DPI,
            sharey=False,
        )
        if n_cols == 1:
            axes = [axes]

        for ax, (col, ylabel) in zip(axes, metric_specs):
            if col not in df_delta.columns:
                ax.set_visible(False)
                continue
            vals = df_delta[col].values
            ax.bar(idx, vals, color=colors)
            ax.set_ylabel(ylabel)
            ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.grid(axis="y", alpha=0.3)
            ax.set_xticks(idx)
            ax.set_xticklabels(labels, rotation=30, ha="right")

        # No (a), (b), (c) labels here
        fig.tight_layout()
        save_plot(out_dir, "delta_bars_MSE_Var_CRPS", dpi=DPI, fig=fig)


    # (7) Scatter: MSE vs Var in Interp vs Extrap regions
    def _add_ref_line(ax, all_vars):
        cats = [a for a in all_vars if a.size > 0 and np.isfinite(a).any()]
        if not cats:
            return
        vmin = min(a.min() for a in cats)
        vmax = max(a.max() for a in cats)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return
        xline = np.linspace(vmin, vmax, 200)
        ax.plot(xline, xline, "k--", lw=1, label="Ideal: MSE=Var")

    fig, axes = plt.subplots(1, 2, figsize=iclr_figsize(layout="double"), dpi=DPI, sharey=True)
    all_vars_i, all_vars_o = [], []

    for name in models_present:
        arr = store[name]
        c = arr["color"]
        lab = arr["label"]
        x = arr["x"]
        v = arr["var"]
        m = arr["mse"]
        interp_mask = arr["interp_mask"]
        extrap_mask = arr["extrap_mask"]

        order = np.argsort(x)
        vi = v[order][interp_mask[order]]
        mi = m[order][interp_mask[order]]
        vo = v[order][extrap_mask[order]]
        mo = m[order][extrap_mask[order]]

        rho_i = safe_spearman(vi, mi)
        rho_o = safe_spearman(vo, mo)

        axes[0].scatter(vi, mi, s=10, alpha=0.35, color=c,
                        label=f"{lab} (ρ={np.nan if np.isnan(rho_i) else rho_i:.2f})")
        axes[1].scatter(vo, mo, s=10, alpha=0.35, color=c,
                        label=f"{lab} (ρ={np.nan if np.isnan(rho_o) else rho_o:.2f})")

        all_vars_i.append(vi)
        all_vars_o.append(vo)

    _add_ref_line(axes[0], all_vars_i)
    _add_ref_line(axes[1], all_vars_o)

    for ax in axes:
        ax.set_xlabel("Variance")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Interp (ID)")
    axes[1].set_title("Extrap (OOD)")
    # label_subplots(axes)
    fig.tight_layout()
    save_plot(out_dir, "scatter_interp_extrap", dpi=DPI, fig=fig)

def run_real_data_analysis(
    seed_dirs,
    out_root,
    overlay_name=None,
    run_models=None,
    save_tables=True,
    save_plots=True,
):
    """
    Real-data analysis for high-dimensional datasets.

    - Aggregates metrics across all provided seed_dirs.
    - Produces a metrics table (RMSE, NLL, CRPS) with mean ± std across seeds.
    - Optionally produces bar plots with error bars for each metric.

    Notes:
        - Assumes each seed_dir is a *seed directory* that contains model subfolders.
        - Uses the same loader infrastructure as run_overlay_analysis, but
          does NOT assume a 1D x-grid or do any function-vs-x plotting.
    """
    import pandas as pd

    print(f"[real-analysis] analyzing seeds: {seed_dirs}")
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

    if not run_list:
        print("[real-analysis] No runs found.")
        return

    # Union of all models that appear in any run
    model_set = set()
    for run_key in run_list:
        model_set.update(metrics_test[run_key].keys())

    if run_models is None:
        run_models = sorted(model_set)
    else:
        run_models = [
            m for m in run_models
            if any(m in metrics_test[rk] for rk in run_list)
        ]

    print(f"[real-analysis] models: {run_models}")

    def mean_std(arr):
        if len(arr) == 0:
            return np.nan, np.nan
        v = np.asarray(arr, dtype=float)
        return float(np.mean(v)), float(np.std(v))

    rows = []
    for model in run_models:
        mse_list = []
        nll_list = []
        crps_list = []

        for run_key in run_list:
            mtest = metrics_test[run_key].get(model)
            if mtest is None:
                continue

            y_test = y_test_all[run_key].squeeze()
            mean_pred = mtest["mean"].squeeze()

            # MSE from mean prediction
            se = (mean_pred - y_test) ** 2
            mse = float(np.mean(se))
            mse_list.append(mse)

            # NLL / NLPD
            if "nlpd_kde" in mtest and np.isfinite(mtest["nlpd_kde"]).any():
                nll = float(np.nanmean(mtest["nlpd_kde"]))
            elif "nlpd_hist" in mtest and np.isfinite(mtest["nlpd_hist"]).any():
                nll = float(np.nanmean(mtest["nlpd_hist"]))
            else:
                nll = np.nan
            nll_list.append(nll)

            # CRPS
            if "crps" in mtest and np.isfinite(mtest["crps"]).any():
                crps_list.append(float(np.nanmean(mtest["crps"])))
            else:
                crps_list.append(np.nan)

        mse_mean, mse_std = mean_std(mse_list)
        nll_mean, nll_std = mean_std(nll_list)
        crps_mean, crps_std = mean_std(crps_list)

        rows.append(
            {
                "Model": model,
                "MSE_mean": mse_mean,
                "MSE_std": mse_std,
                "NLL_mean": nll_mean,
                "NLL_std": nll_std,
                "CRPS_mean": crps_mean,
                "CRPS_std": crps_std,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("MSE_mean")

    # Output dir
    out_dir = os.path.join(out_root, "real_analysis")
    os.makedirs(out_dir, exist_ok=True)

    if save_tables:
        csv_path = os.path.join(out_dir, "real_data_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"[real-analysis] saved metrics table to {csv_path}")

    if save_plots:
        # Simple bar plots with error bars for RMSE, NLL, CRPS
        fig, axes = plt.subplots(
            1, 3, figsize=(9.0, 3.0), dpi=150, sharex=False
        )
        metrics_specs = [
            ("MSE_mean", "MSE_std", "MSE"),
            ("NLL_mean", "NLL_std", "NLL"),
            ("CRPS_mean", "CRPS_std", "CRPS"),
        ]
        idx = np.arange(len(df))

        for ax, (m_mean, m_std, label) in zip(axes, metrics_specs):
            vals = df[m_mean].values
            errs = df[m_std].values
            ax.bar(idx, vals, yerr=errs, capsize=3)
            ax.set_title(label)
            ax.set_xticks(idx)
            ax.set_xticklabels(df["Model"].values, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        save_plot(out_dir, "real_data_metrics_bars", dpi=150, fig=fig)

    print("[real-analysis] done.")

# -----------------------------------------------------------------------------------
# TRAINING DRIVER
# -----------------------------------------------------------------------------------

def run_training_and_get_seed_dirs(args: argparse.Namespace) -> (str, list):
    cfg = baseline_params

    # data
    if args.dataset_mode == "toy":
        print("[main] Using toy dataset (paper specs).")
        input_data_dict = build_input_data_dict_from_toy(cfg, args.data_seed)
    else:
        if args.data_path is None:
            raise ValueError("--data-path required for dataset-mode=npz")
        print(f"[main] Loading npz dataset from {args.data_path}")
        input_data_dict = build_input_data_dict_from_npz(args.data_path)

    # model list & seeds
    model_type = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = args.seeds

    # training save path
    save_path = make_save_path(args.results_root, args.exp_name, args.timestamp)
    print(f"[main] Saving runs under: {save_path}")

    # run per-seed training
    for seed in seeds:
        print(f"\n[main] ==== Training seed {seed} ====")
        exp = SingleTaskExperiment(
            model_type=model_type,
            seeds=[seed],
            plot_dict=cfg["plot_dict"],
            model_dict=cfg["model_dict"]
        )
        exp.run_experiments(
            input_data_dict=input_data_dict,
            epochs=cfg["epochs"],
            beta_param_dict=cfg["cosine_beta_scheduler"],
            checkpoint_dicts=cfg["checkpoint_dict"],
            MC_train=cfg["MC_train"],
            MC_val=cfg["MC_val"],
            MC_test=cfg["MC_test"],
            analysis=False,
            save_switch=True,
            save_path=save_path,
            ensemble_epochs=cfg["ensemble_epochs"],
        )

    # return list of *seed dirs* (each contains model folders)
    seed_dirs = [os.path.join(save_path, f"seed_{s}") for s in seeds]
    return save_path, seed_dirs


# -----------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------

def main():
    args = parse_args()
    debug_data = "npz"
    if debug_data == "npz":
        # ===== DEBUG RIG – TEMPORARY =====
        args.mode = "train_and_analyze"          # train + analyze in one go
        args.dataset_mode = "npz"                # NOT 'realdata' – choices are ['toy', 'npz']

        args.data_path = os.path.join("data", "uci_npz", "airfoil_self_noise_feat_dim_0.npz")
        args.exp_name = "airfoil_debug_cosine_beta_max_1"

        # args.data_path = os.path.join("data", "uci_npz", "energy_efficiency_heating_feat_dim_0.npz")
        # args.exp_name = "energy_efficiency_heating_cosine_beta_max_0.01"

        args.analysis_type = "real"              # use the real-data analysis, not overlay
        args.seeds = [7, 8]                         # or [0,1,2] etc. if parse_args allows list
        # ==================================
    elif debug_data == "toy":
        # ===== DEBUG RIG – TEMPORARY (TOY TASK) =====
        args.mode = "train_and_analyze"      # train + analyze in one go
        args.dataset_mode = "toy"            # use toy generator
        args.exp_name = "toy_debug_2epochs"  # results/single_task_experiment/toy_debug_2epochs
        args.analysis_type = "overlay"       # 1D overlay analysis (default)
        args.data_seed = 0                   # seed for x-grid / toy splits
        args.seeds = [0]                     # training seed(s) = model + function seed for now
        # Optionally restrict models while debugging:
        # args.models = "IC_FDNet,LP_FDNet,BayesNet"
        # ===========================================

    # ------------------------------------------------------------------
    # Optional training phase
    # ------------------------------------------------------------------
    if args.mode in {"train", "train_and_analyze"}:
        # Trains for the requested seeds and returns their seed_* dirs.
        exp_root, new_seed_dirs = run_training_and_get_seed_dirs(args)
    else:
        exp_root, new_seed_dirs = os.path.join('results', 'single_task_experiment', args.exp_name), []

    # ------------------------------------------------------------------
    # Optional analysis phase
    # ------------------------------------------------------------------
    if args.mode in {"analyze", "train_and_analyze"}:
        # Decide which seed directories to analyze
        seed_dirs: list[str] = []

        # 1) Explicit --overlay-runs argument
        if args.overlay_runs is not None:
            # args.overlay_runs is a list of paths; each can be:
            #   - a seed directory:   .../seed_7
            #   - a parent directory: .../paper_repo  (containing seed_* subdirs)
            for p in args.overlay_runs:
                p = os.path.normpath(p)
                if not os.path.isdir(p):
                    raise ValueError(
                        f"--overlay-runs path does not exist or is not a directory: {p}"
                    )

                base = os.path.basename(p)
                if base.startswith("seed"):
                    # Directly a seed directory
                    seed_dirs.append(p)
                else:
                    # Treat as parent; collect immediate seed_* children
                    for name in os.listdir(p):
                        full = os.path.join(p, name)
                        if os.path.isdir(full) and name.startswith("seed"):
                            seed_dirs.append(full)

        # 2) Seeds from the training phase in this same call
        if not seed_dirs and new_seed_dirs:
            seed_dirs = list(new_seed_dirs)

        # 3) Fallback: discover seeds under the expected experiment root
        if not seed_dirs:
            if exp_root is None:
                exp_root = os.path.join(
                    args.results_root, "single_task_experiment", args.exp_name
                )
            if os.path.isdir(exp_root):
                for name in os.listdir(exp_root):
                    full = os.path.join(exp_root, name)
                    if os.path.isdir(full) and name.startswith("seed"):
                        seed_dirs.append(full)

        if not seed_dirs:
            raise ValueError(
                "No seed directories found for analysis. "
                "Use --overlay-runs or run with mode=train_and_analyze so that "
                "new results are available."
            )

        # Make paths unique and deterministic
        seed_dirs = sorted(set(seed_dirs))

        # Models to analyze (all paper models by default)
        model_list = list(baseline_params["model_dict"].keys())

        if args.analysis_type == "overlay":
            # Paper-style 1D overlay: do one run/seed at a time
            for sd in seed_dirs:
                seed_tag = Path(sd).name
                base_name = args.overlay_name or seed_tag
                overlay_name = args.overlay_name if args.overlay_name is not None else seed_tag

                run_overlay_analysis(
                    seed_dir=[sd],
                    out_root=sd,          # saves into that seed_* directory
                    overlay_name=overlay_name,
                    run_models=model_list,
                    save_tables=True,
                    save_plots=True,
                )
        else:
            # Real-data, multi-seed analysis: aggregate over all seed_dirs
            base_name = args.overlay_name or "real_data"

            # Put the real-data summary one level above the seed dirs
            common_root = os.path.commonpath(seed_dirs)
            if os.path.basename(common_root).startswith("seed"):
                common_root = os.path.dirname(common_root)

            # Feature dimension where the split occurs
            feat_dim = int(os.path.basename(args.data_path).replace(".npz", "").split("_")[-1])

            for seed_dir in seed_dirs:
                run_overlay_analysis(
                    seed_dir=seed_dir,
                    out_root=common_root,
                    overlay_name=base_name,
                    run_models=model_list,
                    save_tables=True,
                    save_plots=True,
                    feat_dim=feat_dim,
                )



if __name__ == "__main__":
    main()
