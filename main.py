#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from datetime import datetime
import re

import numpy as np
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
        default="paper_repro",
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
        default=[7, 8, 9],
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
    seed_dirs,
    out_root,
    overlay_name=None,
    run_models=None,
    save_tables=True,
    save_plots=True,
):
    """
    Paper-style overlay analysis: **one task / run at a time**.

    Each element of seed_dirs should look like:
      results/single_task_experiment/<exp_name>/seed_<seed>
    and must contain subfolders named by model type (IC_FDNet, ...).

    For the main paper figs, call this separately for each task/seed:
      - step
      - sine
      - quad
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

    out_dir = os.path.join(out_root, "paper_figs")
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

        mse = (mean_pred - y_true) ** 2
        var = y_hat_var.squeeze()

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

    import pandas as pd

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
        import pandas as pd
        rows = []
        for name in models_present:
            arr = store[name]
            x = arr["x"]
            var = arr["var"]
            mse = arr["mse"]
            crps = arr["crps"]
            interp_mask = arr["interp_mask"]
            extrap_mask = arr["extrap_mask"]

            def region_stats(mask, v, m, c):
                if not np.any(mask):
                    return np.nan, np.nan, np.nan
                return (
                    float(np.mean(m[mask])),
                    float(np.mean(v[mask])),
                    float(np.mean(c[mask])) if np.isfinite(c).any() else np.nan,
                )

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
    # PLOTS (same family as delete.py)
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
    x_plot = store[first_model]["x"]
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
        x = arr["x"]
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
        x = arr["x"]
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
        x = arr["x"]
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
            x = arr["x"]
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

    if args.mode in {"train", "train_and_analyze"}:
        exp_root, new_seed_dirs = run_training_and_get_seed_dirs(args)
    else:
        exp_root, new_seed_dirs = None, None

    if args.mode in {"analyze", "train_and_analyze"}:
        # Determine which seed directories to analyze
        if args.overlay_runs is not None:
            seed_dirs = args.overlay_runs
        elif new_seed_dirs is not None:
            seed_dirs = new_seed_dirs
        else:
            raise ValueError("No seed dirs provided for overlay analysis.")

        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        for sd in seed_dirs:
            # For train_and_analyze, sd is something like
            #   <results_root>/single_task_experiment/<exp_name>/seed_<k>

            base_name = args.overlay_name or "overlay"
            seed_tag = os.path.basename(sd)
            overlay_name = f"{base_name}_{seed_tag}" if len(seed_dirs) > 1 else base_name

            run_overlay_analysis(
                seed_dirs=[sd],
                out_root=sd,
                overlay_name=overlay_name,
                run_models=model_list,
                save_tables=True,
                save_plots=True,
            )


if __name__ == "__main__":
    main()
