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
        choices=["overlay"],
        default="overlay",
        help=(
            "overlay: 1D toy-style overlay analysis (y vs x, shaded ID/OOD). "
            "real: high-dimensional real-data analysis (aggregate metrics across seeds)."
        ),
    )

    # ---- high-level mode ----
    parser.add_argument(
        "--mode",
        choices=["train", "analyze", "train_and_analyze", "seed_agg"],
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
    parser.add_argument(
        "--toy-func-seeds",
        type=int,
        nargs="+",
        default=[24, 25, 26],
        help="One or more seeds for toy function generation.",
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

    # ---- beta scheduler ----
    parser.add_argument(
    "--beta-scheduler",
    type=str,
    default='linear_beta_scheduler',
    help="Beta scheduler for runs.",
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


def find_seed_dirs(root: str):
    """
    Recursively find all 'seed_*' directories under `root`.

    This works both for layouts like:
        root/seed_7/...
    and nested toy layouts like:
        root/toy_seed_26/seed_7/...

    Returns a sorted list of absolute paths.
    """
    root = os.path.abspath(root)
    seed_dirs = []

    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.startswith("seed_"):
                seed_dirs.append(os.path.join(dirpath, d))

    seed_dirs = sorted(seed_dirs)
    print(f"[post-seed-agg] Found {len(seed_dirs)} seed_* dirs under {root}")
    for sd in seed_dirs:
        print(f"    {sd}")
    return seed_dirs


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
# OVERLAY ANALYSIS 
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
    
    ID_SHADE_COLOR = "0.8"   # darker gray
    ID_SHADE_ALPHA = 0.6     # more opaque

    # (1) MSE vs Var scatter + linear fits
    fig, ax = plt.subplots(
        1, 1,
        figsize=iclr_figsize(layout="double"),
        dpi=DPI,
    )

    vmin = min(store[n]["var"].min() for n in models_present)
    vmax = max(store[n]["var"].max() for n in models_present)
    xline = np.linspace(vmin, vmax, 200)

    # ideal line
    ax.plot([0, vmax], [0, vmax], "k--", lw=1, label="Ideal: MSE=Var")

    for name in models_present:
        arr = store[name]
        ax.scatter(
            arr["var"],
            arr["mse"],
            s=8,
            alpha=0.4,
            color=arr["color"],
        )
        a, b = arr["a"], arr["b"]
        ax.plot(
            xline,
            a + b * xline,
            lw=1.6,
            color=arr["color"],
            label=f'{arr["label"]} (a={a:.2g}, b={b:.2g})',
        )

    ax.set_xlabel("Predicted variance")
    ax.set_ylabel("Empirical MSE")
    ax.grid(alpha=0.3)

    # layout axes first
    fig.tight_layout()
    # carve out space at top for legend
    fig.subplots_adjust(top=0.80)

    # figure-level legend on top
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        fontsize=7,
    )

    save_plot(out_dir, "mse_vs_var_scatter", dpi=DPI, fig=fig)


    # (2) Calibration curve: binned MSE vs var
    fig, ax = plt.subplots(
        1, 1,
        figsize=iclr_figsize(layout="double"),
        dpi=DPI,
    )

    ylim = max(store[n]["var"].max() for n in models_present)
    ax.plot([0, ylim], [0, ylim], "k--", lw=1, label="Ideal: y=x")

    for name in models_present:
        mid, emse, pvar = binned_calibration(
            store[name]["var"],
            store[name]["mse"],
            nbins=20,
        )
        ax.plot(
            pvar, emse,
            "o-",
            ms=4,
            lw=1,
            color=store[name]["color"],
            label=store[name]["label"],
        )

    ax.set_xlabel("Predicted variance (bin mean)")
    ax.set_ylabel("Empirical MSE (bin mean)")
    ax.grid(alpha=0.3)

    # layout axes
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

    # figure-level legend above the axis
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        fontsize=7,
    )

    save_plot(out_dir, "calibration_curve", dpi=DPI, fig=fig)



    # (3) Risk–Coverage
    fig, ax = plt.subplots(
        1, 1,
        figsize=iclr_figsize(layout="double"),   # was "single"
        dpi=DPI,
    )

    for name in models_present:
        arr = store[name]
        ax.plot(
            arr["cov"],
            arr["risk"],
            lw=1.6,
            color=arr["color"],
            label=f'{arr["label"]} (AURC={arr["aurc"]:.3f})',
        )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (cumulative mean MSE)")
    ax.grid(alpha=0.3)

    # --- shared legend on top ---
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    save_plot(out_dir, "risk_coverage", dpi=DPI, fig=fig)


    # (4) Predictive mean vs x (with ground truth and ID region shaded)
    # -----------------------------------------------------------------
    fig_mean, ax_mean = plt.subplots(
        1, 1,
        figsize=iclr_figsize(layout="double"),
        dpi=DPI,
    )

    # Use any one model just to get the x-grid and y_true
    first_model = models_present[0]
    x_plot = x_feat_split
    y_true_plot = store[first_model]["y_true"]

    # Shade interpolation (ID) region
    ax_mean.axvspan(
        x_min, x_max,
        color=ID_SHADE_COLOR,
        alpha=ID_SHADE_ALPHA,
        zorder=0,
    )

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

    ax_mean.legend(
        loc="center right",
        bbox_to_anchor=(-0.32, 0.5),
        frameon=False,
        fontsize=7,
        borderaxespad=0.0,
    )

    fig_mean.tight_layout()
    save_plot(out_dir, "mean_vs_x_with_truth", dpi=DPI, fig=fig_mean)

    # (5) MSE, variance, and CRPS vs x (1×3 row, ID region shaded)
    # -------------------------------------------------------------
    have_crps = any(np.isfinite(store[n]["crps"]).any() for n in models_present)

    n_cols = 3 if have_crps else 2
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=iclr_figsize(layout="3x1"),
        dpi=DPI,
        sharex=True,
    )

    if n_cols == 1:
        axes = [axes]

    # Shade interpolation (ID) region in all subplots
    for ax in axes:
        ax.axvspan(
            x_min, x_max,
            color=ID_SHADE_COLOR,
            alpha=ID_SHADE_ALPHA,
            zorder=0,
        )

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

    # Put a single x-label on the middle axis for symmetry
    mid = len(axes) // 2
    for i, ax in enumerate(axes):
        ax.set_xlabel("x" if i == mid else "")

    # ---------- shared legend above the row ----------
    handles, labels = axes[0].get_legend_handles_labels()

    # optional: deduplicate labels (just in case)
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            uniq_handles.append(h)
            uniq_labels.append(lab)

    fig.legend(
        uniq_handles,
        uniq_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,          # tweak if needed
        frameon=False,
        fontsize=7,
    )

    # leave room at the top for the legend
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    save_plot(out_dir, "mse_var_crps_vs_x", dpi=DPI, fig=fig)


    # (6) Delta bar charts: ΔMSE / ΔVar / ΔCRPS as 1×3 row (left-to-right)
    if df_delta is not None:
        # --- Enforce a consistent model order -------------------------------
        # df_delta["Model"] likely has display names like "IC-FDNet"
        # model_colors is keyed by internal names like "IC_FDNet"
        all_labels = df_delta["Model"].values.tolist()

        ordered_labels = []
        for internal_name in model_colors.keys():
            display_name = internal_name.replace("_", "-")
            if display_name in all_labels:
                ordered_labels.append(display_name)

        # If we found a non-empty ordering, reorder df_delta to match
        if ordered_labels:
            df_delta = (
                df_delta.set_index("Model")
                        .loc[ordered_labels]
                        .reset_index()
            )

        # --------------------------------------------------------------------
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
            figsize=iclr_figsize("3x1"),  # tweak width/height as needed
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

    fig, axes = plt.subplots(
        1, 2,
        figsize=iclr_figsize(layout="double"),
        dpi=DPI,
        sharey=True,
    )
    all_vars_i, all_vars_o = [], []

    for name in models_present:
        arr = store[name]
        c = arr["color"]
        lab = arr["label"]

        x = x_feat_split           # shape [N]
        v = arr["var"]             # shape [N]
        m = arr["mse"]             # shape [N]
        interp_mask = arr["interp_mask"]
        extrap_mask = arr["extrap_mask"]

        # you actually don't need sorting for scatter, but it's harmless:
        order = np.argsort(x)
        vi = v[order][interp_mask[order]]
        mi = m[order][interp_mask[order]]
        vo = v[order][extrap_mask[order]]
        mo = m[order][extrap_mask[order]]

        def _format_rho(r):
            if r is None or np.isnan(r):
                return "N/A"
            return f"{r:.2f}"

        rho_i = safe_spearman(vi, mi)
        rho_o = safe_spearman(vo, mo)

        # Build a mathtext label with subscripts
        rho_label = (
            rf"$\rho_{{\mathrm{{ID}}}}$={_format_rho(rho_i)}, "
            rf"$\rho_{{\mathrm{{OOD}}}}$={_format_rho(rho_o)}"
        )

        # ID panel: no legend entry
        axes[0].scatter(
            vi, mi,
            s=10,
            alpha=0.35,
            color=c,
            label="_nolegend_",
        )

        # OOD panel: legend label uses LaTeX-style subscripts
        axes[1].scatter(
            vo, mo,
            s=10,
            alpha=0.35,
            color=c,
            label=f"{lab} ({rho_label})",
        )


        all_vars_i.append(vi)
        all_vars_o.append(vo)

    _add_ref_line(axes[0], all_vars_i)
    _add_ref_line(axes[1], all_vars_o)

    for ax in axes:
        ax.set_xlabel("Variance")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("MSE")
    axes[0].set_title("Interp (ID)")
    axes[1].set_title("Extrap (OOD)")

    # Single legend, outside on the left, based on OOD axis
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(-0.30, 0.5),
        frameon=False,
        fontsize=7,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    save_plot(out_dir, "scatter_interp_extrap", dpi=DPI, fig=fig)



def post_seed_agg(args):
    """
    Aggregate metrics over existing seed_* runs for a given experiment.

    If args.debug_data == "toy":
        - look for toy_seed_* subdirectories under
          results/single_task_experiment/args.exp_name
        - aggregate separately inside each toy_seed_*.

    Otherwise:
        - aggregate directly in results/single_task_experiment/args.exp_name
          over all seed_* folders.
    """
    root = Path("results") / "single_task_experiment" / args.exp_name
    is_toy = getattr(args, "dataset_mode", "") == "toy"

    if is_toy:
        toy_roots = sorted(
            p for p in root.iterdir()
            if p.is_dir() and p.name.startswith("toy_seed_")
        )
        if not toy_roots:
            print(f"[post_seed_agg] No toy_seed_* folders under {root}")
            return
        for tr in toy_roots:
            _aggregate_one_root(tr)
    else:
        _aggregate_one_root(root)


def _aggregate_one_root(seed_root: Path) -> None:
    """
    seed_root: folder that contains seed_* subdirs,
               each with paper_figs/summary_main.csv (and optionally
               paper_figs/delta_id_to_ood.csv).

    Writes:
      - seed_root/summary_seed_agg_by_model.csv
      - seed_root/summary_seed_representative_runs_by_model.csv
      - seed_root/summary_seed_representative_FDN.txt
      - seed_root/seed_agg_scatter_MSE_OOD_vs_Var_OOD.png
    """
    seed_dirs = sorted(
        p for p in seed_root.iterdir()
        if p.is_dir() and p.name.startswith("seed_")
    )

    if not seed_dirs:
        print(f"[post_seed_agg] No seed_* folders under {seed_root}")
        return

    dfs = []
    for sd in seed_dirs:
        figs_dir = sd / "paper_figs"

        main_path = figs_dir / "summary_main.csv"
        if not main_path.exists():
            print(f"[post_seed_agg] Skipping {sd}: missing summary_main.csv")
            continue

        df_main = pd.read_csv(main_path)

        # Try to merge in ID/OOD metrics if available
        delta_path = figs_dir / "delta_id_to_ood.csv"
        if delta_path.exists():
            df_delta = pd.read_csv(delta_path)
            # Merge on 'Model'; summary_main and delta only overlap on 'Model'
            df = df_main.merge(df_delta, on="Model", how="left")
        else:
            df = df_main

        df["seed_dir"] = sd.name
        dfs.append(df)

    if not dfs:
        print(f"[post_seed_agg] No usable CSV files found under {seed_root}")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # Identify numeric metric columns (drop the seed_dir index itself)
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if c != "seed_dir"]

    if not metric_cols:
        print(f"[post_seed_agg] No numeric metric columns found under {seed_root}")
        return

    # ------------------------------------------------------------------
    # 1) Aggregate stats by model
    # ------------------------------------------------------------------
    grouped = df_all.groupby("Model")[metric_cols]

    df_mean = grouped.mean().add_suffix("_mean")
    df_std  = grouped.std(ddof=0).add_suffix("_std")
    df_med  = grouped.median().add_suffix("_median")

    stats = pd.concat([df_mean, df_std, df_med], axis=1)
    stats.to_csv(seed_root / "summary_seed_agg_by_model.csv")

    # ------------------------------------------------------------------
    # 2) Representative row (seed) per MODEL
    #    (same idea as before, just using the extended metric set)
    # ------------------------------------------------------------------
    repr_rows = []
    for model, df_m in df_all.groupby("Model"):
        med_vec = df_m[metric_cols].median(axis=0)
        diff = ((df_m[metric_cols] - med_vec) ** 2).sum(axis=1)
        best_idx = diff.idxmin()

        row = df_m.loc[best_idx].copy()
        row["dist_to_model_median"] = float(diff.loc[best_idx])
        repr_rows.append(row)

    df_repr = pd.DataFrame(repr_rows)
    df_repr.to_csv(
        seed_root / "summary_seed_representative_runs_by_model.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3) Single representative SEED for the FDNs
    #    (IC-FDNet + LP-FDNet jointly, non-cherry-picked)
    # ------------------------------------------------------------------
    FDN_MODELS = ["IC-FDNet", "LP-FDNet"]

    df_fd = df_all[df_all["Model"].isin(FDN_MODELS)].copy()
    if not df_fd.empty:
        # Use a core subset of metrics; fall back to all metrics if missing
        desired_metrics = ["MSE(OOD)", "Var(OOD)", "ΔCRPS", "AURC"]
        metrics_sel = [m for m in desired_metrics if m in df_fd.columns]
        if not metrics_sel:
            metrics_sel = metric_cols  # fallback: all numeric metrics

        # median vector for each FDN model
        med_by_model: dict[str, pd.Series] = {}
        for m in FDN_MODELS:
            df_m = df_fd[df_fd["Model"] == m]
            if df_m.empty:
                continue
            med_by_model[m] = df_m[metrics_sel].median(axis=0)

        # distance of each seed_dir to these medians (summed over FDN models)
        seed_dists: dict[str, float] = {}
        for seed_name, df_seed in df_fd.groupby("seed_dir"):
            total = 0.0
            used_any = False
            for m in FDN_MODELS:
                if m not in med_by_model:
                    continue
                row_m = df_seed[df_seed["Model"] == m]
                if row_m.empty:
                    continue
                vec = row_m[metrics_sel].iloc[0]
                diff = vec - med_by_model[m]
                total += float((diff ** 2).sum())
                used_any = True
            if used_any:
                seed_dists[seed_name] = total

        if seed_dists:
            best_seed = min(seed_dists, key=seed_dists.get)
            out_path = seed_root / "summary_seed_representative_FDN.txt"
            with open(out_path, "w") as f:
                f.write(
                    "Representative seed (closest to FDN median):\n"
                    f"  seed_dir = {best_seed}\n"
                    f"  dist     = {seed_dists[best_seed]:.6g}\n"
                )
            print(
                f"[post_seed_agg] FDN-representative seed for {seed_root}: "
                f"{best_seed}  (dist={seed_dists[best_seed]:.3g})"
            )
        else:
            print(
                f"[post_seed_agg] Could not compute FDN representative seed "
                f"(no overlapping metrics) under {seed_root}"
            )
    else:
        print(f"[post_seed_agg] No IC-FDNet / LP-FDNet rows under {seed_root}")

    # ------------------------------------------------------------------
    # 4) Scatter over seeds: MSE(OOD) vs Var(OOD)
    # ------------------------------------------------------------------
    if "MSE(OOD)" in df_all.columns and "Var(OOD)" in df_all.columns:
        model_colors = {
            "IC-FDNet": "#1f77b4",
            "LP-FDNet": "#ff7f0e",
            "BayesNet": "#2ca02c",
            "GaussHyperNet": "#d62728",
            "MLPDropoutNet": "#9467bd",
            "DeepEnsembleNet": "#8c564b",
        }

        fig, ax = plt.subplots(
            1, 1,
            figsize=iclr_figsize(layout="double"),
            dpi=DPI,
        )

        for model, df_m in df_all.groupby("Model"):
            ax.scatter(
                df_m["Var(OOD)"],
                df_m["MSE(OOD)"],
                s=20,
                alpha=0.6,
                label=model,
                color=model_colors.get(model, "0.4"),
            )

        ax.set_xlabel("Var(OOD)")
        ax.set_ylabel("MSE(OOD)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()

        out_fig = seed_root / "seed_agg_scatter_MSE_OOD_vs_Var_OOD.png"
        fig.savefig(out_fig, dpi=DPI)
        plt.close(fig)

    print(f"[post_seed_agg] Aggregated {len(seed_dirs)} seeds under {seed_root}")



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
    toy_func_seeds = args.toy_func_seeds

    # training save path
    save_path = make_save_path(args.results_root, args.exp_name, args.timestamp)
    print(f"[main] Saving runs under: {save_path}")

    # run per-seed training
    for seed in seeds:
        print(f"\n[main] ==== Training seed {seed} ====")
        if args.dataset_mode == "toy":
            for seed_function in toy_func_seeds:
                save_path_toy_seed = os.path.join(save_path, f"toy_seed_{seed_function}")
                exp = SingleTaskExperiment(
                    model_type=model_type,
                    seeds=[seed],
                    plot_dict=cfg["plot_dict"],
                    model_dict=cfg["model_dict"],
                    seed_function=seed_function
                )
                exp.run_experiments(
                    input_data_dict=input_data_dict,
                    epochs=cfg["epochs"],
                    beta_param_dict=cfg[args.beta_scheduler],
                    checkpoint_dicts=cfg["checkpoint_dict"],
                    MC_train=cfg["MC_train"],
                    MC_val=cfg["MC_val"],
                    MC_test=cfg["MC_test"],
                    analysis=False,
                    save_switch=True,
                    save_path=save_path_toy_seed,
                    ensemble_epochs=cfg["ensemble_epochs"],
                )
        else:
                exp = SingleTaskExperiment(
                    model_type=model_type,
                    seeds=[seed],
                    plot_dict=cfg["plot_dict"],
                    model_dict=cfg["model_dict"]
                )
                exp.run_experiments(
                    input_data_dict=input_data_dict,
                    epochs=cfg["epochs"],
                    beta_param_dict=cfg["linear_beta_scheduler"],
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
    if args.dataset_mode == "toy":
        seed_dirs = [
            os.path.join(save_path, f"toy_seed_{ts}", f"seed_{s}")
            for s in seeds
            for ts in toy_func_seeds
        ]
    else:
        seed_dirs = [
            os.path.join(save_path, f"seed_{s}")
            for s in seeds
        ]

    return save_path, seed_dirs


# -----------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------

def main():
    args = parse_args()
    # debug_data = "npz"
    # if debug_data == "npz":
    #     # ===== DEBUG RIG – TEMPORARY =====
    #     args.mode = "train_and_analyze"          # train + analyze in one go
    #     args.dataset_mode = "npz"                # NOT 'realdata' – choices are ['toy', 'npz']

    #     run = 2
    #     if run == 1:
    #         args.data_path = os.path.join("data", "uci_npz", "airfoil_self_noise_feat_dim_0.npz")
    #         args.exp_name = "airfoil_debug_linear_beta_max_1"

    #     elif run == 2:
    #         args.data_path = os.path.join("data", "uci_npz", "ccpp_power_plant_feat_dim_0.npz")
    #         args.exp_name = "ccpp_power_plant_linear_beta_max_1"

    #     elif run == 3:
    #         args.data_path = os.path.join("data", "uci_npz", "energy_efficiency_heating_feat_dim_0.npz")
    #         args.exp_name = "energy_efficiency_heating_linear_beta_max_1"

    #     # args.data_path = os.path.join("data", "uci_npz", "energy_efficiency_heating_feat_dim_0.npz")
    #     # args.exp_name = "energy_efficiency_heating_cosine_beta_max_0.01"

    #     args.analysis_type = "real"              # use the real-data analysis, not overlay
    #     args.seeds = [n for n in range(3,100)]                         # or [0,1,2] etc. if parse_args allows list
    #     # ==================================
    # elif debug_data == "toy":
    #     # ===== DEBUG RIG – TEMPORARY (TOY TASK) =====
    #     args.mode = "seed_agg"      # train + analyze in one go
    #     args.dataset_mode = "toy"            # use toy generator
    #     args.exp_name = "toy_run_linear_beta_max_1"          # results/single_task_experiment/args.exp_name
    #     args.analysis_type = "overlay"       # 1D overlay analysis (default)
    #     args.data_seed = 0                   # seed for x-grid / toy splits
    #     args.toy_func_seeds = [24, 25, 26]      # seed for toy-function sampling
    #     args.seeds = [n for n in range(18,20)]                      # training seed(s) = model + function seed for now
    #     args.beta_scheduler = 'linear_beta_scheduler'
    #     # Optionally restrict models while debugging:
    #     # args.models = "IC_FDNet,LP_FDNet,BayesNet"
    #     # ===========================================

    if args.mode == "seed_agg":
        post_seed_agg(args)
        return

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
                    elif name.startswith("toy_seed"):
                        seed_dirs = seed_dirs + [os.path.join(exp_root, name, name_sd) for name_sd in os.listdir(full)]

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
                    seed_dir=sd,
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
