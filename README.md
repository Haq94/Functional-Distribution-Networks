# Functional Distribution Networks (FDN)

This repository contains the code for **Functional Distribution Networks (FDN)**.

It trains and evaluates FDN and strong baselines on:

- 1D toy regression families (step / sine / quadratic)
- UCI-style regression datasets (Airfoil, CCPP, Energy)

and reproduces the main figures/tables from the paper (MSE–Var scatter, ΔMSE/ΔVar/ΔCRPS, etc.).

> **Note:** `main.py` is intended to behave purely via command-line arguments.  
> Before packaging the artifact, remove or comment out any hard-coded debug logic
> (e.g., the `debug_data = "npz"` block) so that `parse_args()` fully controls behavior.

---

## 1. Quick start (TL;DR)

From the repo root:

```bash
# 1) Create and activate environment
conda env create -f environment.yml
conda activate fdn_env

# 2) Quick toy run (sanity check)
python main.py \
  --mode train_and_analyze \
  --dataset-mode toy \
  --analysis-type overlay \
  --exp-name toy_quickcheck \
  --data-seed 0 \
  --toy-func-seeds 24 25 26 \
  --models IC_FDNet,LP_FDNet,BayesNet,GaussHyperNet,MLPDropoutNet,DeepEnsembleNet \
  --seeds 0 1 2 \
  --beta-scheduler linear_beta_scheduler

# 3) Quick Airfoil run (sanity check)
python main.py \
  --mode train_and_analyze \
  --dataset-mode npz \
  --analysis-type real \
  --data-path data/uci_npz/airfoil_self_noise_feat_dim_0.npz \
  --exp-name airfoil_quickcheck \
  --models IC_FDNet,LP_FDNet,BayesNet,GaussHyperNet,MLPDropoutNet,DeepEnsembleNet \
  --seeds 3 4 5 \
  --beta-scheduler linear_beta_scheduler
