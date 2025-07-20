# ✅ FDN Paper Submission Checklist — *ICLR 2026*

---

## 🧩 Phase 1: Core Experiments (Toy Tasks)
These establish performance and uncertainty estimates on controlled 1D functions.

- [x] Run FDN, BayesNet, GaussHyperNet, DeepEnsemble, HyperNet, MLP on toy regression.
- [x] Save RMSE, NLL, MSE, bias, var, std, residuals.
- [x] Plot uncertainty (mean ± std), residuals, etc.
- [x] Add reproducible training framework + seed handling.
- [x] Fix variance blow-up in LP-FDN with `softplus`.

---

## 🔄 Phase 2: Learning & Data Curves

### 📉 Learning Curves
- [ ] Train models with varying `epochs` (e.g., 100, 500, 1000).
- [ ] Plot RMSE/NLL over epochs for each model.
- [ ] Compare convergence speed.

### 📊 Data Efficiency
- [ ] Vary `frac_train` (e.g., 10%, 30%, 50%, 70%).
- [ ] Compare model performance vs. context size.
- [ ] Highlight robustness of FDN with fewer points.

---

## 📊 Phase 3: Calibration & OOD Analysis

### 📏 Calibration
- [ ] Implement calibration curve: predicted confidence vs. empirical error.
- [ ] Compute regression ECE (expected calibration error).
- [ ] Compare FDN vs. baselines on calibration.

### 🌐 OOD / Extrapolation
- [x] Plot variance, NLL inside/outside training region.
- [ ] Highlight behavior far from context range (e.g., |x| > 5).
- [ ] Create metric breakdowns: interpolation vs. extrapolation.

---

## 📁 Phase 4: Real-World Evaluation

- [ ] Integrate 1D UCI regression datasets: Boston, Energy, Naval, Protein, etc.
- [ ] Add experiment runner for real data.
- [ ] Compare RMSE/NLL across models.
- [ ] Test generalization with small data.

---

## 🧪 Phase 5: Ablations & Model Variants

- [ ] Ablate KL loss: train FDN with `beta=0`.
- [ ] FDN w/o input conditioning (shared z).
- [ ] Compare IC-FDN vs LP-FDN (done?).
- [ ] Swap `softplus` with `exp` to observe effect on stability.
- [ ] Try shared vs. per-layer conditioning.

---

## 🖼️ Phase 6: Plotting + Paper

- [x] Standardize plot colors by model type.
- [ ] Create final figures for: RMSE/NLL comparison, uncertainty plots, calibration curves.
- [ ] Tables: RMSE, NLL across models/seeds.
- [ ] Begin paper writing: intro + methods + experiments.
- [ ] Create `analysis/` scripts for visualizations.
- [ ] Export all plots to PDF/PNG for LaTeX.
