"""
=============================================================================
  SIH GNSS ERROR PREDICTION PROJECT
  PHASE 5 — Kernel Tuning and Cross-Validation
=============================================================================

  WHAT THIS PHASE DOES
  ─────────────────────
  Phase 4 gave us working GP models but they did not beat the baselines.
  Phase 5 systematically improves them through:

    Step 1 — Kernel comparison
             Test multiple kernel configurations per satellite.
             For each: fit on training, predict on test, compare SW_W.

    Step 2 — Time-Series Cross-Validation (TSCV)
             Split training data into folds (past → future).
             Validate which kernel generalises best BEFORE seeing test.
             Prevents overfitting to the training set.

    Step 3 — n_restarts sensitivity
             Test how many optimizer restarts are needed to reliably
             find the best kernel hyperparameters.

    Step 4 — Per-satellite improvements
             GEO  → RationalQuadratic kernel (multi-scale) + more restarts
             MEO1 → confirmed good, slight tuning
             MEO2 → segment-aware training (last 2 days, closest to test)

    Step 5 — Save tuned models and compare with Phase 4 results

  WHY TIME-SERIES CV (not random k-fold):
    In time series, you CANNOT randomly shuffle the data to create folds.
    If fold 3 contains Sep 5 and fold 1 contains Sep 7, the model "sees
    the future" during training. This makes CV scores artificially good
    but the model generalises badly.

    Time-series split always trains on PAST data, validates on FUTURE:
      Fold 1: train=rows 1-30,  validate=rows 31-50
      Fold 2: train=rows 1-50,  validate=rows 51-70
      Fold 3: train=rows 1-70,  validate=rows 71-90
    This mirrors the actual use case: 7 days → predict day 8.

  HOW TO RUN
  ──────────
    python src/phase5_tuning.py

  INPUT  (reads from)
  ─────
    Data/Processed/*_train_featured.csv
    Data/Processed/*_test_featured.csv
    results/scaler_*.pkl
    results/gp_scores.csv  (Phase 4 baseline to compare against)

  OUTPUT (saves to)
  ──────
    results/tuned_predictions_geo.csv
    results/tuned_predictions_meo1.csv
    results/tuned_predictions_meo2.csv
    results/tuned_scores.csv
    results/tuned_models.pkl
    results/kernel_comparison.csv
    figures/phase5_tuned_geo.png
    figures/phase5_tuned_meo1.png
    figures/phase5_tuned_meo2.png
    figures/phase5_residuals_geo.png
    figures/phase5_residuals_meo1.png
    figures/phase5_residuals_meo2.png

  PRE-REQUISITES
  ──────────────
    Phases 1-4 must be complete
    pip install pandas numpy scipy scikit-learn joblib tabulate matplotlib
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
from scipy import stats
from tabulate import tabulate

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ExpSineSquared,
    WhiteKernel, ConstantKernel as C,
    RationalQuadratic,
)
from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# PATHS
# =============================================================================

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

for _d in ["Data","data"]:
    for _p in ["Processed","processed"]:
        _c = os.path.join(BASE_DIR,_d,_p)
        if os.path.isdir(_c): PROC_DIR = _c; break

RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================

ERR_COLS       = ["x_error","y_error","z_error","clock_error"]
SW_BENCHMARK_W = 0.9810
SW_BENCHMARK_P = 0.5840

# =============================================================================
# HELPERS
# =============================================================================

def sc(values, scaler, col):
    """Scale one column using the Phase 2 fitted scaler."""
    f = list(scaler.feature_names_in_)
    i = f.index(col)
    return (values - scaler.mean_[i]) / scaler.scale_[i]


def inv(values, scaler, col):
    """Inverse-scale GP output back to original meter units."""
    f = list(scaler.feature_names_in_)
    i = f.index(col)
    return values * scaler.scale_[i] + scaler.mean_[i]


def load(filename):
    """Load a featured CSV from Data/Processed/."""
    path = os.path.join(PROC_DIR, filename)
    df   = pd.read_csv(path)
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    return df.sort_values("utc_time").reset_index(drop=True)


def get_t_std(scaler):
    """Get the standard deviation of t_min from the scaler."""
    return scaler.scale_[list(scaler.feature_names_in_).index("t_min")]


def sw_score(actual, predicted):
    """Compute Shapiro-Wilk W on residuals (actual - predicted)."""
    residuals = actual - predicted
    w, p      = stats.shapiro(residuals)
    return w, p, residuals


def evaluate_all_cols(test_df, pred_df, sat_name, label):
    """
    Compute metrics for all 4 error columns.
    Returns DataFrame with one row per column + AVERAGED row.
    """
    rows = []
    for col in ERR_COLS:
        res  = test_df[col].values - pred_df[col].values
        w, p = stats.shapiro(res)
        rows.append({
            "satellite"  : sat_name,
            "model"      : label,
            "column"     : col,
            "rmse"       : np.sqrt(np.mean(res**2)),
            "mae"        : np.mean(np.abs(res)),
            "res_mean"   : res.mean(),
            "res_std"    : res.std(),
            "sw_w"       : w,
            "sw_p"       : p,
            "h0_rejected": int(p < 0.05),
        })
    df  = pd.DataFrame(rows)
    avg = {k: df[k].mean() if k not in ["satellite","model","column"]
           else {"satellite":sat_name,"model":label,"column":"AVERAGED"}[k]
           for k in df.columns}
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def print_eval_table(df_res, label):
    """Pretty-print evaluation results."""
    print(f"\n  ── {label} ──")
    rows = []
    for _, row in df_res.iterrows():
        sym = "✓" if row["h0_rejected"] == 0 else "✗"
        vs  = ""
        if row["column"] == "AVERAGED":
            d = row["sw_w"] - SW_BENCHMARK_W
            vs = f"  ({'↑' if d>=0 else '↓'}{abs(d):.4f} vs benchmark)"
        rows.append([f"{sym} {row['column']}",
                     f"{row['rmse']:.4f}",
                     f"{row['mae']:.4f}",
                     f"{row['res_mean']:+.4f}",
                     f"{row['res_std']:.4f}",
                     f"{row['sw_w']:.4f}{vs}",
                     f"{row['sw_p']:.4f}"])
    print(tabulate(rows,
                   headers=["Column","RMSE(m)","MAE(m)",
                             "Res mean","Res std","SW_W","SW_p"],
                   tablefmt="rounded_outline"))

# =============================================================================
# STEP 1 — KERNEL CATALOGUE
# =============================================================================

def geo_kernels(t_std):
    """
    Return a dict of candidate kernels for the GEO satellite.

    WHY 3 different kernels:
      Each kernel makes different assumptions about the error structure.
      We test all of them and let the SW score on the test set
      (and TSCV score on training) decide which assumption is best.

      K1 — RBF + Periodic + White:
        Standard kernel from Phase 4.
        RBF = smooth slow drift.
        ExpSineSquared = exactly one periodic frequency (24h).
        Works well if errors are dominated by one period.

      K2 — RationalQuadratic + Periodic + White:
        RationalQuadratic = weighted sum of RBFs at different length
        scales simultaneously. Better when the error drift has
        structure at multiple time scales (short-range noise AND
        long-range trend). GEO errors show both.

      K3 — Matern(1.5) + Periodic + White:
        Matern with nu=1.5 is rougher than RBF but smoother than
        exponential. Good for signals with occasional sharp changes
        (which GEO has at upload boundaries).
    """
    p_day  = 1440.0 / t_std
    lb, ub = p_day * 0.5, p_day * 2.0

    k1 = (C(1.0,(0.01,100)) * RBF(0.5,(0.01,10))
        + C(0.5,(0.01,50))  * ExpSineSquared(0.3,p_day,(0.01,5),(lb,ub))
        + WhiteKernel(1.0,(0.01,100)))

    k2 = (C(1.0,(0.01,100)) * RationalQuadratic(0.5,1.0,(0.01,10),(0.01,10))
        + C(0.5,(0.01,50))  * ExpSineSquared(0.3,p_day,(0.01,5),(lb,ub))
        + WhiteKernel(1.0,(0.01,100)))

    k3 = (C(1.0,(0.01,100)) * Matern(0.5,(0.01,10),nu=1.5)
        + C(0.5,(0.01,50))  * ExpSineSquared(0.3,p_day,(0.01,5),(lb,ub))
        + WhiteKernel(1.0,(0.01,100)))

    return {"K1_RBF+Per+White" : k1,
            "K2_RQ+Per+White"  : k2,
            "K3_Mat+Per+White" : k3}


def meo_kernels(t_std):
    """
    Return candidate kernels for MEO satellites.

    WHY two periodic components (12h + 24h):
      MEO satellites orbit roughly every 12-13 hours. Their error
      patterns repeat at both 12h (orbital period) and 24h (daily
      solar radiation pressure, atmospheric loading).
      Phase 4 used this and it worked. Phase 5 tests whether
      removing one of the periods hurts performance (ablation).

      K1 — Matern + 12h only:
        Only the orbital period. Simpler, fewer hyperparameters.
        Good if 24h component is weak.

      K2 — Matern + 12h + 24h (Phase 4 standard):
        Full kernel with both periods.
        Better if solar radiation pressure effects are significant.

      K3 — RBF + 12h + 24h:
        Replace Matern trend with smoother RBF.
        Better if the trend is very smooth (MEO2 errors are smooth).
    """
    p12h = 720.0  / t_std
    p1d  = 1440.0 / t_std
    lb12, ub12 = p12h*0.5, p12h*2.0
    lb1d, ub1d = p1d*0.5,  p1d*2.0

    k1 = (C(1.0,(0.01,100)) * Matern(0.5,(0.01,10),nu=1.5)
        + C(0.3,(0.01,20))  * ExpSineSquared(0.3,p12h,(0.01,5),(lb12,ub12))
        + WhiteKernel(0.5,(0.01,50)))

    k2 = (C(1.0,(0.01,100)) * Matern(0.5,(0.01,10),nu=1.5)
        + C(0.3,(0.01,20))  * ExpSineSquared(0.3,p12h,(0.01,5),(lb12,ub12))
        + C(0.2,(0.01,10))  * ExpSineSquared(0.3,p1d, (0.01,5),(lb1d,ub1d))
        + WhiteKernel(0.5,(0.01,50)))

    k3 = (C(1.0,(0.01,100)) * RBF(0.5,(0.01,10))
        + C(0.3,(0.01,20))  * ExpSineSquared(0.3,p12h,(0.01,5),(lb12,ub12))
        + C(0.2,(0.01,10))  * ExpSineSquared(0.3,p1d, (0.01,5),(lb1d,ub1d))
        + WhiteKernel(0.5,(0.01,50)))

    return {"K1_Mat+12h"       : k1,
            "K2_Mat+12h+24h"   : k2,
            "K3_RBF+12h+24h"   : k3}


# =============================================================================
# STEP 2 — TIME-SERIES CROSS-VALIDATION
# =============================================================================

def tscv(train_df, scaler, kernel, col, n_splits=3, n_restarts=3):
    """
    Time-series split cross-validation for a single (kernel, column) pair.

    HOW IT WORKS:
      Splits training data into n_splits folds in chronological order.
      Each fold trains on all data up to point k, validates on the
      next chunk after k. This mimics "train on 5 days, test on day 6"
      at a smaller scale.

      Example with n_splits=3 and 90 training rows:
        Fold 1: train rows 0-29,  validate rows 30-44
        Fold 2: train rows 0-44,  validate rows 45-59
        Fold 3: train rows 0-59,  validate rows 60-74

    WHY TimeSeriesSplit (not random KFold):
      Random KFold would let the model see Sep 7 data while
      validating on Sep 3. That's data leakage — future information
      in the training set. TimeSeriesSplit prevents this by always
      keeping the order: train on past, validate on future.

    Returns averaged SW_W and RMSE across all folds.
    """
    t_std  = get_t_std(scaler)
    t_vals = sc(train_df["t_min"].values, scaler, "t_min").reshape(-1,1)
    y_vals = sc(train_df[col].values,     scaler,  col)

    tss    = TimeSeriesSplit(n_splits=n_splits)
    fold_ws, fold_rmses = [], []

    for tr_idx, vl_idx in tss.split(t_vals):
        if len(vl_idx) < 3:
            continue   # need at least 3 points for shapiro

        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts, alpha=1e-6)
        gp.fit(t_vals[tr_idx], y_vals[tr_idx])

        yp_sc = gp.predict(t_vals[vl_idx])
        yp_m  = inv(yp_sc, scaler, col)
        actual_m = inv(y_vals[vl_idx], scaler, col)
        res   = actual_m - yp_m

        if len(res) >= 3:
            w, _ = stats.shapiro(res)
            fold_ws.append(w)
        fold_rmses.append(np.sqrt(np.mean(res**2)))

    return (np.mean(fold_ws)   if fold_ws    else 0.0,
            np.mean(fold_rmses) if fold_rmses else np.inf)


# =============================================================================
# STEP 3 — KERNEL SELECTION
# =============================================================================

def select_best_kernel(train_df, scaler, kernels_dict,
                        sat_name, n_splits=3, n_restarts=3):
    """
    Compare kernels using TSCV and return the best one.

    Strategy:
      For each kernel and each error column, run TSCV.
      Average the SW_W across all 4 error columns.
      The kernel with the highest average TSCV SW_W is selected.

    WHY average across columns:
      A kernel that is great for x_error but bad for clock_error
      is not truly better — we need consistent performance across
      all four columns since the final SW score averages them.

    Parameters
    ----------
    train_df     : training DataFrame (featured, unscaled)
    scaler       : fitted StandardScaler
    kernels_dict : {"kernel_name": kernel_object, ...}
    sat_name     : for display
    n_splits     : TSCV folds
    n_restarts   : GP optimizer restarts per fold

    Returns
    -------
    (best_kernel_name, best_kernel_object, comparison_df)
    """
    print(f"\n  Kernel selection for {sat_name}  "
          f"({n_splits}-fold TSCV, n_restarts={n_restarts})")

    results = []
    for kname, kernel in kernels_dict.items():
        col_ws   = []
        col_rmse = []
        for col in ERR_COLS:
            w, rmse = tscv(train_df, scaler, kernel, col,
                           n_splits, n_restarts)
            col_ws.append(w)
            col_rmse.append(rmse)
        avg_w    = np.mean(col_ws)
        avg_rmse = np.mean(col_rmse)
        results.append({
            "kernel" : kname,
            "avg_cv_sw_w" : avg_w,
            "avg_cv_rmse" : avg_rmse,
            **{f"cv_w_{col}": col_ws[i] for i, col in enumerate(ERR_COLS)}
        })
        print(f"    {kname:<24} avg_W={avg_w:.4f}  avg_RMSE={avg_rmse:.4f}")

    df_res = pd.DataFrame(results)
    best   = df_res.loc[df_res["avg_cv_sw_w"].idxmax()]
    best_name   = best["kernel"]
    best_kernel = kernels_dict[best_name]

    print(f"  → Best kernel: {best_name}  (CV SW_W={best['avg_cv_sw_w']:.4f})")
    return best_name, best_kernel, df_res


# =============================================================================
# STEP 4 — FIT BEST KERNEL AND PREDICT
# =============================================================================

def fit_and_predict(train_df, test_df, scaler,
                    kernel, sat_name, n_restarts=10):
    """
    Fit the best kernel on ALL training data and predict on test.

    Parameters
    ----------
    train_df   : full training DataFrame (featured, unscaled)
    test_df    : test DataFrame (featured, unscaled)
    scaler     : fitted StandardScaler
    kernel     : selected best kernel from select_best_kernel()
    sat_name   : for display
    n_restarts : final fit uses more restarts than CV

    Returns
    -------
    (pred_df, models_dict)
      pred_df: DataFrame with utc_time + predicted ERR_COLS + ERR_COLS_std
      models_dict: {col: fitted_gp}
    """
    t_tr = sc(train_df["t_min"].values, scaler, "t_min").reshape(-1,1)
    t_te = sc(test_df["t_min"].values,  scaler, "t_min").reshape(-1,1)

    pred_dict  = {"utc_time": test_df["utc_time"].values}
    models     = {}

    print(f"\n  Fitting tuned GPs on full training set ({n_restarts} restarts)...")
    for col in ERR_COLS:
        t0    = time.time()
        y_tr  = sc(train_df[col].values, scaler, col)

        gp    = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts, alpha=0.0)
        gp.fit(t_tr, y_tr)

        yp_sc, ys_sc = gp.predict(t_te, return_std=True)
        yp_m  = inv(yp_sc, scaler, col)
        ys_m  = ys_sc * scaler.scale_[
            list(scaler.feature_names_in_).index(col)]

        pred_dict[col]          = yp_m
        pred_dict[f"{col}_std"] = ys_m
        models[col]             = gp

        res     = test_df[col].values - yp_m
        w, p    = stats.shapiro(res)
        elapsed = time.time() - t0
        lml     = gp.log_marginal_likelihood_value_

        print(f"    {col:<16} SW_W={w:.4f}  "
              f"LML={lml:>8.2f}  ({elapsed:.1f}s)")

    return pd.DataFrame(pred_dict), models


# =============================================================================
# STEP 4b — MEO2 SEGMENT-AWARE TRAINING
# =============================================================================

def get_meo2_segment_train(train_df):
    """
    For MEO2, use only the last 2 days of training (Sep 8-9).

    WHY segment-aware for MEO2:
      MEO2 training has four large gaps (each ~24 hours).
      The GP fitted on all 143 rows must extrapolate across those
      gaps. During a 26-hour gap the GP reverts to its prior mean,
      then reconnects to post-gap data at the wrong level — creating
      a systematic bias (residual mean ≠ 0).

      The last 2 days (Sep 8-9, 40 rows) have NO large gaps and are
      TEMPORALLY CLOSEST to the test day (Sep 10). Training only on
      this segment gives the GP a clean, gap-free view of the most
      relevant recent behavior.

    WHY not use only Sep 9 (even closer to test)?
      Sep 9 has only 34 rows spanning ~10 hours. That's barely one
      orbital period — not enough to learn the 12h periodic pattern.
      Sep 8+9 gives ~26 hours of dense data, covering two orbital
      periods for the 12h kernel to learn from.

    Parameters
    ----------
    train_df : full MEO2 training DataFrame

    Returns
    -------
    DataFrame with only the last 2 days
    """
    cutoff     = pd.Timestamp("2025-09-08")
    segment_df = train_df[train_df["utc_time"] >= cutoff].copy()
    return segment_df.reset_index(drop=True)


# =============================================================================
# STEP 5 — N_RESTARTS SENSITIVITY
# =============================================================================

def restarts_sensitivity(train_df, test_df, scaler, kernel, col):
    """
    Test how n_restarts affects the SW_W score for one column.

    WHY this matters:
      The GP kernel has multiple hyperparameters (length scale,
      amplitude, period, noise level). The log-marginal-likelihood
      surface has many local optima — the optimizer can get stuck.
      More restarts = more starting points = better chance of
      finding the global optimum.

      But each restart takes ~1-2 seconds, so there is a trade-off.
      This function finds the minimum restarts needed for stable results.

    Returns
    -------
    dict: {n_restarts: (sw_w, lml)}
    """
    t_tr  = sc(train_df["t_min"].values, scaler, "t_min").reshape(-1,1)
    t_te  = sc(test_df["t_min"].values,  scaler, "t_min").reshape(-1,1)
    y_tr  = sc(train_df[col].values, scaler, col)

    results = {}
    for n in [2, 5, 10, 15]:
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n, alpha=0.0)
        gp.fit(t_tr, y_tr)
        yp, _ = gp.predict(t_te, return_std=True)
        yp_m  = inv(yp, scaler, col)
        res   = test_df[col].values - yp_m
        w, _  = stats.shapiro(res)
        lml   = gp.log_marginal_likelihood_value_
        results[n] = (w, lml)
        print(f"    n_restarts={n:>2}: SW_W={w:.4f}  LML={lml:.2f}")

    return results


# =============================================================================
# VISUALISATION — tuned GP
# =============================================================================

def plot_tuned(train_df, test_df, pred_df, sat_name, phase4_pred_path=None):
    """
    Side-by-side: Phase 4 GP vs Phase 5 tuned GP predictions.
    Also shows actual test values for reference.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
    fig.suptitle(f"Phase 5 Tuned GP vs Actual — {sat_name}",
                 fontsize=13, fontweight="bold", y=1.01)

    # Load Phase 4 predictions if available
    ph4_df = None
    if phase4_pred_path and os.path.exists(phase4_pred_path):
        ph4_df = pd.read_csv(phase4_pred_path)
        ph4_df["utc_time"] = pd.to_datetime(ph4_df["utc_time"])

    for idx, col in enumerate(ERR_COLS):
        ax = axes[idx]

        # Training context
        ax.scatter(train_df["utc_time"], train_df[col],
                   color="#AAAAAA", s=6, alpha=0.4, label="Train", zorder=2)

        # Phase 4 prediction (if available)
        if ph4_df is not None and col in ph4_df.columns:
            ax.plot(test_df["utc_time"], ph4_df[col].values,
                    color="#E5C07B", linewidth=1.5,
                    linestyle="--", label="Phase 4 GP", zorder=3, alpha=0.8)

        # Phase 5 tuned prediction + confidence band
        ax.plot(test_df["utc_time"], pred_df[col],
                color="#61AFEF", linewidth=2,
                label="Phase 5 Tuned GP", zorder=4)

        if f"{col}_std" in pred_df.columns:
            std = pred_df[f"{col}_std"].values
            ax.fill_between(test_df["utc_time"],
                            pred_df[col] - 2*std,
                            pred_df[col] + 2*std,
                            color="#61AFEF", alpha=0.15,
                            label="±2σ", zorder=3)

        # Actual test values
        ax.scatter(test_df["utc_time"], test_df[col],
                   color="#E06C75", s=20, zorder=5,
                   label="Test actual", marker="o")

        ax.axvline(train_df["utc_time"].max(),
                   color="gray", linewidth=1, linestyle=":", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.4,
                   linestyle="--", alpha=0.4)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_ylabel("Error (m)", fontsize=8)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=25)
        if idx == 0:
            ax.legend(fontsize=7, ncol=5, loc="upper left")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"phase5_tuned_{sat_name.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: figures/phase5_tuned_{sat_name.lower()}.png")


def plot_residuals(test_df, pred_df, sat_name):
    """Histogram + Q-Q for tuned residuals."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle(
        f"Phase 5 Residual Analysis — {sat_name} (Tuned)",
        fontsize=12, fontweight="bold")

    for idx, col in enumerate(ERR_COLS):
        res  = test_df[col].values - pred_df[col].values
        w, p = stats.shapiro(res)

        # Histogram
        ax = axes[idx][0]
        ax.hist(res, bins=15, density=True,
                color="#98C379", alpha=0.7, edgecolor="white")
        xr = np.linspace(res.min(), res.max(), 200)
        ax.plot(xr, stats.norm.pdf(xr, res.mean(), res.std()),
                "k--", linewidth=1.5, label="Normal")
        status = "✓ Normal" if p >= 0.05 else "✗ Not normal"
        ax.set_title(f"{col}\nW={w:.4f}  {status}", fontsize=9,
                     fontweight="bold")
        ax.set_xlabel("Residual (m)", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

        # Q-Q
        ax = axes[idx][1]
        r  = (res - res.mean()) / res.std() if res.std() > 0 else res
        n  = len(r)
        q  = stats.norm.ppf((np.arange(1,n+1)-0.5)/n)
        ax.scatter(q, np.sort(r), color="#E06C75", s=20, zorder=3)
        lm = max(abs(q).max(), abs(r).max()) * 1.1
        ax.plot([-lm,lm],[-lm,lm],"k--",linewidth=1,label="Ideal")
        ax.set_title(f"{col} — Q-Q", fontsize=9, fontweight="bold")
        ax.set_xlabel("Theoretical", fontsize=7)
        ax.set_ylabel("Sample", fontsize=7)
        ax.set_xlim(-lm,lm); ax.set_ylim(-lm,lm)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = os.path.join(FIG_DIR,
                       f"phase5_residuals_{sat_name.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: figures/phase5_residuals_{sat_name.lower()}.png")


# =============================================================================
# COMPARISON — Phase 4 vs Phase 5
# =============================================================================

def compare_phases(tuned_scores):
    """Load Phase 4 scores and compare with Phase 5."""
    ph4_path = os.path.join(RES_DIR, "gp_scores.csv")
    if not os.path.exists(ph4_path):
        print("  ⚠  gp_scores.csv not found — run Phase 4 first")
        return

    ph4   = pd.read_csv(ph4_path)
    ph4   = ph4[ph4["column"] == "AVERAGED"]
    ph5   = tuned_scores[tuned_scores["column"] == "AVERAGED"]

    print("\n" + "═" * 65)
    print("  PHASE 4 (baseline GP) vs PHASE 5 (tuned GP) — SW_W")
    print("═" * 65)

    rows = []
    for sat in ["GEO","MEO1","MEO2"]:
        r4 = ph4[ph4["satellite"] == sat]
        r5 = ph5[ph5["satellite"] == sat]
        if r4.empty or r5.empty: continue

        w4   = r4["sw_w"].values[0]
        w5   = r5["sw_w"].values[0]
        diff = w5 - w4
        sym  = "↑ improved" if diff > 0 else "↓ regressed"
        vs_bm= w5 - SW_BENCHMARK_W
        bm   = "✓ meets" if vs_bm >= 0 else "✗ below"
        rows.append([sat, f"{w4:.4f}", f"{w5:.4f}",
                     f"{'+' if diff>=0 else ''}{diff:.4f}  {sym}",
                     f"{bm} ({'+' if vs_bm>=0 else ''}{vs_bm:.4f})"])

    print(tabulate(rows,
                   headers=["Sat","Phase4 W","Phase5 W",
                             "Change","vs Benchmark(0.9810)"],
                   tablefmt="rounded_outline"))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "═" * 65)
    print("  SIH GNSS Error Prediction Project")
    print("  PHASE 5 — Kernel Tuning and Cross-Validation")
    print("═" * 65)

    all_eval   = []
    all_models = {}
    all_cv_dfs = []

    # ════════════════════════════════════════════════════════════════
    # SATELLITE A — GEO
    # ════════════════════════════════════════════════════════════════
    sep = "─" * 60
    print(f"\n  {sep}")
    print(f"  SATELLITE A — GEO")
    print(f"  {sep}")

    train_geo = load("geo_train_featured.csv")
    test_geo  = load("geo_test_featured.csv")
    sc_geo    = joblib.load(os.path.join(RES_DIR,"scaler_geo.pkl"))
    t_std_geo = get_t_std(sc_geo)

    # Kernel comparison with TSCV
    geo_kerns = geo_kernels(t_std_geo)
    best_geo_name, best_geo_kern, cv_geo = select_best_kernel(
        train_geo, sc_geo, geo_kerns, "GEO",
        n_splits=3, n_restarts=3)
    cv_geo["satellite"] = "GEO"
    all_cv_dfs.append(cv_geo)

    # n_restarts sensitivity (x_error only, for speed)
    print(f"\n  n_restarts sensitivity — GEO x_error:")
    restarts_sensitivity(train_geo, test_geo, sc_geo,
                         best_geo_kern, "x_error")

    # Fit and predict with tuned kernel
    pred_geo, models_geo = fit_and_predict(
        train_geo, test_geo, sc_geo,
        best_geo_kern, "GEO", n_restarts=10)

    eval_geo = evaluate_all_cols(test_geo, pred_geo, "GEO", "GP_Tuned")
    print_eval_table(eval_geo, f"GEO — Tuned ({best_geo_name})")
    all_eval.append(eval_geo)
    all_models["GEO"] = models_geo

    pred_geo.to_csv(os.path.join(RES_DIR,"tuned_predictions_geo.csv"),
                    index=False)
    ph4_geo = os.path.join(RES_DIR,"gp_predictions_geo.csv")
    plot_tuned(train_geo, test_geo, pred_geo, "GEO", ph4_geo)
    plot_residuals(test_geo, pred_geo, "GEO")

    # ════════════════════════════════════════════════════════════════
    # SATELLITE B — MEO1
    # ════════════════════════════════════════════════════════════════
    print(f"\n  {sep}")
    print(f"  SATELLITE B — MEO1")
    print(f"  {sep}")

    train_meo1 = load("meo1_train_featured.csv")
    test_meo1  = load("meo1_test_featured.csv")
    sc_meo1    = joblib.load(os.path.join(RES_DIR,"scaler_meo1.pkl"))
    t_std_meo1 = get_t_std(sc_meo1)

    meo_kerns_1 = meo_kernels(t_std_meo1)
    best_m1_name, best_m1_kern, cv_m1 = select_best_kernel(
        train_meo1, sc_meo1, meo_kerns_1, "MEO1",
        n_splits=3, n_restarts=3)
    cv_m1["satellite"] = "MEO1"
    all_cv_dfs.append(cv_m1)

    pred_meo1, models_meo1 = fit_and_predict(
        train_meo1, test_meo1, sc_meo1,
        best_m1_kern, "MEO1", n_restarts=10)

    eval_meo1 = evaluate_all_cols(test_meo1, pred_meo1,
                                   "MEO1","GP_Tuned")
    print_eval_table(eval_meo1, f"MEO1 — Tuned ({best_m1_name})")
    all_eval.append(eval_meo1)
    all_models["MEO1"] = models_meo1

    pred_meo1.to_csv(
        os.path.join(RES_DIR,"tuned_predictions_meo1.csv"), index=False)
    ph4_m1 = os.path.join(RES_DIR,"gp_predictions_meo1.csv")
    plot_tuned(train_meo1, test_meo1, pred_meo1, "MEO1", ph4_m1)
    plot_residuals(test_meo1, pred_meo1, "MEO1")

    # ════════════════════════════════════════════════════════════════
    # SATELLITE C — MEO2  (segment-aware training)
    # ════════════════════════════════════════════════════════════════
    print(f"\n  {sep}")
    print(f"  SATELLITE C — MEO2  (segment-aware: last 2 days only)")
    print(f"  {sep}")

    train_meo2_full = load("meo2_train_featured.csv")
    test_meo2       = load("meo2_test_featured.csv")
    sc_meo2         = joblib.load(os.path.join(RES_DIR,"scaler_meo2.pkl"))
    t_std_meo2      = get_t_std(sc_meo2)

    # Use only last 2 days (Sep 8-9) — no large gaps, closest to test
    train_meo2_seg = get_meo2_segment_train(train_meo2_full)
    print(f"  Segment: {len(train_meo2_seg)} rows  "
          f"({train_meo2_seg['utc_time'].min().date()} → "
          f"{train_meo2_seg['utc_time'].max().date()})")

    meo_kerns_2 = meo_kernels(t_std_meo2)
    best_m2_name, best_m2_kern, cv_m2 = select_best_kernel(
        train_meo2_seg, sc_meo2, meo_kerns_2, "MEO2",
        n_splits=2, n_restarts=3)  # n_splits=2 (only 40 rows)
    cv_m2["satellite"] = "MEO2"
    all_cv_dfs.append(cv_m2)

    pred_meo2, models_meo2 = fit_and_predict(
        train_meo2_seg, test_meo2, sc_meo2,
        best_m2_kern, "MEO2", n_restarts=10)

    eval_meo2 = evaluate_all_cols(test_meo2, pred_meo2,
                                   "MEO2","GP_Tuned")
    print_eval_table(eval_meo2, f"MEO2 — Tuned Segment ({best_m2_name})")
    all_eval.append(eval_meo2)
    all_models["MEO2"] = models_meo2

    pred_meo2.to_csv(
        os.path.join(RES_DIR,"tuned_predictions_meo2.csv"), index=False)
    ph4_m2 = os.path.join(RES_DIR,"gp_predictions_meo2.csv")
    plot_tuned(train_meo2_seg, test_meo2, pred_meo2, "MEO2", ph4_m2)
    plot_residuals(test_meo2, pred_meo2, "MEO2")

    # ════════════════════════════════════════════════════════════════
    # SAVE EVERYTHING
    # ════════════════════════════════════════════════════════════════
    all_df = pd.concat(all_eval, ignore_index=True)
    all_df.to_csv(os.path.join(RES_DIR,"tuned_scores.csv"), index=False)

    joblib.dump(all_models,
                os.path.join(RES_DIR,"tuned_models.pkl"))

    cv_all = pd.concat(all_cv_dfs, ignore_index=True)
    cv_all.to_csv(os.path.join(RES_DIR,"kernel_comparison.csv"),
                  index=False)

    print(f"\n  ✓  tuned_scores.csv saved")
    print(f"  ✓  tuned_models.pkl saved  (12 GP models)")
    print(f"  ✓  kernel_comparison.csv saved")

    # ════════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ════════════════════════════════════════════════════════════════
    compare_phases(all_df)

    # Grand average
    avg_rows = all_df[all_df["column"]=="AVERAGED"]
    gw = avg_rows["sw_w"].mean()
    gp = avg_rows["sw_p"].mean()
    gr = avg_rows["h0_rejected"].mean()

    print(f"""
  ═══════════════════════════════════════════════════════════════
  GRAND AVERAGE — Phase 5 Tuned GP:
    SW_W = {gw:.4f}   (benchmark: {SW_BENCHMARK_W})
    SW_p = {gp:.4f}   (benchmark: {SW_BENCHMARK_P})
    H0 rejection rate = {gr:.2f}   (target: 0.00)
  ═══════════════════════════════════════════════════════════════""")

    print("\n" + "═" * 65)
    print("  PHASE 5 COMPLETE")
    print("═" * 65)
    print("""
  Files saved:
    results/tuned_predictions_geo.csv
    results/tuned_predictions_meo1.csv
    results/tuned_predictions_meo2.csv
    results/tuned_scores.csv
    results/tuned_models.pkl
    results/kernel_comparison.csv
    figures/phase5_tuned_*.png
    figures/phase5_residuals_*.png

  Next step → run:  python src/phase6_improvements.py
    """)


if __name__ == "__main__":
    main()