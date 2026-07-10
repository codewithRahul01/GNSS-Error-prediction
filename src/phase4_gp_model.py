"""
=============================================================================
  SIH GNSS ERROR PREDICTION PROJECT
  PHASE 4 — Gaussian Process Regression Model
=============================================================================

  WHAT THIS PHASE DOES
  ─────────────────────
  Trains a Gaussian Process (GP) model for each satellite and each
  error column, predicts test-day errors, and evaluates using the
  Shapiro-Wilk criterion from the problem statement.

  One GP is trained per (satellite, error_column) pair:
    GEO  × 4 columns = 4 GPs
    MEO1 × 4 columns = 4 GPs
    MEO2 × 4 columns = 4 GPs
    Total = 12 GP models

  WHY GAUSSIAN PROCESS (not LSTM):
    • Dataset is tiny: 46–121 training rows per satellite
      LSTM needs thousands of windows — impossible here
    • Sampling is non-uniform: gaps from 1 min to 1556 min
      LSTM expects fixed time steps — GP takes any continuous time
    • Evaluation is residual normality (SW test), not RMSE
      GP explicitly models the posterior distribution → clean residuals
    • Test timestamps are arbitrary (any time the evaluator chooses)
      GP predicts at ANY query point without retraining

  HOW A GP WORKS (simple explanation):
    You give the GP training pairs: (time, error_value)
    The GP learns: "at times close to each other, errors are similar"
    The KERNEL defines what "similar" means.
    At any new test time, the GP computes a weighted average of nearby
    training values — the weights come from the kernel function.
    It also gives you a confidence interval (posterior std).

  KERNEL DESIGN (why these specific kernels):

    GEO kernel:
      ConstantKernel × RBF
        → smooth slow drift between upload cycles
      ConstantKernel × ExpSineSquared(period≈1440min)
        → the 24-hour repeating daily pattern
      WhiteKernel
        → upload-boundary spikes (unpredictable noise)

    MEO kernel:
      ConstantKernel × Matern(nu=1.5)
        → slightly rough trend (more realistic than pure RBF)
      ConstantKernel × ExpSineSquared(period≈720min)
        → the 12-hour half-day orbital period
      ConstantKernel × ExpSineSquared(period≈1440min)
        → the 24-hour daily modulation
      WhiteKernel
        → measurement noise

  HOW TO RUN
  ──────────
    python src/phase4_gp_model.py

  INPUT  (reads from)
  ─────
    Data/Processed/geo_train_featured.csv    (unscaled, has t_min)
    Data/Processed/geo_test_featured.csv
    Data/Processed/meo1_train_featured.csv
    Data/Processed/meo1_test_featured.csv
    Data/Processed/meo2_train_featured.csv
    Data/Processed/meo2_test_featured.csv
    results/scaler_geo.pkl                   (fitted scalers from Phase 2)
    results/scaler_meo1.pkl
    results/scaler_meo2.pkl

  OUTPUT (saves to)
  ──────
    results/gp_predictions_geo.csv
    results/gp_predictions_meo1.csv
    results/gp_predictions_meo2.csv
    results/gp_scores.csv               ← SW scores vs baselines
    results/gp_models.pkl               ← all 12 fitted GP objects
    figures/phase4_gp_geo.png
    figures/phase4_gp_meo1.png
    figures/phase4_gp_meo2.png
    figures/phase4_residuals_geo.png
    figures/phase4_residuals_meo1.png
    figures/phase4_residuals_meo2.png

  PRE-REQUISITES
  ──────────────
    Phase 1, 2, 3 must be complete
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
)

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
        if os.path.isdir(_c):
            PROC_DIR = _c; break

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
N_RESTARTS     = 8    # GP hyperparameter restarts (higher = more thorough)

# =============================================================================
# HELPERS — scaling in/out
# =============================================================================

def scale_col(values: np.ndarray,
              scaler,
              col: str) -> np.ndarray:
    """
    Scale a single column using the fitted scaler from Phase 2.

    WHY scale the INPUT (t_min):
      The GP kernel length scale is measured in the same units as
      the input. If t_min is in raw minutes (0–9000), the kernel
      needs a very large length scale to say "points 500 min apart
      are correlated". After scaling (std≈1), a length scale of 0.5
      means "points 0.5 std apart are correlated" — much easier for
      the optimizer to find.

    WHY scale the OUTPUT (error columns):
      The GP's constant kernel (amplitude) and noise kernel are also
      in output units. Scaling to std≈1 keeps all 4 error columns
      on the same amplitude scale so the same kernel bounds work for
      all of them.

    Parameters
    ----------
    values : raw numpy array
    scaler : fitted StandardScaler from Phase 2
    col    : column name (must be in scaler.feature_names_in_)

    Returns
    -------
    scaled numpy array (mean=0, std≈1)
    """
    fitted = list(scaler.feature_names_in_)
    idx    = fitted.index(col)
    return (values - scaler.mean_[idx]) / scaler.scale_[idx]


def inverse_col(values: np.ndarray,
                scaler,
                col: str) -> np.ndarray:
    """
    Reverse the scaling — convert scaled predictions back to meters.

    WHY reverse:
      The GP predicts in scaled units. The SW test must run on
      residuals in METERS (same units as the true test values).
      If we compute residuals in scaled space, the SW score would
      be computed on wrong units and not comparable to the benchmark.

    Parameters
    ----------
    values : scaled numpy array (GP prediction output)
    scaler : same scaler used in scale_col()
    col    : column name

    Returns
    -------
    numpy array in original meter units
    """
    fitted = list(scaler.feature_names_in_)
    idx    = fitted.index(col)
    return values * scaler.scale_[idx] + scaler.mean_[idx]

# =============================================================================
# DATA LOADING
# =============================================================================

def load(filename: str) -> pd.DataFrame:
    """Load a featured (unscaled) CSV from Data/Processed/."""
    path = os.path.join(PROC_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  ✗  {path}"
            f"\n     Run phase2_preprocessing.py first."
        )
    df = pd.read_csv(path)
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    return df.sort_values("utc_time").reset_index(drop=True)

# =============================================================================
# KERNEL BUILDERS
# =============================================================================

def build_geo_kernel(t_std: float) -> object:
    """
    Build the GP kernel for the GEO satellite.

    Design rationale:
      GEO satellite is geostationary — it stays above the same point
      on Earth. Its errors come from:
        1. Slow drift between uploads      → RBF (smooth long-range)
        2. Daily repeating upload pattern  → ExpSineSquared(24h)
        3. Unpredictable upload spikes     → WhiteKernel (noise)

    Periods in SCALED t_min units:
      Raw 1440 min / t_std ≈ 0.547 in scaled space

    Parameters
    ----------
    t_std : standard deviation of t_min from the fitted scaler
            Used to convert physical periods (minutes) to scaled units.

    Returns
    -------
    sklearn kernel object
    """
    p_daily = 1440.0 / t_std   # 24-hour period in scaled units

    kernel = (
        C(1.0, (0.01, 100))
        * RBF(length_scale=0.5,
              length_scale_bounds=(0.01, 10.0))

        + C(0.5, (0.01, 50))
        * ExpSineSquared(
            length_scale=0.3,
            periodicity=p_daily,
            length_scale_bounds=(0.01, 5.0),
            periodicity_bounds=(p_daily * 0.5, p_daily * 2.0))

        + WhiteKernel(noise_level=1.0,
                      noise_level_bounds=(0.01, 100.0))
    )
    return kernel


def build_meo_kernel(t_std: float) -> object:
    """
    Build the GP kernel for MEO satellites (MEO1 and MEO2).

    Design rationale:
      MEO satellites orbit the Earth roughly every 12-13 hours.
      Their errors come from:
        1. Smooth orbital trend              → Matern(nu=1.5)
        2. 12-hour half-day orbital cycle    → ExpSineSquared(12h)
        3. 24-hour daily modulation          → ExpSineSquared(24h)
        4. Measurement noise                 → WhiteKernel

    WHY Matern instead of RBF for trend:
      RBF produces infinitely smooth functions.
      Matern(nu=1.5) allows one discontinuity in the derivative —
      MEO orbit errors are not perfectly smooth due to the large
      data gaps (26-hour gap in MEO2). Matern handles this better.

    WHY two periodic components:
      MEO1/MEO2 errors show BOTH a 12h pattern (orbital period) AND
      a 24h modulation (solar radiation pressure repeats daily).
      One periodic kernel can only capture one period. We need both.

    Parameters
    ----------
    t_std : standard deviation of t_min from the fitted scaler

    Returns
    -------
    sklearn kernel object
    """
    p_daily = 1440.0 / t_std   # 24h in scaled units
    p_halfd = 720.0  / t_std   # 12h in scaled units

    kernel = (
        C(1.0, (0.01, 100))
        * Matern(length_scale=0.5,
                 length_scale_bounds=(0.01, 10.0),
                 nu=1.5)

        + C(0.3, (0.01, 20))
        * ExpSineSquared(
            length_scale=0.3,
            periodicity=p_halfd,
            length_scale_bounds=(0.01, 5.0),
            periodicity_bounds=(p_halfd * 0.5, p_halfd * 2.0))

        + C(0.2, (0.01, 10))
        * ExpSineSquared(
            length_scale=0.3,
            periodicity=p_daily,
            length_scale_bounds=(0.01, 5.0),
            periodicity_bounds=(p_daily * 0.5, p_daily * 2.0))

        + WhiteKernel(noise_level=0.5,
                      noise_level_bounds=(0.01, 50.0))
    )
    return kernel

# =============================================================================
# GP FIT AND PREDICT — one column at a time
# =============================================================================

def fit_gp(t_train: np.ndarray,
           y_train: np.ndarray,
           kernel,
           n_restarts: int = N_RESTARTS) -> GaussianProcessRegressor:
    """
    Fit a single Gaussian Process on (t_train, y_train).

    Both inputs are already in scaled units.

    HOW THE GP FITS:
      It maximises the log-marginal-likelihood (LML) with respect to
      the kernel hyperparameters. LML balances:
        - Data fit: how well does the kernel explain training values?
        - Model complexity: simpler kernels are penalised less
      n_restarts runs the optimizer from different random starting
      points to avoid local optima. The best result is kept.

    Parameters
    ----------
    t_train    : scaled time array  shape (n,)
    y_train    : scaled error array shape (n,)
    kernel     : sklearn kernel object (from build_geo/meo_kernel)
    n_restarts : number of optimizer restarts

    Returns
    -------
    fitted GaussianProcessRegressor
    """
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=0.0,          # noise is already in WhiteKernel
        normalize_y=False,  # we already scaled y manually
    )
    gp.fit(t_train.reshape(-1, 1), y_train)
    return gp


def predict_gp(gp: GaussianProcessRegressor,
               t_test: np.ndarray) -> tuple:
    """
    Predict at test timestamps using a fitted GP.

    Returns SCALED predictions and SCALED posterior std.
    The caller is responsible for inverse-scaling.

    HOW GP PREDICTION WORKS:
      For each test point t*, the GP computes:
        - Posterior mean:  weighted average of training observations,
          where weights = kernel(t*, t_train) / kernel(t_train, t_train)
          Points close in time (small kernel distance) get higher weight
        - Posterior std:   uncertainty. Small near training data,
          large when extrapolating far from training data.

    Parameters
    ----------
    gp     : fitted GaussianProcessRegressor
    t_test : scaled time array for test points  shape (n,)

    Returns
    -------
    (y_pred_scaled, y_std_scaled)
    """
    y_pred, y_std = gp.predict(t_test.reshape(-1, 1), return_std=True)
    return y_pred, y_std

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(actual_df: pd.DataFrame,
             pred_df:   pd.DataFrame,
             sat_name:  str,
             label:     str = "GP") -> pd.DataFrame:
    """
    Compute all evaluation metrics on residuals (actual - predicted).

    Residuals MUST be in original meter units (not scaled).

    Parameters
    ----------
    actual_df : test dataframe with true ERR_COLS (in meters)
    pred_df   : predictions with ERR_COLS (in meters)
    sat_name  : 'GEO', 'MEO1', 'MEO2'
    label     : model name for display

    Returns
    -------
    pd.DataFrame with one row per ERR_COL + AVERAGED row
    """
    rows = []
    for col in ERR_COLS:
        actual   = actual_df[col].values
        pred     = pred_df[col].values
        residual = actual - pred

        rmse     = np.sqrt(np.mean(residual**2))
        mae      = np.mean(np.abs(residual))
        res_mean = residual.mean()
        res_std  = residual.std()
        w, p     = stats.shapiro(residual)
        rejected = int(p < 0.05)

        rows.append({
            "satellite"  : sat_name,
            "model"      : label,
            "column"     : col,
            "rmse"       : rmse,
            "mae"        : mae,
            "res_mean"   : res_mean,
            "res_std"    : res_std,
            "sw_w"       : w,
            "sw_p"       : p,
            "h0_rejected": rejected,
        })

    df_res = pd.DataFrame(rows)
    avg = {
        "satellite"  : sat_name,
        "model"      : label,
        "column"     : "AVERAGED",
        "rmse"       : df_res["rmse"].mean(),
        "mae"        : df_res["mae"].mean(),
        "res_mean"   : df_res["res_mean"].mean(),
        "res_std"    : df_res["res_std"].mean(),
        "sw_w"       : df_res["sw_w"].mean(),
        "sw_p"       : df_res["sw_p"].mean(),
        "h0_rejected": df_res["h0_rejected"].mean(),
    }
    df_res = pd.concat([df_res, pd.DataFrame([avg])], ignore_index=True)
    return df_res


def print_eval(df_res: pd.DataFrame, label: str):
    """Pretty-print evaluation results."""
    print(f"\n  ── {label} ──")
    rows = []
    for _, row in df_res.iterrows():
        sym = "✓" if row["h0_rejected"] == 0 else "✗"
        vs = ""
        if row["column"] == "AVERAGED":
            diff = row["sw_w"] - SW_BENCHMARK_W
            sym2 = "↑" if diff >= 0 else "↓"
            vs   = f"  ({sym2}{abs(diff):.4f} vs benchmark)"
        rows.append([
            f"{sym} {row['column']}",
            f"{row['rmse']:.4f}",
            f"{row['mae']:.4f}",
            f"{row['res_mean']:+.4f}",
            f"{row['res_std']:.4f}",
            f"{row['sw_w']:.4f}{vs}",
            f"{row['sw_p']:.4f}",
        ])
    hdrs = ["Column","RMSE(m)","MAE(m)","Res mean","Res std","SW_W","SW_p"]
    print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))

# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_gp_predictions(train_df:  pd.DataFrame,
                        test_df:   pd.DataFrame,
                        pred_df:   pd.DataFrame,
                        sat_name:  str):
    """
    Plot GP predictions vs actual values with confidence interval.

    Shows:
      • Grey dots   = training data (context the GP learned from)
      • Black dots  = actual test values (ground truth)
      • Blue line   = GP posterior mean (our prediction)
      • Blue shade  = ±2σ confidence interval (95% uncertainty band)

    HOW TO READ:
      If most black dots fall inside the shaded band → GP is
      well-calibrated (uncertainty estimates are accurate).
      If black dots are mostly outside → GP is overconfident
      (WhiteKernel noise_level is too low).
    """
    fig, axes = plt.subplots(4, 1, figsize=(14,14), sharex=False)
    fig.suptitle(f"Phase 4 GP Predictions vs Actual — {sat_name}",
                 fontsize=13, fontweight="bold", y=1.01)

    for idx, col in enumerate(ERR_COLS):
        ax = axes[idx]

        # Training context
        ax.scatter(train_df["utc_time"], train_df[col],
                   color="#AAAAAA", s=8, alpha=0.5,
                   label="Train (actual)", zorder=2)

        # GP prediction + confidence band
        ax.plot(test_df["utc_time"], pred_df[col],
                color="#61AFEF", linewidth=2,
                label="GP prediction", zorder=4)

        if f"{col}_std" in pred_df.columns:
            std = pred_df[f"{col}_std"].values
            ax.fill_between(
                test_df["utc_time"],
                pred_df[col] - 2*std,
                pred_df[col] + 2*std,
                color="#61AFEF", alpha=0.2,
                label="±2σ interval", zorder=3)

        # Actual test values
        ax.scatter(test_df["utc_time"], test_df[col],
                   color="#E06C75", s=20, zorder=5,
                   label="Test (actual)", marker="o")

        # Train/test split line
        ax.axvline(train_df["utc_time"].max(),
                   color="gray", linewidth=1,
                   linestyle=":", alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.4,
                   linestyle="--", alpha=0.4)

        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_ylabel("Error (m)", fontsize=8)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=25)
        if idx == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper left")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"phase4_gp_{sat_name.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: figures/phase4_gp_{sat_name.lower()}.png")


def plot_residuals(test_df:  pd.DataFrame,
                   pred_df:  pd.DataFrame,
                   sat_name: str):
    """
    Plot residual histogram + normal curve + Q-Q plot.

    TWO plots per error column:
      Left:  Histogram of residuals with fitted normal overlay
             → if histogram matches the curve: SW score will be high
      Right: Q-Q plot (quantile-quantile)
             → points on the diagonal = normal residuals
             → S-curve = heavy tails
             → points cluster off-line = outliers
    """
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle(
        f"Phase 4 Residual Analysis — {sat_name}\n"
        f"(Residuals close to normal = high SW_W score)",
        fontsize=12, fontweight="bold")

    for idx, col in enumerate(ERR_COLS):
        res = test_df[col].values - pred_df[col].values
        w, p = stats.shapiro(res)

        # ── Left: Histogram ───────────────────────────────────────
        ax = axes[idx][0]
        ax.hist(res, bins=15, density=True,
                color="#98C379", alpha=0.7, edgecolor="white")

        x_range = np.linspace(res.min(), res.max(), 200)
        normal  = stats.norm.pdf(x_range, res.mean(), res.std())
        ax.plot(x_range, normal, "k--", linewidth=1.5,
                label="Normal dist")

        status = "✓ Normal" if p >= 0.05 else "✗ Not normal"
        ax.set_title(f"{col} — Residuals\nW={w:.4f}  {status}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Residual (m)", fontsize=7)
        ax.set_ylabel("Density",     fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

        # ── Right: Q-Q plot ────────────────────────────────────────
        ax = axes[idx][1]
        res_std = (res - res.mean()) / res.std() if res.std() > 0 else res
        n       = len(res_std)
        probs   = (np.arange(1, n+1) - 0.5) / n
        theo_q  = stats.norm.ppf(probs)
        emp_q   = np.sort(res_std)

        ax.scatter(theo_q, emp_q, color="#E06C75", s=20, zorder=3)
        lim = max(abs(theo_q).max(), abs(emp_q).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--",
                linewidth=1, label="Perfect normal")

        ax.set_title(f"{col} — Q-Q Plot", fontsize=9, fontweight="bold")
        ax.set_xlabel("Theoretical quantiles", fontsize=7)
        ax.set_ylabel("Sample quantiles",      fontsize=7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"phase4_residuals_{sat_name.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: figures/phase4_residuals_{sat_name.lower()}.png")

# =============================================================================
# FULL PIPELINE — one satellite
# =============================================================================

def run_gp_for_satellite(train_file:  str,
                          test_file:   str,
                          sat_name:    str,
                          sat_type:    str,
                          scaler_name: str) -> dict:
    """
    Full GP pipeline for one satellite:
      1. Load data + scaler
      2. Build kernel (GEO or MEO)
      3. Fit one GP per error column
      4. Predict at test timestamps
      5. Inverse-scale predictions to meters
      6. Evaluate with SW test
      7. Save predictions + plots

    Parameters
    ----------
    train_file  : e.g. "geo_train_featured.csv"
    test_file   : e.g. "geo_test_featured.csv"
    sat_name    : 'GEO', 'MEO1', 'MEO2'
    sat_type    : 'GEO' or 'MEO'
    scaler_name : 'geo', 'meo1', 'meo2'

    Returns
    -------
    dict with keys: eval_df, pred_df, models
    """
    sep = "─" * 60
    print(f"\n  {sep}")
    print(f"  Satellite: {sat_name}  (type: {sat_type})")
    print(f"  {sep}")

    # ── Load data and scaler ──────────────────────────────────────
    train_df = load(train_file)
    test_df  = load(test_file)
    scaler   = joblib.load(
        os.path.join(RES_DIR, f"scaler_{scaler_name}.pkl"))

    print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

    # ── Get t_min std for kernel period calculation ───────────────
    fitted   = list(scaler.feature_names_in_)
    t_idx    = fitted.index("t_min")
    t_std    = scaler.scale_[t_idx]
    print(f"  t_min std (scaler): {t_std:.2f} min")
    if sat_type == "GEO":
        print(f"  Scaled 24h period : {1440/t_std:.4f}")
    else:
        print(f"  Scaled 12h period : {720/t_std:.4f}")
        print(f"  Scaled 24h period : {1440/t_std:.4f}")

    # ── Scale inputs ─────────────────────────────────────────────
    t_train_sc = scale_col(train_df["t_min"].values, scaler, "t_min")
    t_test_sc  = scale_col(test_df["t_min"].values,  scaler, "t_min")

    # ── Build kernel ─────────────────────────────────────────────
    print(f"\n  Kernel: {'GEO (RBF + Periodic_24h + White)' if sat_type=='GEO' else 'MEO (Matern + Periodic_12h + Periodic_24h + White)'}")

    # ── Fit one GP per error column ───────────────────────────────
    models     = {}
    pred_dict  = {"utc_time": test_df["utc_time"].values}

    print(f"\n  Fitting {len(ERR_COLS)} GP models ...")
    for col in ERR_COLS:
        t0 = time.time()

        # Scale the target column
        y_train_sc = scale_col(train_df[col].values, scaler, col)

        # Fresh kernel for each column (each has its own hyperparams)
        if sat_type == "GEO":
            kernel = build_geo_kernel(t_std)
        else:
            kernel = build_meo_kernel(t_std)

        # Fit GP
        gp = fit_gp(t_train_sc, y_train_sc,
                    kernel, N_RESTARTS)

        # Predict (scaled)
        y_pred_sc, y_std_sc = predict_gp(gp, t_test_sc)

        # Inverse-scale to meters
        y_pred_m = inverse_col(y_pred_sc, scaler, col)
        y_std_m  = y_std_sc * scaler.scale_[fitted.index(col)]

        pred_dict[col]           = y_pred_m
        pred_dict[f"{col}_std"]  = y_std_m
        models[col]              = gp

        elapsed = time.time() - t0
        lml     = gp.log_marginal_likelihood_value_
        w, p    = stats.shapiro(test_df[col].values - y_pred_m)

        print(f"    {col:<16} SW_W={w:.4f}  LML={lml:>8.2f}  "
              f"({elapsed:.1f}s)")

    pred_df = pd.DataFrame(pred_dict)

    # ── Evaluate ─────────────────────────────────────────────────
    eval_df = evaluate(test_df, pred_df, sat_name, "GP")
    print_eval(eval_df, f"GP — {sat_name}")

    # ── Save predictions ─────────────────────────────────────────
    out = os.path.join(RES_DIR,
                       f"gp_predictions_{sat_name.lower()}.csv")
    pred_df.to_csv(out, index=False)
    print(f"\n  ✓  Predictions saved: results/gp_predictions_{sat_name.lower()}.csv")

    # ── Plots ─────────────────────────────────────────────────────
    print()
    plot_gp_predictions(train_df, test_df, pred_df, sat_name)
    plot_residuals(test_df, pred_df, sat_name)

    return {
        "eval_df": eval_df,
        "pred_df": pred_df,
        "models" : models,
    }

# =============================================================================
# COMPARISON TABLE — GP vs Baselines
# =============================================================================

def compare_with_baselines(gp_scores: pd.DataFrame):
    """
    Load Phase 3 baseline scores and compare with GP SW scores.

    Answers: by how much did the GP beat the best baseline?
    """
    baseline_path = os.path.join(RES_DIR, "baseline_scores.csv")
    if not os.path.exists(baseline_path):
        print("  ⚠  baseline_scores.csv not found — run Phase 3 first")
        return

    bl = pd.read_csv(baseline_path)
    bl_avg = bl[bl["column"] == "AVERAGED"]

    # Best baseline per satellite
    best_bl = bl_avg.loc[bl_avg.groupby("satellite")["sw_w"].idxmax()]

    # GP averaged per satellite
    gp_avg  = gp_scores[gp_scores["column"] == "AVERAGED"]

    print("\n" + "═" * 65)
    print("  GP vs BEST BASELINE — SW_W comparison")
    print("  (higher W = more Gaussian residuals = better)")
    print("═" * 65)

    rows = []
    for sat in ["GEO","MEO1","MEO2"]:
        bl_row = best_bl[best_bl["satellite"] == sat]
        gp_row = gp_avg[gp_avg["satellite"] == sat]

        if bl_row.empty or gp_row.empty:
            continue

        bl_w   = bl_row["sw_w"].values[0]
        bl_n   = bl_row["baseline"].values[0]
        gp_w   = gp_row["sw_w"].values[0]
        diff   = gp_w - bl_w
        beats  = "✓ GP wins" if diff > 0 else "✗ GP loses"
        vs_bm  = gp_w - SW_BENCHMARK_W
        bm_sym = "✓ meets" if vs_bm >= 0 else "✗ below"

        rows.append([
            sat, bl_n,
            f"{bl_w:.4f}",
            f"{gp_w:.4f}",
            f"{'+' if diff>=0 else ''}{diff:.4f}",
            beats,
            f"{bm_sym} ({'+' if vs_bm>=0 else ''}{vs_bm:.4f})",
        ])

    hdrs = ["Satellite","Best Baseline","Baseline W",
            "GP W","GP-Baseline","Result","vs Benchmark(0.9810)"]
    print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "═" * 65)
    print("  SIH GNSS Error Prediction Project")
    print("  PHASE 4 — Gaussian Process Regression Model")
    print("═" * 65)
    print(f"""
  Model: Gaussian Process Regression
  Input: t_min (scaled continuous time in minutes)
  Output: x_error, y_error, z_error, clock_error (each predicted
          by its own GP, inverse-scaled to meters)
  Kernel:
    GEO  → RBF + ExpSineSquared(24h) + White
    MEO  → Matern(1.5) + ExpSineSquared(12h) + ExpSineSquared(24h) + White
  Evaluation: Shapiro-Wilk W on residuals (benchmark: {SW_BENCHMARK_W})
    """)

    all_eval  = []
    all_models= {}

    # ── Satellite A: GEO ─────────────────────────────────────────────
    result_geo = run_gp_for_satellite(
        train_file  = "geo_train_featured.csv",
        test_file   = "geo_test_featured.csv",
        sat_name    = "GEO",
        sat_type    = "GEO",
        scaler_name = "geo",
    )
    all_eval.append(result_geo["eval_df"])
    all_models["GEO"] = result_geo["models"]

    # ── Satellite B: MEO1 ────────────────────────────────────────────
    result_meo1 = run_gp_for_satellite(
        train_file  = "meo1_train_featured.csv",
        test_file   = "meo1_test_featured.csv",
        sat_name    = "MEO1",
        sat_type    = "MEO",
        scaler_name = "meo1",
    )
    all_eval.append(result_meo1["eval_df"])
    all_models["MEO1"] = result_meo1["models"]

    # ── Satellite C: MEO2 ────────────────────────────────────────────
    result_meo2 = run_gp_for_satellite(
        train_file  = "meo2_train_featured.csv",
        test_file   = "meo2_test_featured.csv",
        sat_name    = "MEO2",
        sat_type    = "MEO",
        scaler_name = "meo2",
    )
    all_eval.append(result_meo2["eval_df"])
    all_models["MEO2"] = result_meo2["models"]

    # ── Combined GP scores ────────────────────────────────────────────
    all_df = pd.concat(all_eval, ignore_index=True)
    all_df.to_csv(os.path.join(RES_DIR, "gp_scores.csv"), index=False)

    # ── Save all 12 GP models ─────────────────────────────────────────
    joblib.dump(all_models,
                os.path.join(RES_DIR, "gp_models.pkl"))
    print("\n  ✓  All 12 GP models saved: results/gp_models.pkl")

    # ── GP vs Baseline comparison ─────────────────────────────────────
    compare_with_baselines(all_df)

    # ── Grand average across all satellites ──────────────────────────
    avg_rows = all_df[all_df["column"] == "AVERAGED"]
    grand_w  = avg_rows["sw_w"].mean()
    grand_p  = avg_rows["sw_p"].mean()
    grand_rej= avg_rows["h0_rejected"].mean()

    print(f"""
  ═══════════════════════════════════════════════════════════════
  GRAND AVERAGE across all 3 satellites + 4 columns:
    SW_W = {grand_w:.4f}   (benchmark: {SW_BENCHMARK_W})
    SW_p = {grand_p:.4f}   (benchmark: {SW_BENCHMARK_P})
    H0 rejection rate = {grand_rej:.2f}  (target: 0.00)
  ═══════════════════════════════════════════════════════════════""")

    # ── Done ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  PHASE 4 COMPLETE")
    print("═" * 65)
    print("""
  Files saved:
    results/gp_predictions_geo.csv
    results/gp_predictions_meo1.csv
    results/gp_predictions_meo2.csv
    results/gp_scores.csv
    results/gp_models.pkl
    figures/phase4_gp_geo.png
    figures/phase4_gp_meo1.png
    figures/phase4_gp_meo2.png
    figures/phase4_residuals_geo.png
    figures/phase4_residuals_meo1.png
    figures/phase4_residuals_meo2.png

  Next step → run:  python src/phase5_tuning.py
    """)


if __name__ == "__main__":
    main()