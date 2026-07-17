"""
=============================================================================
  SIH GNSS ERROR PREDICTION PROJECT
  PHASE 7 — Final Submission Prediction Script
=============================================================================

  PURPOSE
  ───────
  This is the FINAL SUBMISSION SCRIPT for the SIH evaluation.

  At evaluation time the committee will:
    1. Give you a new training CSV file (7 days, same format as your data)
    2. Give you a list of test timestamps (arbitrary, day 8)
    3. Ask you to predict x_error, y_error, z_error, clock_error
       at each of those timestamps

  This script handles everything with ONE command:

    python src/phase7_predict.py \
        --train  path/to/training_data.csv \
        --times  path/to/test_timestamps.txt \
        --output path/to/predictions.csv \
        --type   GEO

  Or with auto-detection of satellite type:

    python src/phase7_predict.py \
        --train  path/to/training_data.csv \
        --times  path/to/test_timestamps.txt \
        --output path/to/predictions.csv

  SELF-CONTAINED
  ──────────────
  This script imports NOTHING from previous phase scripts.
  It contains the complete pipeline end-to-end:
    Step 1  → Load and clean training data
    Step 2  → Auto-detect satellite type (GEO or MEO)
    Step 3  → Apply GEO mode filter (keep 15-min upload mode days only)
    Step 4  → Winsorize outliers in training data
    Step 5  → Build t_min feature (continuous time in minutes)
    Step 6  → Fit StandardScaler on training data
    Step 7  → Build the correct kernel for the detected satellite type
    Step 8  → Fit one GP per error column
    Step 9  → Predict at all test timestamps
    Step 10 → Save predictions CSV

  TIMESTAMP FILE FORMAT
  ──────────────────────
  One timestamp per line, any of these formats:
    2025-09-08 00:11:00
    2025-09-08T00:11:00
    08/09/2025 00:11
  The script auto-parses all standard datetime formats.

  OUTPUT FORMAT
  ─────────────
  CSV with columns:
    utc_time, x_error, y_error, z_error, clock_error

  ALSO SAVES (optional, for your own validation):
    *_with_uncertainty.csv  — same predictions + ±2σ confidence columns

  RUNTIME
  ───────
  ~40–60 seconds on a MacBook Air M-series (12 GP fits × ~5s each).

  PRE-REQUISITES
  ──────────────
  pip install pandas numpy scipy scikit-learn tabulate
  (no dependency on any other phase script)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ExpSineSquared,
    WhiteKernel, ConstantKernel as C,
)
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# =============================================================================
# CONSTANTS
# =============================================================================

ERR_COLS       = ["x_error", "y_error", "z_error", "clock_error"]
OUTLIER_K      = 3.0      # k × IQR for winsorization
N_RESTARTS     = 8        # GP optimizer restarts (balance of speed vs accuracy)
SW_BENCHMARK_W = 0.9810   # target from problem statement

# =============================================================================
# STEP 1 — LOAD AND CLEAN TRAINING DATA
# =============================================================================

def load_and_clean(train_path: str) -> pd.DataFrame:
    """
    Load a raw GNSS error CSV and apply mandatory cleaning.

    Handles:
      - Any column name variations (standardizes to 5 expected names)
      - Duplicate timestamps (MEO Train had 101 duplicates)
      - Sorting by time

    Parameters
    ----------
    train_path : path to the training CSV file

    Returns
    -------
    cleaned pd.DataFrame with columns:
      utc_time, x_error, y_error, z_error, clock_error
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")

    df = pd.read_csv(train_path)

    # Standardize column names regardless of spacing or casing
    if df.shape[1] != 5:
        raise ValueError(
            f"Expected 5 columns, got {df.shape[1]}. "
            f"File must have: utc_time, x_error, y_error, z_error, clock_error"
        )
    df.columns = ["utc_time", "x_error", "y_error", "z_error", "clock_error"]

    # Parse timestamps
    df["utc_time"] = pd.to_datetime(df["utc_time"])

    # Remove duplicates (common data quality issue in this dataset)
    n_before = len(df)
    df = df.drop_duplicates(subset="utc_time").sort_values("utc_time")
    df = df.reset_index(drop=True)
    n_after = len(df)

    if n_before != n_after:
        print(f"  [clean] Removed {n_before - n_after} duplicate rows "
              f"→ {n_after} clean rows")

    return df


# =============================================================================
# STEP 2 — AUTO-DETECT SATELLITE TYPE
# =============================================================================

def detect_satellite_type(df: pd.DataFrame) -> str:
    """
    Automatically determine whether the training data is GEO or MEO.

    Detection logic (two independent signals — both checked):

    Signal 1 — Upload interval gaps:
      GEO satellites are uploaded every ~120 minutes.
      This creates many gaps of exactly 100-130 minutes between rows.
      MEO satellites have irregular larger gaps (no 120-min pattern).
      If ≥ 5 gaps fall in the 100-130 min range → GEO.

    Signal 2 — Error magnitude:
      GEO errors are 5-50× larger than MEO errors because:
        - GEO orbit determination from ground tracking is harder
          (fewer tracking stations can see a geostationary satellite
           vs a MEO satellite that passes over many stations)
        - GEO upload cycles are longer → more error accumulation
      If mean std across 4 error columns > 2.0m → GEO.

    Returns 'GEO' or 'MEO'.
    """
    gaps = df["utc_time"].diff().dropna().dt.total_seconds() / 60

    # Signal 1: count upload-cycle gaps
    n_upload_gaps = int(((gaps >= 100) & (gaps <= 130)).sum())

    # Signal 2: error magnitude
    avg_err_std = df[ERR_COLS].std().mean()

    # Either signal is sufficient for GEO classification
    if n_upload_gaps >= 5 or avg_err_std > 2.0:
        reason = (f"upload gaps={n_upload_gaps}" if n_upload_gaps >= 5
                  else f"err_std={avg_err_std:.2f}m")
        print(f"  [detect] Satellite type: GEO  ({reason})")
        return "GEO"
    else:
        print(f"  [detect] Satellite type: MEO  "
              f"(upload gaps={n_upload_gaps}, err_std={avg_err_std:.2f}m)")
        return "MEO"


# =============================================================================
# STEP 3 — GEO MODE FILTER
# =============================================================================

def apply_geo_mode_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    For GEO satellites: keep only days in 15-minute upload mode.

    WHY this is critical for GEO:
      GEO training data (Sep 1-7) has TWO upload modes:
        Days 1-5: 120-min upload intervals (2-hour ephemeris updates)
        Days 6-7: 15-min upload intervals  (matches the test day Sep 8)

      If you train on all 7 days, the model learns the 2-hour sawtooth
      for 5 days and the 15-min pattern for only 2 days.
      When the test day uses 15-min uploads, the model predicts the
      wrong sawtooth frequency.

      Solution: detect which days use 15-min mode (median gap ≤ 30 min)
      and train ONLY on those days.

    HOW detection works:
      For each calendar day, compute the median time gap between rows.
      Median ≤ 30 min → 15-min upload mode day → KEEP
      Median >  30 min → 120-min upload mode day → DROP

    Parameters
    ----------
    df : cleaned GEO training DataFrame

    Returns
    -------
    DataFrame containing only 15-min mode days
    """
    df         = df.copy()
    df["date"] = df["utc_time"].dt.date
    df["gap_min"] = df["utc_time"].diff().dt.total_seconds().fillna(0) / 60

    daily_med = df.groupby("date")["gap_min"].median()
    mode15_days = daily_med[daily_med <= 30].index

    if len(mode15_days) == 0:
        print("  [geo_filter] No 15-min mode days found — using all data")
        return df.drop(columns=["date","gap_min"])

    filtered = df[df["date"].isin(mode15_days)].copy()
    filtered  = filtered.drop(columns=["date","gap_min"]).reset_index(drop=True)

    print(f"  [geo_filter] Kept {len(filtered)} rows from "
          f"{len(mode15_days)} 15-min mode day(s): "
          f"{[str(d) for d in sorted(mode15_days)]}")
    return filtered


# =============================================================================
# STEP 4 — WINSORIZE OUTLIERS
# =============================================================================

def winsorize(df: pd.DataFrame, k: float = OUTLIER_K) -> pd.DataFrame:
    """
    Clip extreme training values to the k×IQR fence.

    Applied to training data ONLY — never to test timestamps.

    WHY:
      GEO upload-boundary spikes (±40m) and MEO anomaly events
      distort the GP kernel optimization if left in.
      The kernel tries to explain the spikes as signal and sets
      a large length scale — missing the smooth periodic trend.
      Winsorizing forces the kernel to learn the underlying pattern.

    WHY 3×IQR (not 1.5×):
      1.5×IQR clips too aggressively for satellite data.
      Legitimate large values at upload boundaries would be removed.
      3×IQR only removes the truly extreme values (~top 0.3%).
    """
    df = df.copy()
    for col in ERR_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - k * IQR
        hi  = Q3 + k * IQR
        n   = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n > 0:
            df[col] = df[col].clip(lo, hi)
            print(f"  [winsorize] {col}: clipped {n} values "
                  f"to [{lo:.3f}, {hi:.3f}]")
    return df


# =============================================================================
# STEP 5 — TIME FEATURE
# =============================================================================

def build_t_min(train_df: pd.DataFrame,
                test_timestamps: pd.DatetimeIndex) -> tuple:
    """
    Build continuous time feature t_min for train and test.

    t_min = minutes elapsed since the first training timestamp.

    WHY t_min (not raw timestamps):
      The GP computes distance between inputs to measure correlation.
      It cannot compute distance between datetime strings.
      t_min converts timestamps to plain floats the GP can use.

    WHY the SAME reference for train and test:
      If train uses its own start as t_ref and test uses test start,
      the GP sees test timestamps as if they were in the training range.
      This turns extrapolation into interpolation — completely wrong.
      Always use the training start as t_ref for both.

    Returns
    -------
    (t_train_raw, t_test_raw, t_ref)
      Both arrays are in raw minutes (unscaled).
    """
    t_ref      = train_df["utc_time"].min()
    t_train    = (train_df["utc_time"] - t_ref).dt.total_seconds().values / 60
    t_test     = (pd.DatetimeIndex(test_timestamps) - t_ref).total_seconds().values / 60
    return t_train, t_test, t_ref


# =============================================================================
# STEP 6 — SCALING
# =============================================================================

def fit_and_scale(t_train_raw: np.ndarray,
                  train_df: pd.DataFrame) -> tuple:
    """
    Fit StandardScaler on training t_min, scale t_min and error columns.

    WHY scale:
      GP kernel hyperparameters are defined in the same units as inputs
      and outputs. Raw t_min (0–9000 min) needs a large length scale.
      After scaling (std≈1), the length scale is in comparable units
      across all error columns and satellites.

    WHY fit on train only:
      The scaler must NOT see test data during fitting.
      Using test statistics would be data leakage.

    Returns
    -------
    (t_scaler, err_scalers, t_train_sc, y_train_scaled_dict)
    """
    # Scale t_min
    t_scaler    = StandardScaler()
    t_train_sc  = t_scaler.fit_transform(
        t_train_raw.reshape(-1,1)).ravel()

    # Scale each error column independently
    err_scalers = {}
    y_scaled    = {}
    for col in ERR_COLS:
        sc = StandardScaler()
        y_scaled[col] = sc.fit_transform(
            train_df[col].values.reshape(-1,1)).ravel()
        err_scalers[col] = sc

    return t_scaler, err_scalers, t_train_sc, y_scaled


# =============================================================================
# STEP 7 — KERNEL BUILDER
# =============================================================================

def build_kernel(sat_type: str, t_std: float):
    """
    Build the GP kernel appropriate for the satellite type.

    GEO kernel — RBF + Periodic(24h) + White:
      RBF captures the smooth drift within upload cycles.
      ExpSineSquared(24h) captures the daily upload reset pattern.
      WhiteKernel absorbs the unpredictable upload-boundary spikes.
      The 24h period dominates because GEO is geostationary and
      follows Earth's rotation — daily ephemeris patterns repeat.

    MEO kernel — RBF + Periodic(12h) + Periodic(24h) + White:
      RBF captures smooth orbital trend.
      ExpSineSquared(12h) captures MEO orbital period (~12-13h).
      ExpSineSquared(24h) captures daily solar radiation pressure.
      Two periodic components needed because MEO errors show BOTH
      the orbital period and the daily solar cycle simultaneously.

    All periods are converted to SCALED units:
      period_scaled = period_minutes / t_std
    This ensures the kernel optimizer works in a consistent space
    regardless of the actual t_min range.

    Parameters
    ----------
    sat_type : 'GEO' or 'MEO'
    t_std    : std of t_min from StandardScaler

    Returns
    -------
    sklearn GP kernel object
    """
    p12h = 720.0  / t_std   # 12-hour period in scaled units
    p1d  = 1440.0 / t_std   # 24-hour period in scaled units

    if sat_type == "GEO":
        return (
            C(1.0, (0.01, 100))
            * RBF(length_scale=0.5, length_scale_bounds=(0.01, 10.0))

            + C(0.5, (0.01, 50))
            * ExpSineSquared(
                length_scale=0.3, periodicity=p1d,
                length_scale_bounds=(0.01, 5.0),
                periodicity_bounds=(p1d * 0.5, p1d * 2.0))

            + WhiteKernel(noise_level=1.0,
                          noise_level_bounds=(0.01, 100.0))
        )
    else:  # MEO
        return (
            C(1.0, (0.01, 100))
            * RBF(length_scale=0.5, length_scale_bounds=(0.01, 10.0))

            + C(0.3, (0.01, 20))
            * ExpSineSquared(
                length_scale=0.3, periodicity=p12h,
                length_scale_bounds=(0.01, 5.0),
                periodicity_bounds=(p12h * 0.5, p12h * 2.0))

            + C(0.2, (0.01, 10))
            * ExpSineSquared(
                length_scale=0.3, periodicity=p1d,
                length_scale_bounds=(0.01, 5.0),
                periodicity_bounds=(p1d * 0.5, p1d * 2.0))

            + WhiteKernel(noise_level=0.5,
                          noise_level_bounds=(0.01, 50.0))
        )


# =============================================================================
# STEP 8+9 — FIT GP AND PREDICT
# =============================================================================

def fit_and_predict(t_train_sc:   np.ndarray,
                    y_scaled:     dict,
                    t_test_sc:    np.ndarray,
                    t_scaler:     StandardScaler,
                    err_scalers:  dict,
                    sat_type:     str,
                    n_restarts:   int = N_RESTARTS) -> pd.DataFrame:
    """
    Fit one GP per error column and predict at all test timestamps.

    HOW GP PREDICTION WORKS:
      Given training pairs (t_train, error_value), the GP learns
      the covariance structure of the error time series.
      At any new test time t*, it computes a weighted average of
      nearby training observations — points closer in time (according
      to the kernel) get higher weight.
      The kernel hyperparameters (length scale, amplitude, period,
      noise level) are optimized by maximizing the log-marginal-
      likelihood of the training data.

    WHY n_restarts=8:
      The log-marginal-likelihood surface has multiple local optima.
      8 random restarts gives good coverage without being too slow.
      More restarts = slightly better hyperparameters but longer runtime.

    Parameters
    ----------
    t_train_sc  : scaled training t_min  shape (n_train,)
    y_scaled    : {col: scaled_y}  for each error column
    t_test_sc   : scaled test t_min  shape (n_test,)
    t_scaler    : fitted StandardScaler for t_min (for t_std)
    err_scalers : fitted StandardScaler per error column
    sat_type    : 'GEO' or 'MEO'
    n_restarts  : GP optimizer restarts

    Returns
    -------
    pd.DataFrame with columns:
      x_error, y_error, z_error, clock_error
      x_error_std, y_error_std, z_error_std, clock_error_std
    """
    t_std = t_scaler.scale_[0]
    preds = {}

    for col in ERR_COLS:
        t0     = time.time()
        kernel = build_kernel(sat_type, t_std)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            alpha=0.0,        # noise handled by WhiteKernel
            normalize_y=False # we already scaled y manually
        )
        gp.fit(t_train_sc.reshape(-1,1), y_scaled[col])

        y_pred_sc, y_std_sc = gp.predict(
            t_test_sc.reshape(-1,1), return_std=True)

        # Inverse-scale back to original meter units
        sc         = err_scalers[col]
        y_pred_m   = sc.inverse_transform(y_pred_sc.reshape(-1,1)).ravel()
        y_std_m    = y_std_sc * sc.scale_[0]

        preds[col]            = y_pred_m
        preds[f"{col}_std"]   = y_std_m

        elapsed = time.time() - t0
        lml     = gp.log_marginal_likelihood_value_
        print(f"  [GP] {col:<16} LML={lml:>8.2f}  ({elapsed:.1f}s)")

    return pd.DataFrame(preds)


# =============================================================================
# STEP 10 — EVALUATION (if test actuals are available)
# =============================================================================

def evaluate_if_possible(pred_df:        pd.DataFrame,
                         test_actual_df: pd.DataFrame = None):
    """
    Compute SW scores if actual test values are available.

    During the real evaluation, you won't have actual test values —
    only the evaluator does. But when testing your own model before
    submission, you can pass your test CSV here to see your SW scores.

    Parameters
    ----------
    pred_df        : predictions DataFrame from fit_and_predict()
    test_actual_df : optional DataFrame with actual test ERR_COLS
    """
    if test_actual_df is None:
        print("\n  [eval] No actual test values provided — skipping SW check")
        return

    print(f"\n  {'─'*55}")
    print(f"  Self-evaluation (actual vs predicted)")
    print(f"  SW Benchmark: W={SW_BENCHMARK_W}")
    print(f"  {'─'*55}")

    rows = []
    for col in ERR_COLS:
        actual   = test_actual_df[col].values
        pred     = pred_df[col].values
        residual = actual - pred
        w, p     = stats.shapiro(residual)
        rmse     = np.sqrt(np.mean(residual**2))
        sym      = "✓" if p >= 0.05 else "✗"
        rows.append([f"{sym} {col}",
                     f"{rmse:.4f}",
                     f"{w:.4f}",
                     f"{p:.4f}",
                     f"{residual.mean():+.4f}",
                     f"{residual.std():.4f}"])

    print(tabulate(rows,
                   headers=["Column","RMSE(m)","SW_W","SW_p",
                             "Res mean","Res std"],
                   tablefmt="rounded_outline"))

    avg_w = np.mean([stats.shapiro(
        test_actual_df[c].values - pred_df[c].values)[0]
        for c in ERR_COLS])
    d = avg_w - SW_BENCHMARK_W
    print(f"\n  Averaged SW_W = {avg_w:.4f}  "
          f"({'↑' if d>=0 else '↓'}{abs(d):.4f} vs benchmark {SW_BENCHMARK_W})")


# =============================================================================
# LOAD TIMESTAMPS
# =============================================================================

def load_timestamps(times_input) -> pd.DatetimeIndex:
    """
    Load test timestamps from a file or string list.

    Accepts:
      - Path to a .txt or .csv file (one timestamp per line)
      - Python list of timestamp strings

    Supported formats: any standard datetime format including:
      2025-09-08 00:11:00
      2025-09-08T00:11:00
      2025/09/08 00:11
      08-09-2025 00:11:00

    Parameters
    ----------
    times_input : str (file path) or list of str

    Returns
    -------
    pd.DatetimeIndex
    """
    if isinstance(times_input, (list, np.ndarray)):
        return pd.to_datetime(times_input)

    if isinstance(times_input, str):
        if not os.path.exists(times_input):
            raise FileNotFoundError(
                f"Timestamps file not found: {times_input}"
            )
        # Try CSV first
        if times_input.endswith(".csv"):
            df = pd.read_csv(times_input)
            # Use first column regardless of name
            return pd.to_datetime(df.iloc[:, 0])
        else:
            # Plain text file — one timestamp per line
            with open(times_input) as f:
                lines = [l.strip() for l in f if l.strip()]
            return pd.to_datetime(lines)

    raise TypeError(f"times_input must be a file path or list, "
                    f"got {type(times_input)}")


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_prediction(train_path:    str,
                   times_input,
                   output_path:   str,
                   sat_type:      str  = None,
                   test_csv_path: str  = None,
                   n_restarts:    int  = N_RESTARTS) -> pd.DataFrame:
    """
    Full end-to-end prediction pipeline.

    Parameters
    ----------
    train_path    : path to training CSV
    times_input   : path to timestamps file OR list of timestamp strings
    output_path   : where to save prediction CSV
    sat_type      : 'GEO' or 'MEO' — auto-detected if None
    test_csv_path : (optional) path to actual test CSV for self-evaluation
    n_restarts    : GP optimizer restarts per column

    Returns
    -------
    pd.DataFrame with predictions
    """
    print("\n" + "═" * 60)
    print("  SIH GNSS Error Prediction — Phase 7")
    print("═" * 60)

    total_start = time.time()

    # ── Step 1: Load and clean ────────────────────────────────────
    print(f"\n  Step 1 │ Loading training data: {train_path}")
    train_df = load_and_clean(train_path)
    print(f"         │ {len(train_df)} clean rows  "
          f"({train_df['utc_time'].min().date()} → "
          f"{train_df['utc_time'].max().date()})")

    # ── Step 2: Detect satellite type ─────────────────────────────
    print(f"\n  Step 2 │ Detecting satellite type")
    if sat_type is not None:
        print(f"         │ Using provided type: {sat_type}")
    else:
        sat_type = detect_satellite_type(train_df)

    # ── Step 3: GEO mode filter ───────────────────────────────────
    if sat_type == "GEO":
        print(f"\n  Step 3 │ Applying GEO upload-mode filter")
        train_df = apply_geo_mode_filter(train_df)
    else:
        print(f"\n  Step 3 │ MEO satellite — no mode filter needed")

    # ── Step 4: Winsorize ─────────────────────────────────────────
    print(f"\n  Step 4 │ Winsorizing outliers ({OUTLIER_K}×IQR)")
    train_df = winsorize(train_df, OUTLIER_K)

    # ── Step 5: Load test timestamps ─────────────────────────────
    print(f"\n  Step 5 │ Loading test timestamps")
    test_timestamps = load_timestamps(times_input)
    print(f"         │ {len(test_timestamps)} timestamps  "
          f"({test_timestamps.min()} → {test_timestamps.max()})")

    # Validate: test should be in the future relative to train
    if test_timestamps.min() <= train_df["utc_time"].max():
        print(f"  ⚠  Warning: some test timestamps overlap with training data")

    # ── Step 6: Build time features ──────────────────────────────
    print(f"\n  Step 6 │ Building t_min feature")
    t_train_raw, t_test_raw, t_ref = build_t_min(train_df, test_timestamps)
    print(f"         │ t_ref = {t_ref}")
    print(f"         │ Train: {t_train_raw.min():.0f} → {t_train_raw.max():.0f} min")
    print(f"         │ Test:  {t_test_raw.min():.0f} → {t_test_raw.max():.0f} min")

    # ── Step 7: Scale ─────────────────────────────────────────────
    print(f"\n  Step 7 │ Fitting StandardScaler on training data")
    t_scaler, err_scalers, t_train_sc, y_scaled = fit_and_scale(
        t_train_raw, train_df)
    t_test_sc = t_scaler.transform(t_test_raw.reshape(-1,1)).ravel()
    print(f"         │ t_min std = {t_scaler.scale_[0]:.2f} min")

    # ── Step 8+9: Fit GP and predict ─────────────────────────────
    print(f"\n  Step 8+9 │ Fitting {len(ERR_COLS)} GP models and predicting")
    print(f"           │ Satellite: {sat_type}  "
          f"Kernel: {'RBF+Per(24h)+White' if sat_type=='GEO' else 'RBF+Per(12h)+Per(24h)+White'}")
    print(f"           │ n_restarts = {n_restarts}  "
          f"train_rows = {len(train_df)}")

    preds_df = fit_and_predict(
        t_train_sc, y_scaled, t_test_sc,
        t_scaler, err_scalers,
        sat_type, n_restarts)

    # ── Step 10: Assemble output ──────────────────────────────────
    output_df = pd.DataFrame({"utc_time": test_timestamps})
    for col in ERR_COLS:
        output_df[col] = preds_df[col].values

    # Full output with uncertainty
    full_output_df = output_df.copy()
    for col in ERR_COLS:
        full_output_df[f"{col}_std"] = preds_df[f"{col}_std"].values

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                exist_ok=True)
    output_df.to_csv(output_path, index=False)

    # Save with uncertainty too
    unc_path = output_path.replace(".csv", "_with_uncertainty.csv")
    full_output_df.to_csv(unc_path, index=False)

    elapsed = time.time() - total_start
    print(f"\n  ✓  Predictions saved: {output_path}")
    print(f"  ✓  With uncertainty:  {unc_path}")
    print(f"  ✓  Total runtime: {elapsed:.1f}s")

    # ── Optional self-evaluation ──────────────────────────────────
    if test_csv_path:
        test_df = pd.read_csv(test_csv_path)
        test_df.columns = ["utc_time","x_error","y_error",
                           "z_error","clock_error"]
        test_df["utc_time"] = pd.to_datetime(test_df["utc_time"])
        test_df = test_df.drop_duplicates(
            subset="utc_time").sort_values("utc_time").reset_index(drop=True)
        evaluate_if_possible(preds_df, test_df)

    # ── Preview ───────────────────────────────────────────────────
    print(f"\n  Prediction preview (first 5 rows):")
    print(output_df.head(5).to_string(index=False))

    return output_df


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SIH GNSS Error Prediction — Phase 7 Submission Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detects satellite type):
  python src/phase7_predict.py \\
      --train  Data/Raw/DATA_GEO_Train.csv \\
      --times  timestamps.txt \\
      --output results/submission_geo.csv

  # With explicit satellite type:
  python src/phase7_predict.py \\
      --train  Data/Raw/DATA_GEO_Train.csv \\
      --times  timestamps.txt \\
      --output results/submission_geo.csv \\
      --type   GEO

  # With self-evaluation (compare against known test CSV):
  python src/phase7_predict.py \\
      --train  Data/Raw/DATA_GEO_Train.csv \\
      --times  Data/Raw/DATA_GEO_Test.csv \\
      --output results/submission_geo.csv \\
      --actual Data/Raw/DATA_GEO_Test.csv

Timestamp file format (.txt — one per line):
  2025-09-08 00:11:00
  2025-09-08 00:24:00
  ...

Or provide the test CSV directly as the --times argument
(the script will use the utc_time column as timestamps).
        """
    )
    parser.add_argument("--train",  required=True,
                        help="Path to training CSV file")
    parser.add_argument("--times",  required=True,
                        help="Path to timestamps file (.txt or .csv)")
    parser.add_argument("--output", required=True,
                        help="Path for output predictions CSV")
    parser.add_argument("--type",   default=None,
                        choices=["GEO","MEO"],
                        help="Satellite type (auto-detected if omitted)")
    parser.add_argument("--actual", default=None,
                        help="(Optional) path to actual test CSV for self-evaluation")
    parser.add_argument("--restarts", type=int, default=N_RESTARTS,
                        help=f"GP optimizer restarts (default: {N_RESTARTS})")
    return parser.parse_args()


# =============================================================================
# SELF-TEST MODE (run without command-line args)
# =============================================================================

def self_test():
    """
    Run predict.py on all 3 satellites using the known data.
    Use this to verify everything works before the real evaluation.
    """
    SRC  = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.dirname(SRC)

    raw_dir = None
    for d in ["Data/Raw","data/raw","Data/raw"]:
        p = os.path.join(BASE, d)
        if os.path.isdir(p): raw_dir = p; break

    proc_dir = None
    for d in ["Data/Processed","data/processed"]:
        p = os.path.join(BASE, d)
        if os.path.isdir(p): proc_dir = p; break

    res_dir = os.path.join(BASE, "results", "phase7_self_test")
    os.makedirs(res_dir, exist_ok=True)

    configs = [
        ("DATA_GEO_Train.csv",  "geo_test_clean.csv",  "geo"),
        ("DATA_MEO_Train.csv",  "meo1_test_clean.csv", "meo1"),
        ("DATA_MEO_Train2.csv", "meo2_test_clean.csv", "meo2"),
    ]

    print("\n" + "═" * 60)
    print("  PHASE 7 SELF-TEST — All 3 Satellites")
    print("═" * 60)

    for train_f, test_f, label in configs:
        train_path  = os.path.join(raw_dir,  train_f) if raw_dir else None
        test_path   = os.path.join(proc_dir, test_f)  if proc_dir else None
        output_path = os.path.join(res_dir, f"pred_{label}.csv")

        if not train_path or not os.path.exists(train_path):
            print(f"\n  ⚠  Skipping {label} — training file not found")
            continue

        print(f"\n{'═'*60}")
        print(f"  Satellite: {label.upper()}")
        print(f"{'═'*60}")

        run_prediction(
            train_path    = train_path,
            times_input   = test_path if test_path else [],
            output_path   = output_path,
            test_csv_path = test_path,
            n_restarts    = 8,
        )

    print("\n" + "═" * 60)
    print("  SELF-TEST COMPLETE")
    print(f"  Predictions saved to: results/phase7_self_test/")
    print("═" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments given — run self-test on known data
        print("  No arguments provided. Running self-test on known data...")
        print("  (To run on new data: python src/phase7_predict.py --help)")
        self_test()
    else:
        # Parse command-line arguments and run
        args = parse_args()
        run_prediction(
            train_path    = args.train,
            times_input   = args.times,
            output_path   = args.output,
            sat_type      = args.type,
            test_csv_path = args.actual,
            n_restarts    = args.restarts,
        )