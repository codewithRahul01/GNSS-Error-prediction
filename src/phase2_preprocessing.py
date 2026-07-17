"""
=============================================================================
  SIH GNSS ERROR PREDICTION PROJECT
  PHASE 2 — Data Preprocessing
=============================================================================

  WHAT THIS PHASE DOES
  ─────────────────────
  Takes the 7 clean CSVs from Phase 1 (data/Processed/) and transforms
  them into model-ready dataframes for all three satellites.

  6 Steps performed in order:
    Step 1 — Load cleaned data from Data/Processed/
    Step 2 — Winsorize outliers in TRAINING data only
    Step 3 — Add t_min  (continuous time in minutes — the GP input)
    Step 4 — Add cyclical sin/cos features (daily + half-daily periods)
    Step 5 — Add upload boundary feature (GEO only)
    Step 6 — Fit StandardScaler on train, apply same scaler to test
    Step 7 — Save all outputs

  HOW TO RUN
  ──────────
    python src/phase2_preprocessing.py

  INPUT  (reads from Data/Processed/)
  ─────
    geo_train_recent.csv   ← GEO train, Sep 6-7 only (15-min upload mode)
    geo_test_clean.csv
    meo1_train_clean.csv
    meo1_test_clean.csv
    meo2_train_clean.csv
    meo2_test_clean.csv

  OUTPUT (saves to Data/Processed/ and results/)
  ──────
    geo_train_ready.csv    ← scaled + all features → feed to GP model
    geo_test_ready.csv
    meo1_train_ready.csv
    meo1_test_ready.csv
    meo2_train_ready.csv
    meo2_test_ready.csv
    geo_train_featured.csv     ← unscaled with features (for debugging)
    geo_test_featured.csv
    meo1_train_featured.csv
    meo1_test_featured.csv
    meo2_train_featured.csv
    meo2_test_featured.csv
    results/scaler_geo.pkl     ← fitted scalers (needed by predict.py)
    results/scaler_meo1.pkl
    results/scaler_meo2.pkl

  PRE-REQUISITES
  ──────────────
    Phase 1 must be complete (Data/Processed/*.csv must exist)
    pip install pandas numpy scipy scikit-learn joblib tabulate
"""

# IMPORTS

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib
from tabulate import tabulate


# PATHS — auto-detected from this file's location

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

# Try both capitalisation variants (Mac is case-sensitive)
for _d in ["Data", "data"]:
    for _p in ["Processed", "processed"]:
        _candidate = os.path.join(BASE_DIR, _d, _p)
        if os.path.isdir(_candidate):
            PROC_DIR = _candidate
            break

RES_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

# CONSTANTS

ERR_COLS = ["x_error", "y_error", "z_error", "clock_error"]

OUTLIER_K      = 3.0      # k × IQR fence for winsorization
PERIOD_DAILY   = 1440.0   # 24-hour period in minutes
PERIOD_HALFD   = 720.0    # 12-hour period in minutes
UPLOAD_GAP_MIN = 100.0    # gap > this (min) signals a new upload cycle

# Columns to scale (error cols + all feature cols)
SCALE_COLS = ["t_min",
              "sin_daily", "cos_daily",
              "sin_halfd", "cos_halfd"] + ERR_COLS


# STEP 1 — LOAD

def load_processed(filename: str) -> pd.DataFrame:
    """
    Load a cleaned CSV from Data/Processed/ and parse utc_time.

    WHY read from Processed/ not Raw/:
      Phase 1 already removed duplicate rows. Reading Raw/ again
      would bring those duplicates back, corrupting the dataset.

    Parameters
    ----------
    filename : e.g. "geo_train_recent.csv"

    Returns
    -------
    pd.DataFrame  sorted by utc_time ascending
    """
    path = os.path.join(PROC_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  ✗  File not found: {path}"
            f"\n     Run phase0_1_eda.py first to create processed data."
        )

    df = pd.read_csv(path)
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    return df.sort_values("utc_time").reset_index(drop=True)


# STEP 2 — WINSORIZE OUTLIERS


def winsorize(df: pd.DataFrame,
              cols: list = None,
              k: float = OUTLIER_K) -> tuple:
    """
    Clip extreme values in training data to the k×IQR fence.

    WHY winsorize:
      GEO has large spikes (up to ±40m) at upload boundaries.
      If left in, the GP optimizer tries to fit those spikes and
      learns the wrong kernel hyperparameters. Clipping lets the
      kernel focus on the real smooth periodic trend instead.

    WHY ONLY on training data (NEVER test):
      The evaluator compares our predictions to the REAL test values,
      including those large spikes. We must predict those spikes —
      not pretend they don't exist. Winsorizing test data would make
      residuals look artificially small and cheat the SW score.

    WHY 3×IQR (not the standard 1.5×IQR):
      1.5×IQR clips too aggressively. GEO upload events create
      legitimately large values that are real signal, not noise.
      3×IQR only removes the truly extreme values (top ~0.3%).

    Parameters
    ----------
    df   : training dataframe only
    cols : columns to winsorize (default: all 4 error columns)
    k    : IQR multiplier

    Returns
    -------
    (winsorized_df, bounds_dict)
      bounds_dict = {col: (lower_fence, upper_fence, n_clipped)}
    """
    if cols is None:
        cols = ERR_COLS

    df     = df.copy()
    bounds = {}

    for col in cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - k * IQR
        hi  = Q3 + k * IQR
        n   = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col]     = df[col].clip(lo, hi)
        bounds[col] = (lo, hi, n)

    return df, bounds


# STEP 3 — ADD t_min

def add_t_min(df: pd.DataFrame,
              t_ref: pd.Timestamp = None) -> tuple:
    """
    Add 't_min' column = minutes elapsed since t_ref.

    WHY t_min instead of raw timestamps:
      The Gaussian Process model computes correlations based on
      the DISTANCE between input points. It cannot compute distance
      between datetime strings — it needs plain numbers.

      t_min converts every timestamp to a float:
        minutes elapsed since the first training observation.

      Example (GEO, t_ref = Sep 6 00:00):
        Sep 6  00:00  →  t_min =    0.0
        Sep 6  00:15  →  t_min =   15.0
        Sep 7  00:00  →  t_min = 1440.0
        Sep 8  00:00  →  t_min = 2880.0  ← test day (future)

    CRITICAL RULE — same t_ref for both train AND test:
      If you use a different t_ref for test, the GP sees test
      timestamps as if they were near the training start, and
      extrapolation becomes interpolation — completely wrong.
      Always pass the training t_ref when computing test t_min.

    Parameters
    ----------
    df    : dataframe with utc_time column
    t_ref : reference timestamp. If None → uses df minimum timestamp.
            For test data, always pass the training t_ref explicitly.

    Returns
    -------
    (df_with_t_min, t_ref_used)
    """
    if t_ref is None:
        t_ref = df["utc_time"].min()

    df         = df.copy()
    df["t_min"] = (df["utc_time"] - t_ref).dt.total_seconds() / 60.0
    return df, t_ref


# =============================================================================
# STEP 4 — ADD CYCLICAL SIN/COS FEATURES
# =============================================================================

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sin/cos encodings of the daily and half-daily orbital periods.

    WHY cyclical encoding (and not raw hour or minute):
      GNSS errors repeat with the satellite's orbital period:
        GEO  →  24h (1440 min) — geostationary, tied to Earth's rotation
        MEO  →  12h ( 720 min) — orbital resonance

      If we gave the GP a raw "hour of day" number, midnight (hour=0)
      and 11:59 PM (hour=23) would look 23 hours apart. But they are
      only 1 minute apart in orbital terms. The GP would model a big
      discontinuity at midnight that doesn't exist in reality.

      Sin/cos wraps time into a circle where midnight connects
      smoothly back to midnight:
        t=0    (midnight) → sin=0.00,  cos=1.00
        t=360  (6 AM)     → sin=1.00,  cos=0.00
        t=720  (noon)     → sin=0.00,  cos=-1.00
        t=1080 (6 PM)     → sin=-1.00, cos=0.00
        t=1440 (midnight) → sin=0.00,  cos=1.00  ← same as t=0 ✓

    WHY both sin AND cos (not just one):
      sin alone cannot tell apart 6 AM and 6 PM (both have |sin|=1).
      The PAIR (sin, cos) is like x,y on a unit circle — it uniquely
      identifies any point in a cycle. Always use both together.

    Features added:
      sin_daily, cos_daily  →  24h cycle (GEO dominant)
      sin_halfd, cos_halfd  →  12h cycle (MEO dominant)

    Parameters
    ----------
    df : dataframe with 't_min' column (run add_t_min first)

    Returns
    -------
    pd.DataFrame with 4 new columns
    """
    df = df.copy()
    df["sin_daily"] = np.sin(2 * np.pi * df["t_min"] / PERIOD_DAILY)
    df["cos_daily"] = np.cos(2 * np.pi * df["t_min"] / PERIOD_DAILY)
    df["sin_halfd"] = np.sin(2 * np.pi * df["t_min"] / PERIOD_HALFD)
    df["cos_halfd"] = np.cos(2 * np.pi * df["t_min"] / PERIOD_HALFD)
    return df


# =============================================================================
# STEP 5 — UPLOAD BOUNDARY FEATURE (GEO ONLY)
# =============================================================================

def add_upload_features(df: pd.DataFrame,
                        gap_threshold: float = UPLOAD_GAP_MIN) -> pd.DataFrame:
    """
    Add 'time_since_upload_min' and 'upload_segment' columns.

    WHY this feature (GEO only):
      GEO errors follow a sawtooth pattern within each upload cycle:
        → Just AFTER an upload:    error ≈ 0m   (fresh ephemeris)
        → Just BEFORE next upload: error is large (stale prediction)

      By giving the GP this explicit "minutes since last reset" feature,
      it can learn: "when time_since_upload is small → error is small"
      and "when it's large → error is growing". This is exactly the
      sawtooth pattern the evaluator wants us to capture.

    HOW upload boundaries are detected:
      When time gap between consecutive rows > 100 min, we assume
      either a new upload happened OR data collection paused.
      Either way, after the gap the ephemeris state is effectively reset.

    WHY only GEO (not MEO):
      MEO satellites in this dataset don't show the same clear
      upload-cycle sawtooth. Their errors are smoother and dominated
      by orbital mechanics rather than upload resets.

    Columns added:
      time_since_upload_min : float — minutes since last detected reset
      upload_segment        : int   — counter, increments at each reset

    Parameters
    ----------
    df            : dataframe with utc_time column (sorted ascending)
    gap_threshold : gaps larger than this → new upload segment

    Returns
    -------
    pd.DataFrame with 2 new columns
    """
    df   = df.copy().sort_values("utc_time").reset_index(drop=True)
    gaps = df["utc_time"].diff().dt.total_seconds().fillna(0) / 60

    is_new_upload        = gaps > gap_threshold
    df["upload_segment"] = is_new_upload.cumsum().astype(int)

    # Build time_since_upload_min row by row
    time_since      = []
    seg_start_time  = df["utc_time"].iloc[0]

    for i in range(len(df)):
        if is_new_upload.iloc[i]:
            seg_start_time = df["utc_time"].iloc[i]
        elapsed = (df["utc_time"].iloc[i] - seg_start_time).total_seconds() / 60
        time_since.append(elapsed)

    df["time_since_upload_min"] = time_since
    return df


# =============================================================================
# STEP 6 — SCALING
# =============================================================================

def fit_scaler(train_df: pd.DataFrame,
               cols: list = None) -> StandardScaler:
    """
    Fit a StandardScaler on the training data.

    WHY scale at all:
      The GP kernel measures distances between inputs and amplitudes
      of outputs. If t_min ranges 0→9000 (minutes) but x_error
      ranges -5→+5 (meters), the optimizer finds very different
      "natural" hyperparameter values for each — making convergence
      slow and unreliable. After scaling everything has mean=0, std=1
      and the optimizer works in the same space for all features.

    WHY StandardScaler (mean=0, std=1) vs MinMaxScaler (0→1):
      The GP assumes output data is roughly Gaussian with zero mean.
      StandardScaler directly satisfies this assumption.
      MinMaxScaler squashes everything into [0,1] which distorts
      the shape and breaks the Gaussian assumption.

    WHY fit on TRAINING data only (never include test):
      If you compute mean/std on train+test together, the scaler
      "knows" the future (test data's mean and variance). This is
      called DATA LEAKAGE. The model will appear to perform better
      than it really does on unseen data.

      CORRECT workflow:
        scaler.fit(train_data)       ← learns mean/std from train only
        scaler.transform(train_data) ← scales train
        scaler.transform(test_data)  ← scales test using TRAIN stats

    Parameters
    ----------
    train_df : training dataframe — never include test rows here
    cols     : columns to fit (default: SCALE_COLS constant)

    Returns
    -------
    fitted StandardScaler with .feature_names_in_ attribute set
    """
    if cols is None:
        cols = SCALE_COLS

    available               = [c for c in cols if c in train_df.columns]
    scaler                  = StandardScaler()
    scaler.fit(train_df[available].values)
    scaler.feature_names_in_ = np.array(available)
    return scaler


def apply_scaler(df: pd.DataFrame,
                 scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to transform a dataframe's columns.

    Only transforms columns that the scaler was fitted on AND
    that exist in df. Safe to call on both train and test.

    Parameters
    ----------
    df     : train OR test dataframe
    scaler : fitted StandardScaler from fit_scaler()

    Returns
    -------
    pd.DataFrame with transformed values in the relevant columns
    """
    df      = df.copy()
    fitted  = list(scaler.feature_names_in_)
    present = [c for c in fitted if c in df.columns]
    df[present] = scaler.transform(df[present].values)
    return df


def inverse_scale_errors(scaled_preds: np.ndarray,
                         scaler: StandardScaler) -> np.ndarray:
    """
    Convert scaled GP predictions back to original meter units.

    After the GP predicts in scaled space, we must reverse the
    scaling before computing residuals for the SW test.
    The SW test must run on residuals in METERS, not scaled units.

    Parameters
    ----------
    scaled_preds : shape (n_samples, 4) — one col per error type
    scaler       : the fitted scaler used during training

    Returns
    -------
    np.ndarray  shape (n_samples, 4)  in original meter units
    """
    fitted = list(scaler.feature_names_in_)
    result = scaled_preds.copy().astype(float)

    for i, col in enumerate(ERR_COLS):
        if col in fitted:
            idx          = fitted.index(col)
            result[:, i] = scaled_preds[:, i] * scaler.scale_[idx] \
                           + scaler.mean_[idx]
    return result


# =============================================================================
# FULL PIPELINE — one satellite at a time
# =============================================================================

def preprocess_satellite(train_filename: str,
                         test_filename:  str,
                         sat_name:       str,
                         sat_type:       str,
                         winsorize_k:    float = OUTLIER_K) -> dict:
    """
    Run the complete Phase 2 pipeline for one satellite.

    Order of operations:
      1. Load clean train + test from Data/Processed/
      2. Winsorize training data (clip outliers)
      3. Add t_min to both (same t_ref)
      4. Add sin/cos cyclical features to both
      5. Add upload features to both (GEO only)
      6. Fit scaler on train → apply to both
      7. Save: *_featured.csv (unscaled), *_ready.csv (scaled), scaler.pkl

    Parameters
    ----------
    train_filename : e.g. "geo_train_recent.csv"
    test_filename  : e.g. "geo_test_clean.csv"
    sat_name       : 'geo' | 'meo1' | 'meo2'
    sat_type       : 'GEO' | 'MEO'
    winsorize_k    : IQR multiplier for outlier clipping

    Returns
    -------
    dict with keys:
      train_raw, test_raw         → DataFrames before scaling
      train_scaled, test_scaled   → DataFrames after scaling
      scaler                      → fitted StandardScaler
      t_ref                       → reference timestamp for t_min
      bounds                      → winsorization bounds per column
    """
    sep = "─" * 58
    print(f"\n  {sep}")
    print(f"  Satellite: {sat_name.upper()}  (type: {sat_type})")
    print(f"  {sep}")

    # ── Step 1: Load ──────────────────────────────────────────────
    print(f"\n  Step 1 │ Loading data")
    train = load_processed(train_filename)
    test  = load_processed(test_filename)
    print(f"         │ Train : {len(train):>4} rows  "
          f"{train['utc_time'].min().date()} → {train['utc_time'].max().date()}")
    print(f"         │ Test  : {len(test):>4} rows  "
          f"{test['utc_time'].min().date()} → {test['utc_time'].max().date()}")

    # ── Step 2: Winsorize (train only) ────────────────────────────
    print(f"\n  Step 2 │ Winsorizing training outliers  (k={winsorize_k}×IQR)")
    train, bounds = winsorize(train, ERR_COLS, winsorize_k)
    total = sum(v[2] for v in bounds.values())
    if total == 0:
        print(f"         │ No outliers found — nothing clipped")
    else:
        print(f"         │ Total values clipped: {total}")
        for col, (lo, hi, n) in bounds.items():
            if n > 0:
                print(f"         │   {col:<16}  {n:>2} clipped  "
                      f"fence=[{lo:.3f}, {hi:.3f}]")

    # ── Step 3: Add t_min ─────────────────────────────────────────
    print(f"\n  Step 3 │ Adding t_min  (continuous time in minutes)")
    train, t_ref = add_t_min(train, t_ref=None)
    test,  _     = add_t_min(test,  t_ref=t_ref)
    print(f"         │ t_ref (reference timestamp) = {t_ref}")
    print(f"         │ Train t_min : {train['t_min'].min():>8.1f} → "
          f"{train['t_min'].max():.1f} min")
    print(f"         │ Test  t_min : {test['t_min'].min():>8.1f} → "
          f"{test['t_min'].max():.1f} min  ← future ✓")

    # ── Step 4: Cyclical features ─────────────────────────────────
    print(f"\n  Step 4 │ Adding cyclical sin/cos features")
    train = add_cyclical_features(train)
    test  = add_cyclical_features(test)
    print(f"         │ Added: sin_daily, cos_daily "
          f"(24h period)  +  sin_halfd, cos_halfd (12h period)")

    # ── Step 5: Upload boundary features (GEO only) ───────────────
    if sat_type == "GEO":
        print(f"\n  Step 5 │ Adding upload boundary features  (GEO only)")
        train = add_upload_features(train)
        test  = add_upload_features(test)
        n_tr  = int(train["upload_segment"].max()) + 1
        n_te  = int(test["upload_segment"].max())  + 1
        print(f"         │ Train: {n_tr} upload segments detected")
        print(f"         │ Test : {n_te} upload segments detected")
        print(f"         │ Added: time_since_upload_min, upload_segment")
    else:
        print(f"\n  Step 5 │ Skipping upload features  (not needed for MEO)")

    # ── Step 6: Scale ─────────────────────────────────────────────
    print(f"\n  Step 6 │ Fitting StandardScaler on training data")

    scale_cols = SCALE_COLS.copy()
    if sat_type == "GEO":
        scale_cols += ["time_since_upload_min"]

    scaler   = fit_scaler(train, scale_cols)
    train_sc = apply_scaler(train, scaler)
    test_sc  = apply_scaler(test,  scaler)

    fitted_names = list(scaler.feature_names_in_)
    print(f"         │ Columns scaled: {len(fitted_names)}")
    print(f"         │ {fitted_names}")

    # Print mean/std for the 4 error columns
    rows = []
    for col in ERR_COLS:
        if col in fitted_names:
            idx = fitted_names.index(col)
            rows.append([col,
                         f"{scaler.mean_[idx]:>9.4f}",
                         f"{scaler.scale_[idx]:>9.4f}"])
    print(f"\n         │ Error column scaler statistics:")
    tbl = tabulate(rows, headers=["Column","Mean (subtracted)",
                                  "Std (divided by)"], tablefmt="simple")
    for line in tbl.split("\n"):
        print(f"         │   {line}")

    # ── Step 7: Save ──────────────────────────────────────────────
    print(f"\n  Step 7 │ Saving outputs")

    # Unscaled (featured) — useful for debugging and visualization
    p = os.path.join(PROC_DIR, f"{sat_name}_train_featured.csv")
    train.to_csv(p, index=False)
    p = os.path.join(PROC_DIR, f"{sat_name}_test_featured.csv")
    test.to_csv(p, index=False)

    # Scaled and ready for GP
    p = os.path.join(PROC_DIR, f"{sat_name}_train_ready.csv")
    train_sc.to_csv(p, index=False)
    print(f"         │ ✓  {sat_name}_train_ready.csv   "
          f"({len(train_sc)} rows, {len(train_sc.columns)} cols)")

    p = os.path.join(PROC_DIR, f"{sat_name}_test_ready.csv")
    test_sc.to_csv(p, index=False)
    print(f"         │ ✓  {sat_name}_test_ready.csv    "
          f"({len(test_sc)} rows, {len(test_sc.columns)} cols)")

    # Scaler
    scaler_path = os.path.join(RES_DIR, f"scaler_{sat_name}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"         │ ✓  scaler_{sat_name}.pkl  (saved to results/)")

    return {
        "train_raw"    : train,
        "test_raw"     : test,
        "train_scaled" : train_sc,
        "test_scaled"  : test_sc,
        "scaler"       : scaler,
        "t_ref"        : t_ref,
        "bounds"       : bounds,
    }


# =============================================================================
# VERIFICATION
# =============================================================================

def verify(result: dict, sat_name: str) -> bool:
    """
    Run sanity checks on the preprocessed output.

    Checks performed:
      1. t_min is strictly increasing in training data
      2. Test t_min > Train t_min max  (test is truly in the future)
      3. Scaled error columns have mean≈0, std≈1 on training data
      4. No NaN values in train or test
      5. All required columns are present
    """
    print(f"\n  Verifying {sat_name.upper()} ...")
    train = result["train_scaled"]
    test  = result["test_scaled"]
    ok    = True

    # 1. t_min monotonic
    if train["t_min"].is_monotonic_increasing:
        print(f"  ✓  t_min strictly increasing in train")
    else:
        print(f"  ✗  t_min NOT monotonic — check sorting"); ok = False

    # 2. Test is in the future
    if test["t_min"].min() > train["t_min"].max():
        print(f"  ✓  Test t_min ({test['t_min'].min():.1f}) > "
              f"Train max ({train['t_min'].max():.1f})  ← future confirmed")
    else:
        print(f"  ✗  Test t_min overlaps train — t_ref mismatch"); ok = False

    # 3. Scaled error columns ≈ N(0,1)
    for col in ERR_COLS:
        if col in train.columns:
            mu  = train[col].mean()
            std = train[col].std()
            sym = "✓" if abs(mu) < 0.15 and 0.85 < std < 1.15 else "⚠"
            print(f"  {sym}  {col:<16} scaled mean={mu:>7.4f}  std={std:.4f}")

    # 4. No NaNs
    nans = train.isnull().sum().sum() + test.isnull().sum().sum()
    if nans == 0:
        print(f"  ✓  No NaN values in train or test")
    else:
        print(f"  ✗  {nans} NaN values found"); ok = False

    # 5. Required columns exist
    required = ["utc_time","t_min","sin_daily","cos_daily",
                "sin_halfd","cos_halfd"] + ERR_COLS
    missing  = [c for c in required if c not in train.columns]
    if not missing:
        print(f"  ✓  All required columns present")
    else:
        print(f"  ✗  Missing columns: {missing}"); ok = False

    return ok


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def summary(results: dict):
    """
    Print before/after comparison for each satellite.

    Shows: raw SW statistic vs scaled SW statistic to confirm
    that scaling does NOT change the distribution shape
    (only centers and normalises the scale — SW stays the same).
    """
    print("\n" + "═" * 65)
    print("  PHASE 2 SUMMARY — Before vs After Preprocessing")
    print("═" * 65)

    for sat, res in results.items():
        print(f"\n  ── {sat.upper()} ──")
        rows = []
        for col in ERR_COLS:
            raw    = res["train_raw"][col].values
            scaled = res["train_scaled"][col].values
            rw, _  = stats.shapiro(raw)
            sw, _  = stats.shapiro(scaled)
            rows.append([col,
                         f"{raw.mean():>8.3f}",
                         f"{raw.std():>7.3f}",
                         f"{rw:.4f}",
                         f"{scaled.mean():>8.4f}",
                         f"{scaled.std():>7.4f}",
                         f"{sw:.4f}"])
        hdrs = ["Column",
                "Raw mean","Raw std","Raw SW_W",
                "Scaled mean","Scaled std","Scaled SW_W"]
        print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))

    print(f"""
  KEY OBSERVATIONS
  ────────────────────────────────────────────────────────────
  1. Scaled mean ≈ 0.000  (centering worked ✓)
  2. Scaled std  ≈ 1.000  (normalization worked ✓)
  3. Raw SW_W  =  Scaled SW_W  (scaling does NOT change the
     distribution shape — only location and scale change)
  4. SW_W is still far below 0.9810 — that is EXPECTED here.
     The SW score only improves after the GP model removes the
     systematic patterns. Phase 2 just prepares the data.

  Target: residual SW_W ≥ 0.9810  after GP prediction (Phase 4)
  ────────────────────────────────────────────────────────────""")


# =============================================================================
# COLUMN GUIDE
# =============================================================================

def print_column_guide():
    print("""
  COLUMN GUIDE — what is in each *_ready.csv file
  ────────────────────────────────────────────────────────────
  utc_time              original timestamp (NOT scaled)
  x_error               SCALED  (mean=0, std=1)
  y_error               SCALED
  z_error               SCALED
  clock_error           SCALED
  t_min                 SCALED  continuous time (GP input feature)
  sin_daily             SCALED  sin of 24h cycle
  cos_daily             SCALED  cos of 24h cycle
  sin_halfd             SCALED  sin of 12h cycle
  cos_halfd             SCALED  cos of 12h cycle
  time_since_upload_min SCALED  (GEO files only)
  upload_segment        INTEGER counter (GEO files only, NOT scaled)

  HOW PHASE 4 (GP model) uses these:
    INPUT  to GP  →  t_min (scaled)
    OUTPUT of GP  →  x_error, y_error, z_error, clock_error (scaled)
    The GP learns:  f(t_min) ≈ error at that time
    After prediction, scaler.inverse_transform() converts back to meters
    THEN we compute residuals and run the Shapiro-Wilk test.
  ────────────────────────────────────────────────────────────""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "═" * 65)
    print("  SIH GNSS Error Prediction Project")
    print("  PHASE 2 — Data Preprocessing")
    print("═" * 65)

    results = {}

    # ── Satellite A: GEO ─────────────────────────────────────────────────────
    # Uses geo_train_recent.csv (Sep 3-7, 15-min upload mode only)
    # NOT geo_train_clean.csv (that has Sep 1-5 which is 120-min mode)
    results["geo"] = preprocess_satellite(
        train_filename = "geo_train_recent.csv",
        test_filename  = "geo_test_clean.csv",
        sat_name       = "geo",
        sat_type       = "GEO",
    )

    # ── Satellite B: MEO1 ────────────────────────────────────────────────────
    results["meo1"] = preprocess_satellite(
        train_filename = "meo1_train_clean.csv",
        test_filename  = "meo1_test_clean.csv",
        sat_name       = "meo1",
        sat_type       = "MEO",
    )

    # ── Satellite C: MEO2 ────────────────────────────────────────────────────
    results["meo2"] = preprocess_satellite(
        train_filename = "meo2_train_clean.csv",
        test_filename  = "meo2_test_clean.csv",
        sat_name       = "meo2",
        sat_type       = "MEO",
    )

    # ── Verification ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  Verification Checks")
    print("═" * 65)

    all_ok = True
    for sat, res in results.items():
        ok = verify(res, sat)
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  ✓  All verification checks passed")
    else:
        print("\n  ⚠  Some checks failed — review output above")

    # ── Summary + Column Guide ────────────────────────────────────────────────
    summary(results)
    print_column_guide()

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  PHASE 2 COMPLETE")
    print("═" * 65)
    print("""
  Files created in Data/Processed/:
    geo_train_ready.csv    geo_test_ready.csv
    meo1_train_ready.csv   meo1_test_ready.csv
    meo2_train_ready.csv   meo2_test_ready.csv
    geo_train_featured.csv  (unscaled + all features — for debugging)
    meo1_train_featured.csv
    meo2_train_featured.csv

  Files created in results/:
    scaler_geo.pkl    scaler_meo1.pkl    scaler_meo2.pkl

  Next step → run:  python src/phase3_baselines.py
    """)


if __name__ == "__main__":
    main()