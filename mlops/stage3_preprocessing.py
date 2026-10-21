"""
GNSS MLOps — Stage 3: Automated Preprocessing
==============================================
Fetches training data from the Stage 1 database,
applies the full preprocessing pipeline, fits a
StandardScaler, and saves the artifacts to disk.

Stage 4 (retraining) loads these artifacts and
calls transform() on new data — never fit() again.

HOW TO RUN
----------
  # Preprocess all 3 satellites and save artifacts:
  python mlops/stage3_preprocessing.py --mode run

  # Check what artifacts exist:
  python mlops/stage3_preprocessing.py --mode status

FILE LOCATION
-------------
  GNSS-Error-prediction/mlops/stage3_preprocessing.py
  Artifacts saved to: mlops/artifacts/
"""

import argparse, json, logging, sqlite3, sys, warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DB_PATH      = ROOT / "mlops" / "database" / "gnss.db"
ARTIFACT_DIR = ROOT / "mlops" / "artifacts"
LOG_DIR      = ROOT / "mlops" / "logs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "stage3_preprocessing.log"),
    ],
)
log = logging.getLogger("STAGE3")

# ── Constants ────────────────────────────────────────────────
ERR_COLS  = ["x_error", "y_error", "z_error", "clock_error"]
FEAT_COLS = ["t_min"] + ERR_COLS
WINSOR_K  = 3.0

SATELLITES = {
    "GEO_A": "GEO",
    "MEO_B": "MEO",
    "MEO_C": "MEO",
}


# ════════════════════════════════════════════════════════════
# STEP 1 — Load training data from Stage 1 database
# ════════════════════════════════════════════════════════════

def load_training_data(satellite_id: str) -> pd.DataFrame:
    """
    Query the Stage 1 database for training rows only.
    Returns raw unprocessed DataFrame sorted by utc_time.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run Stage 1 first: python mlops/stage1_ingestion.py --mode seed"
        )

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT utc_time, x_error, y_error, z_error, clock_error
           FROM gnss_errors
           WHERE satellite_id = ?
             AND source_file LIKE '%Train%'
           ORDER BY utc_time""",
        conn, params=(satellite_id,),
    )
    conn.close()

    if df.empty:
        raise ValueError(f"No training data found for {satellite_id}")

    df["utc_time"] = pd.to_datetime(df["utc_time"])
    log.info("%s: loaded %d training rows from DB", satellite_id, len(df))
    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# STEP 2 — GEO upload-mode filter
# ════════════════════════════════════════════════════════════

def geo_mode_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    GEO ONLY. Keep only days in 15-minute upload mode.

    WHY: GEO training has two modes —
      Sep 1-5: 120-min intervals (old upload schedule)
      Sep 6-7: 15-min intervals (matches test day Sep 8)
    Training on 120-min data teaches the wrong sawtooth frequency.
    We keep only days where median gap ≤ 30 min.
    """
    df = df.copy()
    df["_date"]    = df["utc_time"].dt.date
    df["_gap_min"] = df["utc_time"].diff().dt.total_seconds().fillna(0) / 60
    daily_med      = df.groupby("_date")["_gap_min"].median()
    keep_days      = daily_med[daily_med <= 30].index

    filtered = df[df["_date"].isin(keep_days)].drop(
        columns=["_date", "_gap_min"]
    ).reset_index(drop=True)

    log.info("GEO mode filter: %d → %d rows (kept %d 15-min days)",
             len(df), len(filtered), len(keep_days))
    return filtered


# ════════════════════════════════════════════════════════════
# STEP 3 — Winsorize outliers
# ════════════════════════════════════════════════════════════

def winsorize(df: pd.DataFrame, k: float = WINSOR_K) -> pd.DataFrame:
    """
    Clip extreme training values at k×IQR fences.
    Applied to TRAINING DATA ONLY — never to test/new data.

    WHY: Upload-boundary spikes drag the GP kernel optimizer
    toward fitting noise instead of the smooth periodic trend.
    3×IQR only clips genuinely extreme values (~top 0.3%).
    """
    df = df.copy()
    for col in ERR_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1 - k * IQR, Q3 + k * IQR
        clipped = int(((df[col] < lo) | (df[col] > hi)).sum())
        if clipped:
            log.info("Winsorize %s: clipped %d values to [%.3f, %.3f]",
                     col, clipped, lo, hi)
        df[col] = df[col].clip(lo, hi)
    return df


# ════════════════════════════════════════════════════════════
# STEP 4 — Build t_min feature
# ════════════════════════════════════════════════════════════

def build_t_min(df: pd.DataFrame,
                t_ref: pd.Timestamp = None) -> tuple:
    """
    Convert timestamps to continuous minutes since t_ref.

    WHY t_min: GP kernel computes distances between inputs.
    It cannot compute distances between datetime strings —
    it needs plain floats. t_min = (timestamp - t_ref) in minutes.

    WHY same t_ref for train and test:
    If train uses its own start and test uses test start,
    the GP sees test timestamps as if they were in the training
    range — turning extrapolation into interpolation. Wrong.
    Always compute test t_min using the TRAINING t_ref.

    Returns (df_with_t_min, t_ref)
    """
    df = df.copy()
    if t_ref is None:
        t_ref = df["utc_time"].min()
    df["t_min"] = (df["utc_time"] - t_ref).dt.total_seconds() / 60
    return df, t_ref


# ════════════════════════════════════════════════════════════
# STEP 5 — Fit StandardScaler
# ════════════════════════════════════════════════════════════

def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """
    Fit StandardScaler on [t_min, x_error, y_error, z_error, clock_error].

    WHY scale t_min:
    GP kernel length scale is in the same units as inputs.
    Raw t_min (0–9000 min) forces a huge length scale search.
    Scaled t_min (std≈1) keeps optimizer numerically stable.

    WHY scale error columns:
    ConstantKernel amplitudes are in output units.
    GEO errors (std≈5m) vs MEO errors (std≈0.3m) differ by 15×.
    After scaling, all 4 columns have comparable amplitudes,
    so the same kernel bounds work for all columns.

    WHY fit on TRAIN only:
    Fitting on train+test gives the scaler knowledge of the future.
    That is data leakage. Fit once on training, save, apply forever.
    """
    scaler = StandardScaler()
    scaler.fit(df[FEAT_COLS])
    log.info("Scaler fitted: t_min mean=%.1f std=%.1f",
             scaler.mean_[0], scaler.scale_[0])
    return scaler


# ════════════════════════════════════════════════════════════
# STEP 6 — Apply scaler (transform only — never refit)
# ════════════════════════════════════════════════════════════

def transform(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply a previously fitted scaler to a DataFrame.
    Returns a new DataFrame with scaled FEAT_COLS added as *_scaled columns.
    Original columns are preserved.
    """
    df = df.copy()
    scaled = scaler.transform(df[FEAT_COLS])
    for i, col in enumerate(FEAT_COLS):
        df[f"{col}_scaled"] = scaled[:, i]
    return df


# ════════════════════════════════════════════════════════════
# STEP 7 — Save artifacts
# ════════════════════════════════════════════════════════════

def save_artifacts(satellite_id: str,
                   train_df: pd.DataFrame,
                   scaler: StandardScaler,
                   t_ref: pd.Timestamp,
                   sat_type: str,
                   meta: dict):
    """
    Save everything Stage 4 needs to retrain and predict:
      - processed training CSV
      - fitted scaler pkl
      - metadata JSON (t_ref, sat_type, row count, timestamp)
    """
    sid = satellite_id.lower().replace("_", "")  # geo_a → geoa

    csv_path    = ARTIFACT_DIR / f"train_{sid}.csv"
    scaler_path = ARTIFACT_DIR / f"scaler_{sid}.pkl"
    meta_path   = ARTIFACT_DIR / f"meta_{sid}.json"

    train_df.to_csv(csv_path, index=False)
    joblib.dump(scaler, scaler_path)

    meta.update({
        "satellite_id": satellite_id,
        "satellite_type": sat_type,
        "t_ref": str(t_ref),
        "rows": len(train_df),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "scaler_path": str(scaler_path),
        "csv_path": str(csv_path),
    })
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info("Saved: %s  %s  %s", csv_path.name,
             scaler_path.name, meta_path.name)
    return csv_path, scaler_path, meta_path


# ════════════════════════════════════════════════════════════
# FULL PIPELINE — one satellite
# ════════════════════════════════════════════════════════════

def run_pipeline(satellite_id: str) -> dict:
    """
    Complete preprocessing pipeline for one satellite.
    Returns a result dict for logging and Stage 4 to use.
    """
    sat_type = SATELLITES[satellite_id]
    log.info("=" * 50)
    log.info("Preprocessing: %s (%s)", satellite_id, sat_type)
    log.info("=" * 50)

    # 1. Load from DB
    df = load_training_data(satellite_id)
    raw_rows = len(df)

    # 2. GEO mode filter
    if sat_type == "GEO":
        df = geo_mode_filter(df)

    # 3. Winsorize
    df = winsorize(df)

    # 4. t_min
    df, t_ref = build_t_min(df)

    # 5. Fit scaler
    scaler = fit_scaler(df)

    # 6. Transform (adds *_scaled columns)
    df = transform(df, scaler)

    # 7. Save
    meta = {
        "winsor_k": WINSOR_K,
        "feat_cols": FEAT_COLS,
        "raw_rows": raw_rows,
    }
    csv_p, scaler_p, meta_p = save_artifacts(
        satellite_id, df, scaler, t_ref, sat_type, meta)

    result = {
        "satellite_id": satellite_id,
        "satellite_type": sat_type,
        "raw_rows": raw_rows,
        "processed_rows": len(df),
        "t_ref": t_ref,
        "scaler": scaler,
        "train_df": df,
        "scaler_path": scaler_p,
        "csv_path": csv_p,
        "meta_path": meta_p,
    }

    log.info("Done: %s → %d rows processed", satellite_id, len(df))
    return result


# ════════════════════════════════════════════════════════════
# PUBLIC API — used by Stage 4
# ════════════════════════════════════════════════════════════

def load_artifacts(satellite_id: str) -> dict:
    """
    Load saved preprocessing artifacts for one satellite.
    Called by Stage 4 to get training data and scaler.

    Returns dict with keys:
      train_df, scaler, t_ref, satellite_type, meta
    """
    sid       = satellite_id.lower().replace("_", "")
    meta_path = ARTIFACT_DIR / f"meta_{sid}.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No artifacts for {satellite_id}. "
            "Run: python mlops/stage3_preprocessing.py --mode run"
        )

    meta         = json.loads(meta_path.read_text())
    scaler       = joblib.load(meta["scaler_path"])
    train_df     = pd.read_csv(meta["csv_path"])
    train_df["utc_time"] = pd.to_datetime(train_df["utc_time"])

    return {
        "train_df"      : train_df,
        "scaler"        : scaler,
        "t_ref"         : pd.Timestamp(meta["t_ref"]),
        "satellite_type": meta["satellite_type"],
        "meta"          : meta,
    }


def transform_new_data(df: pd.DataFrame,
                       satellite_id: str) -> pd.DataFrame:
    """
    Apply saved preprocessing to NEW incoming data.
    This is what Stage 4 calls on each new day of data.

    CRITICAL: loads the SAVED scaler — never refits.
    Applies the SAME t_ref from training so GP sees
    new data as future timestamps (t_min > training max).
    """
    artifacts = load_artifacts(satellite_id)
    scaler    = artifacts["scaler"]
    t_ref     = artifacts["t_ref"]

    df = df.copy()
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    df, _ = build_t_min(df, t_ref=t_ref)
    df    = transform(df, scaler)
    return df


# ════════════════════════════════════════════════════════════
# MODES
# ════════════════════════════════════════════════════════════

def mode_run():
    """Run preprocessing for all 3 satellites."""
    print("\n" + "=" * 55)
    print("  STAGE 3 — Preprocessing all satellites")
    print("=" * 55 + "\n")

    results = []
    for sat_id in SATELLITES:
        try:
            r = run_pipeline(sat_id)
            results.append(r)
            print(f"  ✓  {sat_id:<8} "
                  f"raw={r['raw_rows']:>3}  "
                  f"processed={r['processed_rows']:>3}  "
                  f"t_ref={r['t_ref'].date()}")
        except Exception as e:
            log.error("Failed: %s — %s", sat_id, e)
            print(f"  ✗  {sat_id}: {e}")

    print(f"\n  Artifacts saved to: {ARTIFACT_DIR}")
    print()


def mode_status():
    """Show saved artifacts."""
    print("\n" + "=" * 55)
    print("  STAGE 3 — Artifact Status")
    print("=" * 55)

    for sat_id in SATELLITES:
        sid       = sat_id.lower().replace("_", "")
        meta_path = ARTIFACT_DIR / f"meta_{sid}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print(f"\n  {sat_id}:")
            print(f"    rows        : {meta.get('rows')}")
            print(f"    t_ref       : {meta.get('t_ref')}")
            print(f"    processed_at: {meta.get('processed_at','?')[:19]}")
            print(f"    scaler_path : {Path(meta.get('scaler_path','')).name}")
        else:
            print(f"\n  {sat_id}: ✗ no artifacts — run --mode run")

    print()


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GNSS MLOps Stage 3 — Preprocessing")
    parser.add_argument(
        "--mode", required=True,
        choices=["run", "status"],
        help="run=preprocess all satellites, status=show artifacts")
    args = parser.parse_args()

    if args.mode == "run":
        mode_run()
    elif args.mode == "status":
        mode_status()


if __name__ == "__main__":
    main()