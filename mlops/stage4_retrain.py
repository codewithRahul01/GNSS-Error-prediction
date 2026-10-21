"""
GNSS MLOps — Stage 4: Triggered Retraining
===========================================
Loads preprocessed data from Stage 3, fits 4 GP models
per satellite, evaluates using Shapiro-Wilk, compares
against the current best model in the registry, and
promotes the new model if it is better.

Can be triggered:
  - Manually (run once)
  - On a schedule (daily via APScheduler)
  - Automatically by Stage 7 when SW_W drops

HOW TO RUN
----------
  # Retrain all satellites:
  python mlops/stage4_retrain.py --mode run

  # Retrain one satellite:
  python mlops/stage4_retrain.py --mode run --satellite GEO_A

  # Show registry:
  python mlops/stage4_retrain.py --mode status

FILE LOCATION
-------------
  GNSS-Error-prediction/mlops/stage4_retrain.py
  Models saved to: mlops/models/
"""

import argparse, json, logging, sqlite3, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, WhiteKernel, ConstantKernel as C,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DB_PATH   = ROOT / "mlops" / "database" / "gnss.db"
MODEL_DIR = ROOT / "mlops" / "models"
LOG_DIR   = ROOT / "mlops" / "logs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "stage4_retrain.log"),
    ],
)
log = logging.getLogger("STAGE4")

# ── Constants ────────────────────────────────────────────────
ERR_COLS   = ["x_error", "y_error", "z_error", "clock_error"]
N_RESTARTS = 8
SATELLITES = {"GEO_A": "GEO", "MEO_B": "MEO", "MEO_C": "MEO"}


# ════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_registry():
    """Create model_registry table if it does not exist."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            satellite_id   TEXT    NOT NULL,
            trained_at     TEXT    NOT NULL,
            train_rows     INTEGER,
            sw_w_avg       REAL,
            sw_w_x         REAL,
            sw_w_y         REAL,
            sw_w_z         REAL,
            sw_w_clock     REAL,
            model_path     TEXT,
            status         TEXT    DEFAULT 'staging',
            n_restarts     INTEGER,
            satellite_type TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_current_best_sw(satellite_id: str) -> float:
    """
    Return the sw_w_avg of the current production model.
    Returns 0.0 if no production model exists yet —
    meaning any new model will be promoted on first run.
    """
    conn = get_conn()
    row  = conn.execute("""
        SELECT sw_w_avg FROM model_registry
        WHERE satellite_id = ? AND status = 'production'
        ORDER BY trained_at DESC LIMIT 1
    """, (satellite_id,)).fetchone()
    conn.close()
    return float(row[0]) if row else 0.0


def register_model(satellite_id, satellite_type, train_rows,
                   sw_scores, avg_sw, model_path, n_restarts):
    """Insert a new row into model_registry with status='staging'."""
    conn = get_conn()
    conn.execute("""
        INSERT INTO model_registry
            (satellite_id, trained_at, train_rows,
             sw_w_avg, sw_w_x, sw_w_y, sw_w_z, sw_w_clock,
             model_path, status, n_restarts, satellite_type)
        VALUES (?,?,?,?,?,?,?,?,?,'staging',?,?)
    """, (
        satellite_id,
        datetime.now(timezone.utc).isoformat(),
        train_rows,
        avg_sw,
        sw_scores["x_error"]["sw_w"],
        sw_scores["y_error"]["sw_w"],
        sw_scores["z_error"]["sw_w"],
        sw_scores["clock_error"]["sw_w"],
        str(model_path),
        n_restarts,
        satellite_type,
    ))
    conn.commit()
    conn.close()


def promote_model(satellite_id, model_path):
    """
    Demote all existing production models to 'archived',
    then promote the new model (by model_path) to 'production'.

    WHY keep old models:
    If the new model causes issues in production, you can
    roll back by flipping the status field — no retraining needed.
    """
    conn = get_conn()
    # Archive existing production
    conn.execute("""
        UPDATE model_registry
        SET status = 'archived'
        WHERE satellite_id = ? AND status = 'production'
    """, (satellite_id,))
    # Promote new model
    conn.execute("""
        UPDATE model_registry
        SET status = 'production'
        WHERE satellite_id = ? AND model_path = ?
    """, (satellite_id, str(model_path)))
    conn.commit()
    conn.close()
    log.info("Promoted to production: %s", model_path.name)


# ════════════════════════════════════════════════════════════
# KERNEL BUILDER
# ════════════════════════════════════════════════════════════

def build_kernel(sat_type: str, t_std: float):
    """
    Build the GP kernel for GEO or MEO.

    GEO:  RBF (smooth drift) + ExpSineSquared(24h) + White (spikes)
    MEO:  RBF (trend) + ExpSineSquared(12h orbital)
                      + ExpSineSquared(24h solar) + White (noise)

    Periods are converted to scaled units: period_min / t_std.
    WHY: the scaler changes the units of t_min from raw minutes
    to standard deviations. The kernel period must be in the
    same units as the (scaled) input, not raw minutes.
    """
    p12h = 720.0  / t_std
    p1d  = 1440.0 / t_std

    if sat_type == "GEO":
        return (
            C(1.0, (0.01, 100)) * RBF(0.5, (0.01, 10))
          + C(0.5, (0.01,  50)) * ExpSineSquared(
                0.3, p1d, (0.01, 5), (p1d * 0.5, p1d * 2))
          + WhiteKernel(1.0, (0.01, 100))
        )
    return (
        C(1.0, (0.01, 100)) * RBF(0.5, (0.01, 10))
      + C(0.3, (0.01,  20)) * ExpSineSquared(
            0.3, p12h, (0.01, 5), (p12h * 0.5, p12h * 2))
      + C(0.2, (0.01,  10)) * ExpSineSquared(
            0.3, p1d,  (0.01, 5), (p1d  * 0.5, p1d  * 2))
      + WhiteKernel(0.5, (0.01, 50))
    )


# ════════════════════════════════════════════════════════════
# SCALE HELPERS
# ════════════════════════════════════════════════════════════

def _sc(v, scaler, col):
    f = list(scaler.feature_names_in_)
    i = f.index(col)
    return (v - scaler.mean_[i]) / scaler.scale_[i]

def _inv(v, scaler, col):
    f = list(scaler.feature_names_in_)
    i = f.index(col)
    return v * scaler.scale_[i] + scaler.mean_[i]


# ════════════════════════════════════════════════════════════
# STEP 1 — FIT 4 GP MODELS
# ════════════════════════════════════════════════════════════

def fit_gp_models(train_df: pd.DataFrame,
                  scaler,
                  sat_type: str,
                  n_restarts: int = N_RESTARTS) -> dict:
    """
    Fit one GP per error column using the preprocessed training data.

    Both input (t_min) and output (error values) are scaled before
    fitting. The GP works in scaled space and returns scaled predictions.
    Stage 4's predict function inverse-scales back to meters.

    Returns dict: {col: fitted_GaussianProcessRegressor}
    """
    t_std = scaler.scale_[list(scaler.feature_names_in_).index("t_min")]
    t_tr  = _sc(train_df["t_min"].values, scaler, "t_min").reshape(-1, 1)
    models = {}

    for col in ERR_COLS:
        t0  = time.time()
        gp  = GaussianProcessRegressor(
            kernel=build_kernel(sat_type, t_std),
            n_restarts_optimizer=n_restarts,
            alpha=0.0,
        )
        gp.fit(t_tr, _sc(train_df[col].values, scaler, col))
        elapsed = time.time() - t0
        log.info("  %-16s LML=%7.2f  (%.1fs)",
                 col, gp.log_marginal_likelihood_value_, elapsed)
        models[col] = gp

    return models


# ════════════════════════════════════════════════════════════
# STEP 2 — PREDICT AT TEST TIMESTAMPS
# ════════════════════════════════════════════════════════════

def predict(models: dict, scaler, t_ref: pd.Timestamp,
            test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict at test timestamps using fitted GP models.

    Applies the SAME t_ref and scaler as training.
    Returns DataFrame with predicted error values in meters.
    """
    test_df = test_df.copy()
    test_df["utc_time"] = pd.to_datetime(test_df["utc_time"])
    test_df["t_min"]    = (
        (test_df["utc_time"] - t_ref).dt.total_seconds() / 60
    )
    t_te = _sc(test_df["t_min"].values, scaler, "t_min").reshape(-1, 1)

    pred = {"utc_time": test_df["utc_time"].values}
    for col in ERR_COLS:
        yp_sc, ys_sc = models[col].predict(t_te, return_std=True)
        pred[col]            = _inv(yp_sc, scaler, col)
        pred[f"{col}_std"]   = ys_sc * scaler.scale_[
            list(scaler.feature_names_in_).index(col)]

    return pd.DataFrame(pred)


# ════════════════════════════════════════════════════════════
# STEP 3 — EVALUATE (Shapiro-Wilk on residuals)
# ════════════════════════════════════════════════════════════

def evaluate(test_df: pd.DataFrame,
             pred_df: pd.DataFrame) -> tuple:
    """
    Compute Shapiro-Wilk W on residuals for each error column.

    The SW test checks whether residuals (actual - predicted)
    are normally distributed. W close to 1.0 = normal = good.
    The evaluation benchmark is W = 0.9810.

    Returns (sw_scores dict, avg_sw_w float)
    """
    sw_scores = {}
    for col in ERR_COLS:
        act  = test_df[col].values
        pred = pred_df[col].values
        res  = act - pred
        w, p = stats.shapiro(res)
        rmse = float(np.sqrt(np.mean(res ** 2)))
        sw_scores[col] = {
            "sw_w": round(float(w), 4),
            "sw_p": round(float(p), 4),
            "rmse": round(rmse, 4),
            "h0_rejected": int(p < 0.05),
        }
        sym = "✓" if p >= 0.05 else "✗"
        log.info("  %s %-16s SW_W=%.4f  p=%.4f  RMSE=%.4fm",
                 sym, col, w, p, rmse)

    avg_sw = float(np.mean([v["sw_w"] for v in sw_scores.values()]))
    return sw_scores, round(avg_sw, 4)


# ════════════════════════════════════════════════════════════
# STEP 4 — SAVE MODEL
# ════════════════════════════════════════════════════════════

def save_model(satellite_id: str,
               models: dict,
               scaler,
               t_ref: pd.Timestamp,
               meta: dict) -> Path:
    """
    Save all 4 GP models + scaler + metadata as one .pkl bundle.

    WHY bundle everything:
    Stage 6 (API) loads the model and must also have the
    scaler and t_ref to transform incoming timestamps correctly.
    One file = one atomic unit that is either complete or not.

    File is named with a timestamp so old versions are never
    overwritten — the registry tracks which version is active.
    """
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sid  = satellite_id.lower().replace("_", "")
    path = MODEL_DIR / f"gp_{sid}_{ts}.pkl"

    bundle = {
        "satellite_id"  : satellite_id,
        "satellite_type": meta["satellite_type"],
        "models"        : models,
        "scaler"        : scaler,
        "t_ref"         : t_ref,
        "trained_at"    : datetime.now(timezone.utc).isoformat(),
        "sw_scores"     : meta.get("sw_scores", {}),
        "sw_w_avg"      : meta.get("sw_w_avg", 0.0),
        "train_rows"    : meta.get("train_rows", 0),
    }
    joblib.dump(bundle, path)
    log.info("Model saved: %s", path.name)
    return path


# ════════════════════════════════════════════════════════════
# PUBLIC API — used by Stage 6 (API) and Stage 7 (monitoring)
# ════════════════════════════════════════════════════════════

def load_production_model(satellite_id: str) -> dict:
    """
    Load the current production model bundle for a satellite.
    Called by Stage 6 (FastAPI) to serve predictions.

    Returns the saved bundle dict with keys:
      models, scaler, t_ref, satellite_type,
      sw_w_avg, trained_at
    """
    conn = get_conn()
    row  = conn.execute("""
        SELECT model_path FROM model_registry
        WHERE satellite_id = ? AND status = 'production'
        ORDER BY trained_at DESC LIMIT 1
    """, (satellite_id,)).fetchone()
    conn.close()

    if not row:
        raise FileNotFoundError(
            f"No production model for {satellite_id}. "
            "Run: python mlops/stage4_retrain.py --mode run"
        )

    path = Path(row[0])
    if not path.exists():
        raise FileNotFoundError(f"Model file missing: {path}")

    return joblib.load(path)


def predict_from_production(satellite_id: str,
                             timestamps: list) -> pd.DataFrame:
    """
    Load production model and predict at given timestamps.
    Convenience function for Stage 6 API calls.

    Parameters
    ----------
    satellite_id : 'GEO_A', 'MEO_B', or 'MEO_C'
    timestamps   : list of timestamp strings or pd.Timestamps

    Returns
    -------
    DataFrame with utc_time + 4 error columns + 4 std columns
    """
    bundle = load_production_model(satellite_id)
    test_df = pd.DataFrame({"utc_time": pd.to_datetime(timestamps)})
    # Add dummy error cols so predict() doesn't fail on column check
    for col in ERR_COLS:
        test_df[col] = 0.0

    return predict(
        bundle["models"],
        bundle["scaler"],
        bundle["t_ref"],
        test_df,
    )[["utc_time"] + ERR_COLS + [f"{c}_std" for c in ERR_COLS]]


# ════════════════════════════════════════════════════════════
# FULL RETRAIN PIPELINE — one satellite
# ════════════════════════════════════════════════════════════

def retrain(satellite_id: str,
            n_restarts: int = N_RESTARTS,
            force_promote: bool = False) -> dict:
    """
    Complete retraining pipeline for one satellite:
      1. Load Stage 3 artifacts (training data + scaler)
      2. Fit 4 GP models
      3. Load test data from DB for evaluation
      4. Predict and compute SW scores
      5. Compare with current production model
      6. Save and promote if better (or force_promote=True)

    Parameters
    ----------
    satellite_id   : 'GEO_A', 'MEO_B', or 'MEO_C'
    n_restarts     : GP optimizer restarts (more = better, slower)
    force_promote  : promote even if SW score is not better
                     (use on first run when no production model exists)

    Returns
    -------
    dict with satellite_id, sw_scores, sw_w_avg, promoted, model_path
    """
    log.info("=" * 55)
    log.info("Retraining: %s", satellite_id)
    log.info("=" * 55)

    # ── Load Stage 3 artifacts ───────────────────────────────
    try:
        sys.path.insert(0, str(ROOT))
        from mlops.stage3_preprocessing import load_artifacts
    except ImportError:
        from stage3_preprocessing import load_artifacts

    artifacts    = load_artifacts(satellite_id)
    train_df     = artifacts["train_df"]
    scaler       = artifacts["scaler"]
    t_ref        = artifacts["t_ref"]
    sat_type     = artifacts["satellite_type"]

    log.info("Loaded: %d training rows  t_ref=%s",
             len(train_df), t_ref.date())

    # ── Fit GPs ──────────────────────────────────────────────
    log.info("Fitting %d GPs  (n_restarts=%d)...", len(ERR_COLS), n_restarts)
    t0     = time.time()
    models = fit_gp_models(train_df, scaler, sat_type, n_restarts)
    log.info("Training time: %.1fs", time.time() - t0)

    # ── Load test data for evaluation ────────────────────────
    conn    = get_conn()
    test_df = pd.read_sql("""
        SELECT utc_time, x_error, y_error, z_error, clock_error
        FROM gnss_errors
        WHERE satellite_id = ?
          AND source_file LIKE '%Test%'
        ORDER BY utc_time
    """, conn, params=(satellite_id,))
    conn.close()

    if test_df.empty:
        log.warning("No test data found for %s — skipping SW eval", satellite_id)
        sw_scores, avg_sw = {col: {"sw_w":0.0,"sw_p":0.0,"rmse":0.0,"h0_rejected":1}
                             for col in ERR_COLS}, 0.0
    else:
        log.info("Evaluating on %d test rows...", len(test_df))
        pred_df   = predict(models, scaler, t_ref, test_df)
        sw_scores, avg_sw = evaluate(test_df, pred_df)

    log.info("SW_W average: %.4f", avg_sw)

    # ── Compare with current production ──────────────────────
    current_best = get_current_best_sw(satellite_id)
    is_better    = avg_sw > current_best
    promoted     = is_better or force_promote

    log.info("Current best: %.4f  New: %.4f  Better: %s",
             current_best, avg_sw, is_better)

    # ── Save model bundle ─────────────────────────────────────
    meta = {
        "satellite_type": sat_type,
        "sw_scores"     : sw_scores,
        "sw_w_avg"      : avg_sw,
        "train_rows"    : len(train_df),
    }
    model_path = save_model(satellite_id, models, scaler, t_ref, meta)

    # ── Register in DB ────────────────────────────────────────
    register_model(satellite_id, sat_type, len(train_df),
                   sw_scores, avg_sw, model_path, n_restarts)

    # ── Promote if better ─────────────────────────────────────
    if promoted:
        promote_model(satellite_id, model_path)
        log.info("✓ Promoted to production")
    else:
        log.info("✗ Not promoted (%.4f ≤ current best %.4f)",
                 avg_sw, current_best)

    return {
        "satellite_id": satellite_id,
        "sw_scores"   : sw_scores,
        "sw_w_avg"    : avg_sw,
        "promoted"    : promoted,
        "model_path"  : model_path,
    }


# ════════════════════════════════════════════════════════════
# MODES
# ════════════════════════════════════════════════════════════

def mode_run(satellite_id: str = None, n_restarts: int = N_RESTARTS):
    """Retrain one or all satellites."""
    init_registry()

    sats = [satellite_id] if satellite_id else list(SATELLITES.keys())

    print("\n" + "=" * 55)
    print("  STAGE 4 — GP Retraining")
    print("=" * 55 + "\n")

    for sat in sats:
        if sat not in SATELLITES:
            print(f"  ✗ Unknown satellite: {sat}")
            print(f"    Valid options: {list(SATELLITES.keys())}")
            continue
        try:
            r = retrain(sat, n_restarts=n_restarts, force_promote=True)
            sym = "↑" if r["promoted"] else "─"
            print(f"  {sym}  {sat:<8}  "
                  f"SW_W={r['sw_w_avg']:.4f}  "
                  f"{'PROMOTED' if r['promoted'] else 'staging'}")
        except Exception as e:
            log.exception("Retrain failed: %s", sat)
            print(f"  ✗  {sat}: {e}")

    print()
    mode_status()


def mode_status():
    """Show model registry."""
    init_registry()
    conn = get_conn()
    rows = conn.execute("""
        SELECT satellite_id, trained_at, train_rows,
               sw_w_avg, status, n_restarts
        FROM model_registry
        ORDER BY trained_at DESC
        LIMIT 15
    """).fetchall()
    conn.close()

    print("\n" + "─" * 72)
    print("  MODEL REGISTRY")
    print("─" * 72)
    if not rows:
        print("  No models yet. Run: python mlops/stage4_retrain.py --mode run")
    else:
        print(f"  {'Satellite':<10} {'Trained at':<22} "
              f"{'Rows':<6} {'SW_W':<8} {'Status':<12} Restarts")
        print("  " + "─" * 68)
        for sat, ts, rows_, sw, status, nr in rows:
            star = " ★" if status == "production" else ""
            print(f"  {sat:<10} {ts[:19]:<22} "
                  f"{str(rows_):<6} {str(sw):<8} {status:<12} {nr}{star}")
    print("─" * 72 + "\n")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GNSS MLOps Stage 4 — GP Retraining")
    parser.add_argument(
        "--mode", required=True,
        choices=["run", "status"],
        help="run=retrain, status=show registry")
    parser.add_argument(
        "--satellite", default=None,
        choices=list(SATELLITES.keys()),
        help="Retrain one satellite (default: all)")
    parser.add_argument(
        "--restarts", type=int, default=N_RESTARTS,
        help=f"GP optimizer restarts (default: {N_RESTARTS})")
    args = parser.parse_args()

    if args.mode == "run":
        mode_run(args.satellite, args.restarts)
    elif args.mode == "status":
        mode_status()


if __name__ == "__main__":
    main()