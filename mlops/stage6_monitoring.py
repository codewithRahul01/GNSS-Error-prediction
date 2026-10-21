"""
GNSS MLOps — Stage 6: Model Monitoring
=======================================
Tracks model health over time by computing SW_W on recent
residuals and detecting data/concept drift using KS tests.
Writes all metrics to the database for Stage 7 (alerting)
and Stage 8 (dashboard) to read.

WHAT IT MONITORS
----------------
  1. SW_W on recent residuals     — is the model still good?
  2. KS drift on error columns    — has the data distribution shifted?
  3. Prediction latency           — is the API still fast?
  4. Rolling SW_W history         — trend over time

HOW TO RUN
----------
  # Run a monitoring check right now:
  python mlops/stage6_monitoring.py --mode run

  # Run for one satellite only:
  python mlops/stage6_monitoring.py --mode run --satellite MEO_B

  # Show latest metrics:
  python mlops/stage6_monitoring.py --mode status

  # Show history (last N checks):
  python mlops/stage6_monitoring.py --mode history --limit 20

FILE LOCATION
-------------
  GNSS-Error-prediction/mlops/stage6_monitoring.py
  Metrics stored in: mlops/database/gnss.db (monitoring_log table)
"""

import argparse, logging, sqlite3, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "mlops" / "database" / "gnss.db"
LOG_DIR = ROOT / "mlops" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "stage6_monitoring.log"),
    ],
)
log = logging.getLogger("STAGE6")

# ── Constants ────────────────────────────────────────────────
ERR_COLS   = ["x_error", "y_error", "z_error", "clock_error"]
SATELLITES = {"GEO_A": "GEO", "MEO_B": "MEO", "MEO_C": "MEO"}

# Thresholds — Stage 7 reads these to decide whether to alert
SW_W_WARN      = 0.85   # SW_W below this → warning
SW_W_CRITICAL  = 0.70   # SW_W below this → critical alert
KS_P_WARN      = 0.05   # KS p-value below this → drift detected
LATENCY_WARN   = 2.0    # seconds per prediction above this → slow API


# ════════════════════════════════════════════════════════════
# DATABASE SETUP
# ════════════════════════════════════════════════════════════

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_monitoring_tables():
    """
    Create monitoring tables if they don't exist.

    monitoring_log    — one row per satellite per check run.
                        Stores SW_W, KS stats, latency, drift flag.
    residual_archive  — stores actual residuals so trends can be
                        re-analysed without re-running predictions.
    """
    conn = get_conn()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            checked_at      TEXT    NOT NULL,
            satellite_id    TEXT    NOT NULL,
            n_test_rows     INTEGER,
            -- Shapiro-Wilk on residuals
            sw_w_avg        REAL,
            sw_w_x          REAL,
            sw_w_y          REAL,
            sw_w_z          REAL,
            sw_w_clock      REAL,
            -- KS drift detection vs training distribution
            ks_stat_x       REAL,
            ks_p_x          REAL,
            ks_stat_y       REAL,
            ks_p_y          REAL,
            ks_stat_z       REAL,
            ks_p_z          REAL,
            ks_stat_clock   REAL,
            ks_p_clock      REAL,
            drift_detected  INTEGER DEFAULT 0,
            -- Prediction speed
            latency_sec     REAL,
            -- Status flags (read by Stage 7)
            status          TEXT    DEFAULT 'ok',
            alert_reason    TEXT,
            -- Which model was active
            model_sw_w_avg  REAL,
            model_trained_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS residual_archive (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            archived_at  TEXT NOT NULL,
            satellite_id TEXT NOT NULL,
            utc_time     TEXT NOT NULL,
            col          TEXT NOT NULL,
            actual       REAL,
            predicted    REAL,
            residual     REAL
        )
    """)

    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════
# STEP 1 — COMPUTE RESIDUALS ON TEST DATA
# ════════════════════════════════════════════════════════════

def compute_residuals(satellite_id: str) -> tuple:
    """
    Load production model, run predictions on test data,
    return (test_df, pred_df, latency_sec).

    WHY test data for monitoring:
    In production you'd pass in the newest day of observations.
    Here we use the held-out test set as a proxy for 'recent data'
    since that's what we have — the logic is identical.
    """
    from mlops.stage4_retrain import load_production_model, predict

    bundle = load_production_model(satellite_id)

    conn = get_conn()
    test_df = pd.read_sql("""
        SELECT utc_time, x_error, y_error, z_error, clock_error
        FROM gnss_errors
        WHERE satellite_id = ?
          AND source_file LIKE '%Test%'
        ORDER BY utc_time
    """, conn, params=(satellite_id,))
    conn.close()

    if test_df.empty:
        raise ValueError(f"No test data for {satellite_id}")

    test_df["utc_time"] = pd.to_datetime(test_df["utc_time"])

    t0 = time.time()
    pred_df = predict(
        bundle["models"],
        bundle["scaler"],
        bundle["t_ref"],
        test_df,
    )
    latency = round(time.time() - t0, 4)

    return test_df, pred_df, latency, bundle


# ════════════════════════════════════════════════════════════
# STEP 2 — SHAPIRO-WILK ON RESIDUALS
# ════════════════════════════════════════════════════════════

def compute_sw_scores(test_df: pd.DataFrame,
                      pred_df: pd.DataFrame) -> tuple:
    """
    Compute Shapiro-Wilk W on (actual - predicted) residuals.

    WHY SW on residuals (not on raw values):
    The GP model assumes residuals are normally distributed.
    If W is high (≥0.85), the assumption holds and the model
    is calibrated. If W drops, the model is missing structure
    — either the data has drifted or the model needs retraining.

    Returns (sw_scores dict, avg_sw float)
    """
    sw_scores = {}
    for col in ERR_COLS:
        res = test_df[col].values - pred_df[col].values
        w, p = stats.shapiro(res)
        rmse  = float(np.sqrt(np.mean(res ** 2)))
        sw_scores[col] = {
            "sw_w": round(float(w), 4),
            "sw_p": round(float(p), 4),
            "rmse": round(rmse, 4),
        }

    avg_sw = round(np.mean([v["sw_w"] for v in sw_scores.values()]), 4)
    return sw_scores, avg_sw


# ════════════════════════════════════════════════════════════
# STEP 3 — KS DRIFT DETECTION
# ════════════════════════════════════════════════════════════

def compute_ks_drift(satellite_id: str,
                     test_df: pd.DataFrame) -> tuple:
    """
    Compare test data distribution vs training distribution
    using the two-sample Kolmogorov-Smirnov test.

    WHY KS test:
    The GP was trained on data from days 1-7. If the satellite's
    error distribution on day 8+ has shifted (different amplitude,
    bias, or variance), the KS statistic will be high and p-value
    low — signalling concept drift. This catches distribution
    changes that SW alone wouldn't catch (SW only measures the
    model's residuals, not the raw data distribution).

    Returns (ks_scores dict, drift_detected bool)
    """
    from mlops.stage3_preprocessing import load_artifacts

    artifacts  = load_artifacts(satellite_id)
    train_df   = artifacts["train_df"]

    ks_scores     = {}
    drift_detected = False

    for col in ERR_COLS:
        train_vals = train_df[col].values
        test_vals  = test_df[col].values

        stat, p = stats.ks_2samp(train_vals, test_vals)
        drift    = p < KS_P_WARN

        ks_scores[col] = {
            "ks_stat": round(float(stat), 4),
            "ks_p"   : round(float(p),    4),
            "drift"  : drift,
        }
        if drift:
            drift_detected = True
            log.info("  DRIFT detected: %s %s  KS=%.3f  p=%.4f",
                     satellite_id, col, stat, p)

    return ks_scores, drift_detected


# ════════════════════════════════════════════════════════════
# STEP 4 — DETERMINE STATUS AND ALERT REASON
# ════════════════════════════════════════════════════════════

def determine_status(avg_sw: float,
                     drift_detected: bool,
                     latency_sec: float) -> tuple:
    """
    Map metric values to a status string and human-readable reason.

    Status levels (read by Stage 7 to decide alert severity):
      ok       — all metrics healthy
      warning  — SW_W low or drift detected or slow API
      critical — SW_W very low (model likely needs retraining)

    Returns (status str, alert_reason str)
    """
    reasons = []

    if avg_sw < SW_W_CRITICAL:
        status = "critical"
        reasons.append(f"SW_W={avg_sw:.4f} < critical threshold {SW_W_CRITICAL}")
    elif avg_sw < SW_W_WARN:
        status = "warning"
        reasons.append(f"SW_W={avg_sw:.4f} < warning threshold {SW_W_WARN}")
    else:
        status = "ok"

    if drift_detected:
        if status == "ok":
            status = "warning"
        reasons.append("KS drift detected in ≥1 error column")

    if latency_sec > LATENCY_WARN:
        if status == "ok":
            status = "warning"
        reasons.append(f"Prediction latency {latency_sec:.2f}s > {LATENCY_WARN}s")

    alert_reason = "; ".join(reasons) if reasons else None
    return status, alert_reason


# ════════════════════════════════════════════════════════════
# STEP 5 — ARCHIVE RESIDUALS
# ════════════════════════════════════════════════════════════

def archive_residuals(satellite_id: str,
                      test_df: pd.DataFrame,
                      pred_df: pd.DataFrame):
    """
    Store residuals in residual_archive table.
    Allows Stage 8 (dashboard) to plot residual trends over time
    without re-running predictions.
    """
    now  = datetime.now(timezone.utc).isoformat()
    conn = get_conn()

    rows = []
    for col in ERR_COLS:
        for i, row in test_df.iterrows():
            rows.append((
                now, satellite_id,
                str(row["utc_time"]),
                col,
                float(row[col]),
                float(pred_df.loc[i, col]),
                float(row[col] - pred_df.loc[i, col]),
            ))

    conn.executemany("""
        INSERT INTO residual_archive
            (archived_at, satellite_id, utc_time, col, actual, predicted, residual)
        VALUES (?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()
    log.info("Archived %d residual rows for %s", len(rows), satellite_id)


# ════════════════════════════════════════════════════════════
# STEP 6 — LOG TO DATABASE
# ════════════════════════════════════════════════════════════

def log_to_db(satellite_id: str,
              n_test: int,
              sw_scores: dict,
              avg_sw: float,
              ks_scores: dict,
              drift_detected: bool,
              latency_sec: float,
              status: str,
              alert_reason: str,
              bundle: dict):
    """Insert one monitoring check row into monitoring_log."""
    conn = get_conn()
    conn.execute("""
        INSERT INTO monitoring_log (
            checked_at, satellite_id, n_test_rows,
            sw_w_avg, sw_w_x, sw_w_y, sw_w_z, sw_w_clock,
            ks_stat_x, ks_p_x, ks_stat_y, ks_p_y,
            ks_stat_z, ks_p_z, ks_stat_clock, ks_p_clock,
            drift_detected, latency_sec,
            status, alert_reason,
            model_sw_w_avg, model_trained_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        satellite_id, n_test,
        avg_sw,
        sw_scores["x_error"]["sw_w"],
        sw_scores["y_error"]["sw_w"],
        sw_scores["z_error"]["sw_w"],
        sw_scores["clock_error"]["sw_w"],
        ks_scores["x_error"]["ks_stat"],   ks_scores["x_error"]["ks_p"],
        ks_scores["y_error"]["ks_stat"],   ks_scores["y_error"]["ks_p"],
        ks_scores["z_error"]["ks_stat"],   ks_scores["z_error"]["ks_p"],
        ks_scores["clock_error"]["ks_stat"],ks_scores["clock_error"]["ks_p"],
        int(drift_detected),
        latency_sec,
        status, alert_reason,
        bundle.get("sw_w_avg"),
        bundle.get("trained_at", "")[:19],
    ))
    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════
# FULL MONITORING PIPELINE — one satellite
# ════════════════════════════════════════════════════════════

def run_check(satellite_id: str) -> dict:
    """
    Run the full monitoring pipeline for one satellite.
    Returns a result dict with all metrics.
    """
    log.info("─" * 50)
    log.info("Monitoring: %s", satellite_id)
    log.info("─" * 50)

    # 1. Compute residuals
    test_df, pred_df, latency, bundle = compute_residuals(satellite_id)
    log.info("Test rows: %d  Prediction latency: %.3fs", len(test_df), latency)

    # 2. SW_W on residuals
    sw_scores, avg_sw = compute_sw_scores(test_df, pred_df)
    for col, v in sw_scores.items():
        sym = "✓" if v["sw_w"] >= SW_W_WARN else "✗"
        log.info("  %s %-16s SW_W=%.4f  RMSE=%.4fm", sym, col,
                 v["sw_w"], v["rmse"])

    # 3. KS drift
    ks_scores, drift_detected = compute_ks_drift(satellite_id, test_df)

    # 4. Status
    status, alert_reason = determine_status(avg_sw, drift_detected, latency)
    status_sym = {"ok": "✓", "warning": "⚠", "critical": "✗"}[status]
    log.info("%s Status: %s  SW_W_avg=%.4f  drift=%s",
             status_sym, status.upper(), avg_sw, drift_detected)
    if alert_reason:
        log.info("  Reason: %s", alert_reason)

    # 5. Archive residuals
    archive_residuals(satellite_id, test_df, pred_df)

    # 6. Log to DB
    log_to_db(satellite_id, len(test_df), sw_scores, avg_sw,
              ks_scores, drift_detected, latency, status,
              alert_reason, bundle)

    return {
        "satellite_id"  : satellite_id,
        "sw_w_avg"      : avg_sw,
        "sw_scores"     : sw_scores,
        "ks_scores"     : ks_scores,
        "drift_detected": drift_detected,
        "latency_sec"   : latency,
        "status"        : status,
        "alert_reason"  : alert_reason,
        "n_test"        : len(test_df),
    }


# ════════════════════════════════════════════════════════════
# PUBLIC API — used by Stage 7 (alerting)
# ════════════════════════════════════════════════════════════

def get_latest_metrics(satellite_id: str = None) -> list:
    """
    Return the most recent monitoring result for each satellite.
    Called by Stage 7 to check if alerts need to be sent.

    Parameters
    ----------
    satellite_id : if given, return only that satellite's latest row.
                   if None, return latest row for all satellites.
    """
    conn = get_conn()
    if satellite_id:
        rows = conn.execute("""
            SELECT satellite_id, checked_at, sw_w_avg,
                   drift_detected, latency_sec, status, alert_reason
            FROM monitoring_log
            WHERE satellite_id = ?
            ORDER BY checked_at DESC LIMIT 1
        """, (satellite_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT satellite_id, checked_at, sw_w_avg,
                   drift_detected, latency_sec, status, alert_reason
            FROM monitoring_log
            WHERE id IN (
                SELECT MAX(id) FROM monitoring_log
                GROUP BY satellite_id
            )
            ORDER BY satellite_id
        """).fetchall()
    conn.close()

    return [
        {
            "satellite_id"  : r[0],
            "checked_at"    : r[1][:19],
            "sw_w_avg"      : r[2],
            "drift_detected": bool(r[3]),
            "latency_sec"   : r[4],
            "status"        : r[5],
            "alert_reason"  : r[6],
        }
        for r in rows
    ]


def get_sw_history(satellite_id: str, limit: int = 30) -> pd.DataFrame:
    """
    Return SW_W history for a satellite.
    Called by Stage 8 (dashboard) to plot trends.
    """
    conn = get_conn()
    df = pd.read_sql("""
        SELECT checked_at, sw_w_avg, sw_w_x, sw_w_y, sw_w_z, sw_w_clock,
               drift_detected, status
        FROM monitoring_log
        WHERE satellite_id = ?
        ORDER BY checked_at DESC
        LIMIT ?
    """, conn, params=(satellite_id, limit))
    conn.close()
    return df.iloc[::-1].reset_index(drop=True)  # oldest first


def needs_retraining(satellite_id: str) -> bool:
    """
    Returns True if the latest check shows status='critical'.
    Called by Stage 7 to trigger auto-retraining.
    """
    metrics = get_latest_metrics(satellite_id)
    if not metrics:
        return False
    return metrics[0]["status"] == "critical"


# ════════════════════════════════════════════════════════════
# MODES
# ════════════════════════════════════════════════════════════

def mode_run(satellite_id: str = None):
    init_monitoring_tables()

    sats = [satellite_id] if satellite_id else list(SATELLITES.keys())

    print("\n" + "=" * 58)
    print("  STAGE 6 — Model Monitoring")
    print("=" * 58 + "\n")

    results = []
    for sat in sats:
        if sat not in SATELLITES:
            print(f"  ✗ Unknown satellite: {sat}")
            continue
        try:
            r = run_check(sat)
            results.append(r)
            sym = {"ok": "✓", "warning": "⚠", "critical": "✗"}[r["status"]]
            drift = "DRIFT" if r["drift_detected"] else "no drift"
            print(f"  {sym}  {sat:<8}  SW_W={r['sw_w_avg']:.4f}  "
                  f"{drift:<10}  {r['status'].upper()}")
            if r["alert_reason"]:
                print(f"       ↳ {r['alert_reason']}")
        except Exception as e:
            log.exception("Monitoring failed: %s", sat)
            print(f"  ✗  {sat}: {e}")

    print()
    mode_status()
    return results


def mode_status():
    init_monitoring_tables()
    metrics = get_latest_metrics()

    print("─" * 65)
    print("  LATEST MONITORING STATUS")
    print("─" * 65)
    if not metrics:
        print("  No monitoring data yet. Run: python mlops/stage6_monitoring.py --mode run")
    else:
        print(f"  {'Satellite':<10} {'Checked at':<22} {'SW_W':<8} "
              f"{'Drift':<7} {'Latency':<10} Status")
        print("  " + "─" * 61)
        for m in metrics:
            drift   = "YES" if m["drift_detected"] else "no"
            sym     = {"ok": "✓", "warning": "⚠", "critical": "✗"}.get(m["status"], "?")
            latency = f"{m['latency_sec']:.3f}s" if m["latency_sec"] else "N/A"
            print(f"  {sym} {m['satellite_id']:<9} "
                  f"{m['checked_at']:<22} "
                  f"{str(m['sw_w_avg']):<8} "
                  f"{drift:<7} "
                  f"{latency:<10} "
                  f"{m['status'].upper()}")
    print("─" * 65 + "\n")


def mode_history(satellite_id: str = None, limit: int = 10):
    init_monitoring_tables()

    sats = [satellite_id] if satellite_id else list(SATELLITES.keys())

    for sat in sats:
        df = get_sw_history(sat, limit=limit)
        print(f"\n  SW_W history — {sat} (last {len(df)} checks)")
        print("  " + "─" * 55)
        if df.empty:
            print("  No history yet.")
        else:
            for _, row in df.iterrows():
                sym = {"ok": "✓", "warning": "⚠", "critical": "✗"}.get(
                    row["status"], "?")
                print(f"  {sym} {row['checked_at'][:19]}  "
                      f"avg={row['sw_w_avg']:.4f}  "
                      f"x={row['sw_w_x']:.3f}  "
                      f"y={row['sw_w_y']:.3f}  "
                      f"z={row['sw_w_z']:.3f}  "
                      f"clk={row['sw_w_clock']:.3f}")


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GNSS MLOps Stage 6 — Model Monitoring")
    parser.add_argument(
        "--mode", required=True,
        choices=["run", "status", "history"],
        help="run=check now, status=latest metrics, history=trend")
    parser.add_argument(
        "--satellite", default=None,
        choices=list(SATELLITES.keys()),
        help="Monitor one satellite (default: all)")
    parser.add_argument(
        "--limit", type=int, default=10,
        help="History rows to show (default: 10)")
    args = parser.parse_args()

    if args.mode == "run":
        mode_run(args.satellite)
    elif args.mode == "status":
        mode_status()
    elif args.mode == "history":
        mode_history(args.satellite, args.limit)


if __name__ == "__main__":
    main()