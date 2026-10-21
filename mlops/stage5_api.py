"""
GNSS MLOps — Stage 5: FastAPI Prediction API
=============================================
Exposes GP model predictions as a REST API.
Models are loaded ONCE at startup and cached
in memory — predictions served in <100ms.

HOW TO RUN
----------
  pip install fastapi uvicorn

  # Start the API server:
  python mlops/stage5_api.py

  # Or with uvicorn directly:
  uvicorn mlops.stage5_api:app --host 0.0.0.0 --port 8000 --reload

  # Interactive docs (auto-generated):
  http://localhost:8000/docs

ENDPOINTS
---------
  GET  /health                  → server + model status
  GET  /satellites              → list available satellites
  GET  /model/{satellite_id}    → model info for one satellite
  POST /predict                 → predict errors at timestamps
  POST /predict/batch           → predict for multiple satellites
  GET  /registry                → full model registry from DB

FILE LOCATION
-------------
  GNSS-Error-prediction/mlops/stage5_api.py
"""

import logging, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

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
        logging.FileHandler(LOG_DIR / "stage5_api.log"),
    ],
)
log = logging.getLogger("STAGE5")

# ── Constants ────────────────────────────────────────────────
SATELLITES   = ["GEO_A", "MEO_B", "MEO_C"]
ERR_COLS     = ["x_error", "y_error", "z_error", "clock_error"]
SERVER_START = datetime.now(timezone.utc).isoformat()


# ════════════════════════════════════════════════════════════
# MODEL CACHE
# Load models once at startup — never reload per request.
# WHY: joblib.load reads a .pkl file from disk.
# If we loaded on every request, each prediction would take
# 500ms+ just for disk I/O. Caching gives <5ms per prediction.
# ════════════════════════════════════════════════════════════

_model_cache: dict = {}

def load_all_models():
    """
    Load production models for all satellites into memory cache.
    Called once when the API server starts.
    """
    from mlops.stage4_retrain import load_production_model
    for sat in SATELLITES:
        try:
            _model_cache[sat] = load_production_model(sat)
            log.info("Loaded model: %s  SW_W=%.4f",
                     sat, _model_cache[sat]["sw_w_avg"])
        except FileNotFoundError:
            log.warning("No production model for %s — skipping", sat)

def get_model(satellite_id: str) -> dict:
    """Get cached model or raise 404."""
    if satellite_id not in _model_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No production model for '{satellite_id}'. "
                   f"Run Stage 4: python mlops/stage4_retrain.py --mode run"
        )
    return _model_cache[satellite_id]


# ════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════

app = FastAPI(
    title="GNSS Error Prediction API",
    description=(
        "Predicts GNSS satellite clock and ephemeris errors "
        "(X, Y, Z, Clock in meters) using Gaussian Process Regression. "
        "SIH 2025 — ISRO Problem Statement."
    ),
    version="1.0.0",
)

# Allow all origins for development
# In production: restrict to your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")  # noqa: deprecated but works fine
def startup():
    """Load all production models into cache when server starts."""
    log.info("Starting GNSS Prediction API...")
    load_all_models()
    log.info("API ready. Models loaded: %d", len(_model_cache))


# ════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS (Pydantic)
# WHY Pydantic: FastAPI uses these for automatic request
# validation + JSON serialisation + docs generation.
# If a client sends wrong types, FastAPI rejects with 422.
# ════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    satellite_id : str
    timestamps   : list[str]

    @field_validator("satellite_id")
    @classmethod
    def validate_satellite(cls, v):
        if v not in SATELLITES:
            raise ValueError(
                f"Unknown satellite '{v}'. "
                f"Valid: {SATELLITES}"
            )
        return v

    @field_validator("timestamps")
    @classmethod
    def validate_timestamps(cls, v):
        if not v:
            raise ValueError("timestamps list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 timestamps per request")
        # Verify parseability
        try:
            pd.to_datetime(v)
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {e}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "satellite_id": "MEO_B",
                "timestamps": [
                    "2025-09-08 00:11:00",
                    "2025-09-08 00:24:00",
                    "2025-09-08 00:39:00",
                ]
            }
        }
    }


class PredictionRow(BaseModel):
    utc_time    : str
    x_error     : float
    y_error     : float
    z_error     : float
    clock_error : float
    x_error_std : float
    y_error_std : float
    z_error_std : float
    clock_error_std: float


class PredictResponse(BaseModel):
    satellite_id   : str
    satellite_type : str
    n_predictions  : int
    model_sw_w_avg : float
    model_trained_at: str
    predictions    : list[PredictionRow]


class BatchRequest(BaseModel):
    requests: list[PredictRequest]

    model_config = {
        "json_schema_extra": {
            "example": {
                "requests": [
                    {
                        "satellite_id": "GEO_A",
                        "timestamps": ["2025-09-08 00:11:00"]
                    },
                    {
                        "satellite_id": "MEO_B",
                        "timestamps": ["2025-09-08 00:11:00"]
                    }
                ]
            }
        }
    }


# ════════════════════════════════════════════════════════════
# PREDICTION HELPER
# ════════════════════════════════════════════════════════════

def run_prediction(satellite_id: str,
                   timestamps: list[str]) -> pd.DataFrame:
    """
    Run GP prediction for a satellite at given timestamps.
    Uses cached model bundle — no disk I/O per request.
    """
    from mlops.stage4_retrain import predict_from_production

    # Use cached model directly instead of reloading
    bundle  = get_model(satellite_id)
    from mlops.stage4_retrain import predict, ERR_COLS as EC
    import numpy as np

    test_df = pd.DataFrame({
        "utc_time": pd.to_datetime(timestamps)
    })
    for col in EC:
        test_df[col] = 0.0

    return predict(
        bundle["models"],
        bundle["scaler"],
        bundle["t_ref"],
        test_df,
    )


# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.get("/health", tags=["Status"])
def health():
    """
    Server health check.
    Returns server status, uptime, and which models are loaded.
    Use this to verify the API is running before sending predictions.
    """
    return {
        "status"        : "ok",
        "server_start"  : SERVER_START,
        "models_loaded" : {
            sat: {
                "sw_w_avg"  : _model_cache[sat]["sw_w_avg"],
                "trained_at": _model_cache[sat]["trained_at"][:19],
            }
            for sat in _model_cache
        },
        "total_models": len(_model_cache),
    }


@app.get("/satellites", tags=["Status"])
def list_satellites():
    """
    List all satellites that have a production model loaded.
    """
    return {
        "available": [
            {
                "satellite_id"  : sat,
                "satellite_type": _model_cache[sat]["satellite_type"],
                "sw_w_avg"      : _model_cache[sat]["sw_w_avg"],
                "trained_at"    : _model_cache[sat]["trained_at"][:19],
                "train_rows"    : _model_cache[sat]["train_rows"],
            }
            for sat in _model_cache
        ]
    }


@app.get("/model/{satellite_id}", tags=["Status"])
def model_info(satellite_id: str):
    """
    Detailed info for one satellite's production model.
    Shows SW_W scores per error column, training date, row count.
    """
    if satellite_id not in SATELLITES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown satellite. Valid: {SATELLITES}"
        )
    bundle = get_model(satellite_id)
    return {
        "satellite_id"  : satellite_id,
        "satellite_type": bundle["satellite_type"],
        "trained_at"    : bundle["trained_at"][:19],
        "train_rows"    : bundle["train_rows"],
        "sw_w_avg"      : bundle["sw_w_avg"],
        "sw_w_per_column": bundle.get("sw_scores", {}),
        "t_ref"         : str(bundle["t_ref"]),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_endpoint(req: PredictRequest):
    """
    Predict GNSS errors at given timestamps for one satellite.

    **satellite_id**: GEO_A, MEO_B, or MEO_C

    **timestamps**: list of datetime strings in any standard format
    (e.g. "2025-09-08 00:11:00" or "2025-09-08T00:11:00")

    Returns predicted x_error, y_error, z_error, clock_error
    in meters, plus ±1σ uncertainty for each.
    """
    try:
        bundle  = get_model(req.satellite_id)
        pred_df = run_prediction(req.satellite_id, req.timestamps)

        rows = []
        for _, row in pred_df.iterrows():
            rows.append(PredictionRow(
                utc_time        = str(row["utc_time"]),
                x_error         = round(float(row["x_error"]),       6),
                y_error         = round(float(row["y_error"]),       6),
                z_error         = round(float(row["z_error"]),       6),
                clock_error     = round(float(row["clock_error"]),   6),
                x_error_std     = round(float(row["x_error_std"]),   6),
                y_error_std     = round(float(row["y_error_std"]),   6),
                z_error_std     = round(float(row["z_error_std"]),   6),
                clock_error_std = round(float(row["clock_error_std"]),6),
            ))

        return PredictResponse(
            satellite_id    = req.satellite_id,
            satellite_type  = bundle["satellite_type"],
            n_predictions   = len(rows),
            model_sw_w_avg  = bundle["sw_w_avg"],
            model_trained_at= bundle["trained_at"][:19],
            predictions     = rows,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(req: BatchRequest):
    """
    Predict for multiple satellites in one request.
    Each item in 'requests' is a separate satellite + timestamps pair.
    Returns a list of prediction results in the same order.
    """
    results = []
    for r in req.requests:
        try:
            result = predict_endpoint(r)
            results.append({
                "satellite_id": r.satellite_id,
                "status"      : "ok",
                "data"        : result,
            })
        except HTTPException as e:
            results.append({
                "satellite_id": r.satellite_id,
                "status"      : "error",
                "detail"      : e.detail,
            })
    return {"results": results, "total": len(results)}


@app.get("/registry", tags=["Status"])
def registry():
    """
    Show full model registry from the database.
    Lists all models (production, staging, archived) with SW scores.
    """
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute("""
        SELECT satellite_id, trained_at, train_rows,
               sw_w_avg, sw_w_x, sw_w_y, sw_w_z, sw_w_clock,
               status, n_restarts, satellite_type
        FROM model_registry
        ORDER BY trained_at DESC
    """).fetchall()
    conn.close()

    return {
        "registry": [
            {
                "satellite_id"  : r[0],
                "trained_at"    : r[1][:19],
                "train_rows"    : r[2],
                "sw_w_avg"      : r[3],
                "sw_w_x"        : r[4],
                "sw_w_y"        : r[5],
                "sw_w_z"        : r[6],
                "sw_w_clock"    : r[7],
                "status"        : r[8],
                "n_restarts"    : r[9],
                "satellite_type": r[10],
            }
            for r in rows
        ],
        "total": len(rows),
    }


@app.post("/reload", tags=["Admin"])
def reload_models():
    """
    Reload all production models from disk into the cache.
    Call this after running Stage 4 retraining while the API
    is running — no restart needed.
    """
    _model_cache.clear()
    load_all_models()
    return {
        "status"       : "reloaded",
        "models_loaded": len(_model_cache),
        "satellites"   : list(_model_cache.keys()),
    }


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "stage5_api:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,
    )