"""
GNSS MLOps - Stage 1: Data Ingestion

Pipeline:

    CSV
      |
      v
    Read raw data
      |
      v
    Normalize column names
      |
      v
    Stage 2 validation
      |
      +---------------- FAIL ----------------+
      |                                      |
      v                                      v
    Quarantine                         NO database insert
      |
      |
      +---------------- PASS ----------------+
                         |
                         v
                  Clean for insertion
                         |
                         v
                    SQLite DB
                         |
                         v
                  Ingestion log

Stage 1 responsibilities:
    - File discovery
    - File hashing
    - Idempotency
    - CSV loading
    - Column normalization
    - Satellite identification
    - Stage 2 validation integration
    - Quarantine
    - Database insertion
    - Audit logging
    - Watch mode

Stage 1 does NOT perform:
    - feature scaling
    - t_min creation
    - winsorization
    - cyclical feature engineering
    - GP preprocessing

Those belong to later stages.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


import pandas as pd


# ============================================================
# APSCHEDULER
# ============================================================

try:
    from apscheduler.schedulers.blocking import (
        BlockingScheduler,
    )
except ImportError:
    BlockingScheduler = None


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLOPS_DIR = PROJECT_ROOT / "mlops"

DATABASE_DIR = MLOPS_DIR / "database"

DATABASE_PATH = (
    DATABASE_DIR / "gnss.db"
)

INCOMING_DIR = (
    MLOPS_DIR / "incoming"
)

QUARANTINE_DIR = (
    MLOPS_DIR / "quarantine"
)

LOG_DIR = (
    MLOPS_DIR / "logs"
)

RAW_DATA_DIR = (
    PROJECT_ROOT / "Data" / "Raw"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INCOMING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

QUARANTINE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = (
    LOG_DIR / "stage1_ingestion.log"
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_FILE
        ),
    ],
)

logger = logging.getLogger(
    "GNSS_STAGE1"
)


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    """
    Return current UTC time as ISO-8601.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Create a SQLite connection.

    Foreign keys are enabled.
    """

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def create_tables(
    conn: sqlite3.Connection,
) -> None:
    """
    Create all required tables if they don't exist.

    Existing tables are NOT deleted.
    """

    # --------------------------------------------------------
    # Main GNSS observation table
    # --------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gnss_errors (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            utc_time TEXT NOT NULL,

            satellite_id TEXT NOT NULL,

            satellite_type TEXT NOT NULL,

            x_error REAL,

            y_error REAL,

            z_error REAL,

            clock_error REAL,

            source_file TEXT NOT NULL,

            ingested_at TEXT NOT NULL,

            UNIQUE (
                utc_time,
                satellite_id
            )
        )
        """
    )

    # --------------------------------------------------------
    # Processed files
    # --------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_files (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            filename TEXT NOT NULL,

            file_hash TEXT NOT NULL,

            rows_added INTEGER DEFAULT 0,

            ingested_at TEXT NOT NULL,

            UNIQUE (
                filename,
                file_hash
            )
        )
        """
    )

    # --------------------------------------------------------
    # Stage 1 ingestion audit log
    # --------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_log (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            run_at TEXT NOT NULL,

            satellite TEXT,

            rows_added INTEGER DEFAULT 0,

            rows_skipped INTEGER DEFAULT 0,

            status TEXT NOT NULL,

            message TEXT
        )
        """
    )

    # --------------------------------------------------------
    # Stage 2 validation log
    # --------------------------------------------------------

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS validation_log (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            run_at TEXT NOT NULL,

            filename TEXT NOT NULL,

            satellite TEXT,

            rows INTEGER,

            duplicates INTEGER,

            hard_failures TEXT,

            warnings TEXT,

            status TEXT NOT NULL,

            stats TEXT
        )
        """
    )

    conn.commit()


# ============================================================
# DATABASE MIGRATION
# ============================================================

def migrate_database(
    conn: sqlite3.Connection,
) -> None:
    """
    Upgrade an existing database without deleting data.

    This is necessary because your gnss.db was created by an
    earlier version of Stage 1.

    SQLite CREATE TABLE IF NOT EXISTS does NOT modify an
    existing table.

    Therefore we explicitly add missing columns.
    """

    logger.info(
        "Checking database schema..."
    )

    # ========================================================
    # ingested_files
    # ========================================================

    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(ingested_files)"
        ).fetchall()
    }

    if "rows_added" not in columns:

        logger.info(
            "Migration: adding "
            "ingested_files.rows_added"
        )

        conn.execute(
            """
            ALTER TABLE ingested_files
            ADD COLUMN rows_added INTEGER DEFAULT 0
            """
        )

    if "file_hash" not in columns:

        logger.info(
            "Migration: adding "
            "ingested_files.file_hash"
        )

        conn.execute(
            """
            ALTER TABLE ingested_files
            ADD COLUMN file_hash TEXT
            """
        )

    if "ingested_at" not in columns:

        logger.info(
            "Migration: adding "
            "ingested_files.ingested_at"
        )

        conn.execute(
            """
            ALTER TABLE ingested_files
            ADD COLUMN ingested_at TEXT
            """
        )

    # ========================================================
    # ingestion_log
    # ========================================================

    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(ingestion_log)"
        ).fetchall()
    }

    if "run_at" not in columns:

        logger.info(
            "Migration: adding "
            "ingestion_log.run_at"
        )

        conn.execute(
            """
            ALTER TABLE ingestion_log
            ADD COLUMN run_at TEXT
            """
        )

    if "satellite" not in columns:

        logger.info(
            "Migration: adding "
            "ingestion_log.satellite"
        )

        conn.execute(
            """
            ALTER TABLE ingestion_log
            ADD COLUMN satellite TEXT
            """
        )

    if "rows_added" not in columns:

        logger.info(
            "Migration: adding "
            "ingestion_log.rows_added"
        )

        conn.execute(
            """
            ALTER TABLE ingestion_log
            ADD COLUMN rows_added INTEGER DEFAULT 0
            """
        )

    if "rows_skipped" not in columns:

        logger.info(
            "Migration: adding "
            "ingestion_log.rows_skipped"
        )

        conn.execute(
            """
            ALTER TABLE ingestion_log
            ADD COLUMN rows_skipped INTEGER DEFAULT 0
            """
        )

    if "status" not in columns:

        logger.info(
            "Migration: adding "
            "ingestion_log.status"
        )

        conn.execute(
            """
            ALTER TABLE ingestion_log
            ADD COLUMN status TEXT
            """
        )

    if "message" not in columns:

        logger.info(
            "Migration: adding "
            "ingestion_log.message"
        )

        conn.execute(
            """
            ALTER TABLE ingestion_log
            ADD COLUMN message TEXT
            """
        )

    conn.commit()

    logger.info(
        "Database schema migration completed"
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database() -> None:
    """
    Create tables and migrate existing schema.
    """

    conn = get_connection()

    try:

        create_tables(conn)

        migrate_database(conn)

    finally:

        conn.close()

    logger.info(
        "Database ready: %s",
        DATABASE_PATH,
    )


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(
    path: Path,
) -> str:
    """
    Calculate MD5 hash of a file.
    """

    md5 = hashlib.md5()

    with path.open(
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):

            md5.update(
                chunk
            )

    return md5.hexdigest()


# ============================================================
# SATELLITE DETECTION
# ============================================================

def detect_satellite(
    filename: str,
) -> tuple[str, str]:
    """
    Detect satellite ID and satellite type.

    Supported:

        GEO              -> GEO_A
        MEO Train/Test   -> MEO_B
        MEO Train2/Test2 -> MEO_C

    Additional filename suffixes are supported.
    """

    name = filename.upper()

    # --------------------------------------------------------
    # GEO
    # --------------------------------------------------------

    if "GEO" in name:

        return (
            "GEO_A",
            "GEO",
        )

    # --------------------------------------------------------
    # MEO_C
    # --------------------------------------------------------

    if (
        "MEO_TRAIN2" in name
        or "MEO_TEST2" in name
        or "MEO2" in name
    ):

        return (
            "MEO_C",
            "MEO",
        )

    # --------------------------------------------------------
    # MEO_B
    # --------------------------------------------------------

    if "MEO" in name:

        return (
            "MEO_B",
            "MEO",
        )

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    return (
        "UNKNOWN",
        "UNKNOWN",
    )


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert source GNSS column names into the canonical
    names expected by Stage 2 and SQLite.

    Example:

        x_error (m)          -> x_error
        y_error  (m)         -> y_error
        z_error (m)          -> z_error
        satclockerror (m)    -> clock_error
    """

    df = df.copy()

    # --------------------------------------------------------
    # Strip whitespace
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Explicit mappings for known dataset
    # --------------------------------------------------------

    mapping = {

        "utc_time":
            "utc_time",

        "x_error (m)":
            "x_error",

        "x_error_(m)":
            "x_error",

        "y_error (m)":
            "y_error",

        "y_error  (m)":
            "y_error",

        "y_error_(m)":
            "y_error",

        "z_error (m)":
            "z_error",

        "z_error_(m)":
            "z_error",

        "satclockerror (m)":
            "clock_error",

        "satclockerror_(m)":
            "clock_error",

        "clock_error (m)":
            "clock_error",

        "clock_error_(m)":
            "clock_error",
    }

    renamed = {}

    for column in df.columns:

        if column in mapping:

            renamed[column] = mapping[
                column
            ]

            continue

        normalized = (
            column
            .lower()
            .strip()
        )

        normalized = re.sub(
            r"\s+",
            "_",
            normalized,
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        if normalized in {
            "utc_time",
            "timestamp",
            "time",
        }:

            renamed[column] = (
                "utc_time"
            )

        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        elif (
            "x_error" in normalized
            and "clock" not in normalized
        ):

            renamed[column] = (
                "x_error"
            )

        # ----------------------------------------------------
        # Y
        # ----------------------------------------------------

        elif (
            "y_error" in normalized
            and "clock" not in normalized
        ):

            renamed[column] = (
                "y_error"
            )

        # ----------------------------------------------------
        # Z
        # ----------------------------------------------------

        elif (
            "z_error" in normalized
            and "clock" not in normalized
        ):

            renamed[column] = (
                "z_error"
            )

        # ----------------------------------------------------
        # Clock
        # ----------------------------------------------------

        elif (
            "satclock" in normalized
            or "clock_error" in normalized
        ):

            renamed[column] = (
                "clock_error"
            )

        else:

            renamed[column] = (
                normalized
            )

    df = df.rename(
        columns=renamed
    )
        # --------------------------------------------------------
    # Detect duplicate canonical columns
    # --------------------------------------------------------

    duplicated_columns = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated_columns:

        raise ValueError(
            "Duplicate columns after normalization: "
            f"{duplicated_columns}"
        )

    return df


# ============================================================
# INGESTION LOG
# ============================================================
def write_ingestion_log(
    satellite: Optional[str],
    rows_added: int,
    rows_skipped: int,
    status: str,
    message: str,
    filename: Optional[str] = None,
) -> None:
    """
    Write an ingestion event to SQLite.

    Supports both the old database schema and the new
    Stage 1 schema so existing databases remain usable.
    """

    now = utc_now_iso()

    conn = get_connection()

    try:

        conn.execute(
            """
            INSERT INTO ingestion_log (
                filename,
                status,
                rows_inserted,
                message,
                timestamp,
                run_at,
                satellite,
                rows_added,
                rows_skipped
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                status,
                rows_added,
                message,
                now,
                now,
                satellite,
                rows_added,
                rows_skipped,
            ),
        )

        conn.commit()

    finally:

        conn.close()

# ============================================================
# IDEMPOTENCY CHECK
# ============================================================

def already_ingested(
    filename: str,
    file_hash: str,
) -> bool:
    """
    Return True if this exact file has already been
    successfully recorded.
    """

    conn = get_connection()

    try:

        row = conn.execute(
            """
            SELECT 1
            FROM ingested_files
            WHERE filename = ?
              AND file_hash = ?
            LIMIT 1
            """,
            (
                filename,
                file_hash,
            ),
        ).fetchone()

        return row is not None

    finally:

        conn.close()


# ============================================================
# PREPARE FOR INSERTION
# ============================================================

def prepare_for_insertion(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean a DataFrame AFTER Stage 2 has passed.

    Operations:

        - remove exact duplicates
        - remove partial-null GNSS rows
        - parse timestamps
        - remove invalid timestamps
        - sort chronologically
    """

    clean_df = df.copy()

    error_columns = [
        "x_error",
        "y_error",
        "z_error",
        "clock_error",
    ]

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    clean_df = (
        clean_df
        .drop_duplicates()
    )

    # --------------------------------------------------------
    # Remove partial-null GNSS values
    # --------------------------------------------------------

    clean_df = (
        clean_df
        .dropna(
            subset=error_columns
        )
    )

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    clean_df[
        "utc_time"
    ] = pd.to_datetime(
        clean_df[
            "utc_time"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove invalid timestamps
    # --------------------------------------------------------

    clean_df = (
        clean_df
        .dropna(
            subset=[
                "utc_time"
            ]
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    clean_df = (
        clean_df
        .sort_values(
            "utc_time"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Convert timestamps to database strings
    # --------------------------------------------------------

    clean_df[
        "utc_time"
    ] = (
        clean_df[
            "utc_time"
        ]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    )

    return clean_df


# ============================================================
# DATABASE INSERT
# ============================================================

def insert_rows(
    df: pd.DataFrame,
) -> tuple[int, int]:
    """
    Insert prepared GNSS rows into SQLite.

    Database uniqueness:

        utc_time + satellite_id

    prevents duplicate observations.
    """

    conn = get_connection()

    inserted = 0
    skipped = 0

    try:

        for row in df.itertuples(
            index=False
        ):

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO gnss_errors (
                    utc_time,
                    satellite_id,
                    satellite_type,
                    x_error,
                    y_error,
                    z_error,
                    clock_error,
                    source_file,
                    ingested_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.utc_time,
                    row.satellite_id,
                    row.satellite_type,
                    float(row.x_error),
                    float(row.y_error),
                    float(row.z_error),
                    float(row.clock_error),
                    row.source_file,
                    row.ingested_at,
                ),
            )

            if cursor.rowcount == 1:

                inserted += 1

            else:

                skipped += 1

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()

    return (
        inserted,
        skipped,
    )


# ============================================================
# RECORD INGESTED FILE
# ============================================================

def record_ingested_file(
    filename: str,
    file_hash: str,
    rows_added: int,
) -> None:
    """
    Record successful file processing.
    """

    conn = get_connection()

    try:

        conn.execute(
            """
            INSERT OR REPLACE INTO ingested_files (
                filename,
                file_hash,
                rows_added,
                ingested_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                filename,
                file_hash,
                rows_added,
                utc_now_iso(),
            ),
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# QUARANTINE
# ============================================================

def quarantine_file(
    path: Path,
) -> Path:
    """
    Move a rejected file to mlops/quarantine.
    """

    destination = (
        QUARANTINE_DIR
        / path.name
    )

    if destination.exists():

        timestamp = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        destination = (
            QUARANTINE_DIR
            / (
                f"{path.stem}_"
                f"{timestamp}"
                f"{path.suffix}"
            )
        )

    shutil.move(
        str(path),
        str(destination),
    )

    return destination


# ============================================================
# STAGE 2 VALIDATION
# ============================================================

def run_stage2_validation(
    df: pd.DataFrame,
    filename: str,
    satellite_id: str,
) -> dict:
    """
    Call the Stage 2 validation module.

    Supports both:

        python mlops/stage1_ingestion.py

    and:

        python -m mlops.stage1_ingestion
    """

    try:

        from mlops.stage2_validation import (
            validate_and_log,
        )

    except ModuleNotFoundError:

        from stage2_validation import (
            validate_and_log,
        )

    return validate_and_log(
        df=df,
        filename=filename,
        satellite_id=satellite_id,
    )


# ============================================================
# INGEST ONE FILE
# ============================================================

def ingest_file(
    path: Path,
    force: bool = False,
) -> tuple[int, int]:
    """
    Complete Stage 1 ingestion pipeline.
    """

    filename = path.name

    logger.info(
        "Processing: %s",
        filename,
    )

    # ========================================================
    # 1. Detect satellite
    # ========================================================

    satellite_id, satellite_type = (
        detect_satellite(
            filename
        )
    )

    logger.info(
        "Satellite detected: %s (%s)",
        satellite_id,
        satellite_type,
    )

    # ========================================================
    # 2. Calculate file hash
    # ========================================================

    file_hash = get_file_hash(
        path
    )

    # ========================================================
    # 3. Idempotency
    # ========================================================

    if (
        not force
        and already_ingested(
            filename,
            file_hash,
        )
    ):

        logger.info(
            "Already ingested — unchanged: %s",
            filename,
        )

        return (
            0,
            0,
        )

    # ========================================================
    # 4. Read RAW CSV
    # ========================================================

    try:

        df = pd.read_csv(
            path
        )

    except Exception as exc:

        message = (
            f"CSV read failed: {exc}"
        )

        logger.exception(
            message
        )

        write_ingestion_log(
            satellite_id,
            0,
            0,
            "FAILED",
            message,
        )

        return (
            0,
            0,
        )

    raw_rows = len(df)

    logger.info(
        "Read %d raw rows from %s",
        raw_rows,
        filename,
    )

    # ========================================================
    # 5. Normalize columns
    # ========================================================

    df = normalize_columns(
        df
    )

    logger.info(
        "Normalized columns: %s",
        list(df.columns),
    )

    # ========================================================
    # 6. STAGE 2 VALIDATION
    # ========================================================

    logger.info(
        "Running Stage 2 validation: %s",
        filename,
    )

    try:

        validation_result = (
            run_stage2_validation(
                df=df,
                filename=filename,
                satellite_id=satellite_id,
            )
        )

    except Exception as exc:

        message = (
            f"Stage 2 validation error: {exc}"
        )

        logger.exception(
            message
        )

        write_ingestion_log(
            satellite_id,
            0,
            raw_rows,
            "FAILED",
            message,
        )

        return (
            0,
            raw_rows,
        )

    # ========================================================
    # 7. HARD FAILURE
    # ========================================================

    if not validation_result.get(
        "valid",
        False,
    ):

        failures = (
            validation_result.get(
                "hard_failures",
                [],
            )
        )

        message = (
            "Stage 2 validation FAILED: "
            + " | ".join(
                failures
            )
        )

        logger.error(
            "REJECTED: %s",
            filename,
        )

        logger.error(
            "%s",
            message,
        )

        # ----------------------------------------------------
        # Log rejection
        # ----------------------------------------------------

        write_ingestion_log(
            satellite_id,
            0,
            raw_rows,
            "FAILED",
            message,
        )

        # ----------------------------------------------------
        # Quarantine
        # ----------------------------------------------------

        try:

            quarantined = (
                quarantine_file(
                    path
                )
            )

            logger.error(
                "Moved rejected file to: %s",
                quarantined,
            )

        except Exception as exc:

            logger.exception(
                "Could not quarantine %s: %s",
                filename,
                exc,
            )

        # ----------------------------------------------------
        # VERY IMPORTANT:
        #
        # insert_rows() is NOT called.
        # ----------------------------------------------------

        return (
            0,
            raw_rows,
        )

    # ========================================================
    # 8. VALIDATION PASSED
    # ========================================================

    logger.info(
        "Stage 2 validation PASSED: %s",
        filename,
    )

    # ========================================================
    # 9. Clean after validation
    # ========================================================

    clean_df = (
        prepare_for_insertion(
            df
        )
    )

    clean_rows = len(
        clean_df
    )

    logger.info(
        "Rows after Stage 2 cleaning: %d",
        clean_rows,
    )

    # ========================================================
    # 10. Add metadata
    # ========================================================

    clean_df[
        "satellite_id"
    ] = satellite_id

    clean_df[
        "satellite_type"
    ] = satellite_type

    clean_df[
        "source_file"
    ] = filename

    clean_df[
        "ingested_at"
    ] = utc_now_iso()

    # ========================================================
    # 11. Required database columns
    # ========================================================

    required_columns = [
        "utc_time",
        "satellite_id",
        "satellite_type",
        "x_error",
        "y_error",
        "z_error",
        "clock_error",
        "source_file",
        "ingested_at",
    ]

    missing = [
        column
        for column in required_columns
        if column not in clean_df.columns
    ]

    if missing:

        message = (
            "Missing insertion columns: "
            f"{missing}"
        )

        logger.error(
            message
        )

        write_ingestion_log(
            satellite_id,
            0,
            clean_rows,
            "FAILED",
            message,
        )

        return (
            0,
            clean_rows,
        )

    clean_df = clean_df[
        required_columns
    ]

    # ========================================================
    # 12. INSERT
    # ========================================================

    try:

        inserted, skipped = (
            insert_rows(
                clean_df
            )
        )

    except Exception as exc:

        message = (
            f"Database insertion failed: "
            f"{exc}"
        )

        logger.exception(
            message
        )

        write_ingestion_log(
    satellite=satellite_id,
    rows_added=0,
    rows_skipped=raw_rows,
    status="FAILED",
    message=message,
    filename=filename,
)

        return (
            0,
            clean_rows,
        )

    # ========================================================
    # 13. Record file
    # ========================================================

    try:

        record_ingested_file(
            filename=filename,
            file_hash=file_hash,
            rows_added=inserted,
        )

    except Exception as exc:

        logger.exception(
            "Could not record processed file %s: %s",
            filename,
            exc,
        )

    # ========================================================
    # 14. Audit log
    # ========================================================

    message = (
        f"Raw rows={raw_rows}; "
        f"clean rows={clean_rows}; "
        f"inserted={inserted}; "
        f"skipped={skipped}"
    )

    write_ingestion_log(
    satellite=satellite_id,
    rows_added=inserted,
    rows_skipped=skipped,
    status="SUCCESS",
    message=message,
    filename=filename,
)

    logger.info(
        "SUCCESS: %s → %d rows inserted, %d skipped",
        filename,
        inserted,
        skipped,
    )

    return (
        inserted,
        skipped,
    )


# ============================================================
# FIND CSV FILES
# ============================================================

def find_csv_files(
    directory: Path,
) -> list[Path]:
    """
    Find CSV files in a directory.
    """

    if not directory.exists():

        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() == ".csv"
                and not path.name.startswith(".")
            )
        ]
    )


# ============================================================
# SEED MODE
# ============================================================

def mode_seed() -> None:
    """
    Process all CSV files from Data/Raw/.
    """

    initialize_database()

    files = find_csv_files(
        RAW_DATA_DIR
    )

    logger.info(
        "Found %d CSV files",
        len(files),
    )

    total_inserted = 0
    total_skipped = 0

    for path in files:

        try:

            inserted, skipped = (
                ingest_file(
                    path
                )
            )

            total_inserted += (
                inserted
            )

            total_skipped += (
                skipped
            )

        except Exception:

            logger.exception(
                "Could not ingest %s",
                path.name,
            )

    logger.info(
        "Seed complete: "
        "inserted=%d, skipped=%d",
        total_inserted,
        total_skipped,
    )


# ============================================================
# WATCH SCAN
# ============================================================

def scan_incoming() -> None:
    """
    Scan mlops/incoming for CSV files.
    """

    files = find_csv_files(
        INCOMING_DIR
    )

    if not files:

        return

    logger.info(
        "Found %d incoming CSV file(s)",
        len(files),
    )

    for path in files:

        try:

            ingest_file(
                path
            )

        except Exception:

            logger.exception(
                "Could not process %s",
                path.name,
            )


# ============================================================
# WATCH MODE
# ============================================================

def mode_watch() -> None:
    """
    Monitor mlops/incoming every 60 seconds.
    """

    initialize_database()

    if BlockingScheduler is None:

        raise RuntimeError(
            "APScheduler is not installed.\n"
            "Run:\n"
            "pip install apscheduler"
        )

    logger.info(
        "Starting watch mode"
    )

    logger.info(
        "Watching: %s",
        INCOMING_DIR,
    )

    logger.info(
        "Checking every 60 seconds"
    )

    scheduler = BlockingScheduler(
        timezone="Asia/Kolkata"
    )

    scheduler.add_job(
        scan_incoming,
        trigger="interval",
        seconds=60,
        id="scan_incoming",
        replace_existing=True,

        # Prevent multiple scans from running
        # simultaneously.
        max_instances=1,

        # If the Mac sleeps or execution is delayed,
        # don't execute a pile of old scans.
        coalesce=True,

        # Allow a job to be up to 30 seconds late.
        misfire_grace_time=30,
    )

    try:

        scheduler.start()

    except (
        KeyboardInterrupt,
        SystemExit,
    ):

        logger.info(
            "Watch mode stopped"
        )

        scheduler.shutdown(
            wait=False
        )


# ============================================================
# DATABASE STATUS
# ============================================================

def mode_status() -> None:
    """
    Display database contents and ingestion history.
    """

    initialize_database()

    conn = get_connection()

    try:

        # ----------------------------------------------------
        # Total observations
        # ----------------------------------------------------

        total = conn.execute(
            """
            SELECT COUNT(*)
            FROM gnss_errors
            """
        ).fetchone()[0]

        # ----------------------------------------------------
        # Satellite rows
        # ----------------------------------------------------

        rows = conn.execute(
            """
            SELECT
                satellite_id,
                satellite_type,
                COUNT(*) AS row_count,
                MIN(utc_time) AS earliest,
                MAX(utc_time) AS latest
            FROM gnss_errors
            GROUP BY
                satellite_id,
                satellite_type
            ORDER BY satellite_id
            """
        ).fetchall()

        # ----------------------------------------------------
        # Processed files
        # ----------------------------------------------------

        processed = conn.execute(
            """
            SELECT COUNT(*)
            FROM ingested_files
            """
        ).fetchone()[0]

        # ----------------------------------------------------
        # Ingestion history
        # ----------------------------------------------------

        history = conn.execute(
            """
            SELECT
                status,
                COUNT(*)
            FROM ingestion_log
            GROUP BY status
            ORDER BY status
            """
        ).fetchall()

    finally:

        conn.close()

    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "                     DATABASE STATUS"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Total observations : {total}"
    )

    print()

    print(
        "Satellite   Type      Rows      "
        "Earliest              Latest"
    )

    print(
        "-" * 72
    )

    for (
        satellite_id,
        satellite_type,
        row_count,
        earliest,
        latest,
    ) in rows:

        print(
            f"{satellite_id:<11}"
            f"{satellite_type:<10}"
            f"{row_count:<10}"
            f"{str(earliest):<21}"
            f"{latest}"
        )

    print()

    print(
        f"Processed files : {processed}"
    )

    print()

    print(
        "Ingestion history:"
    )

    if not history:

        print(
            "  No ingestion history"
        )

    else:

        for status, count in history:

            print(
                f"  {status:<8}: {count}"
            )

    print(
        "=" * 72
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "GNSS MLOps Stage 1 - "
            "Data Ingestion"
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "seed",
            "watch",
            "status",
        ],
        help=(
            "seed   = ingest Data/Raw\n"
            "watch  = monitor mlops/incoming\n"
            "status = show database status"
        ),
    )

    args = parser.parse_args()

    if args.mode == "seed":

        mode_seed()

    elif args.mode == "watch":

        mode_watch()

    elif args.mode == "status":

        mode_status()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()