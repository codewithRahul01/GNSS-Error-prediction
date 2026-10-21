from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
  PROJECT_ROOT /"mlops"
  /"database"
  /"gnss.db"
)

LOG_DIR = PROJECT_ROOT / "mlops" / "logs"

VALIDATION_LOG_FILE = ( LOG_DIR / "validation.log" )

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
    handlers=[
        logging.FileHandler(
            VALIDATION_LOG_FILE
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(
    __name__
)

# schema 


EXPECTED_COLUMNS = [
    "utc_time",
    "x_error",
    "y_error",
    "z_error",
    "clock_error",
]


PHYSICAL_LIMITS = {
    "x_error": (-1000.0, 1000.0),
    "y_error": (-1000.0, 1000.0),
    "z_error": (-1000.0, 1000.0),
    "clock_error": (-500.0, 500.0),
}



ERROR_COLUMNS = [
    "x_error",
    "y_error",
    "z_error",
    "clock_error",
]

def check_column_count(
    df: pd.DataFrame,
    filename: str,
) -> list[str]:
    """
    Rule 1:
    The CSV must contain exactly five columns.

    Returns a list of hard-failure messages.
    """

    failures = []

    actual_count = len(df.columns)

    if actual_count != 5:

        failures.append(
            f"Expected 5 columns, got {actual_count}"
        )

    else:

        logger.info(
            "✓ Column count: 5 columns OK"
        )

    return failures


def check_column_names(
    df: pd.DataFrame,
) -> list[str]:
    """
    Rule 2:
    After standardization, the columns must be:

        utc_time
        x_error
        y_error
        z_error
        clock_error
    """

    failures = []

    actual_columns = list(
        df.columns
    )

    if actual_columns != EXPECTED_COLUMNS:

        failures.append(
            "Wrong column names. "
            f"Expected: {EXPECTED_COLUMNS}; "
            f"Actual: {actual_columns}"
        )

    else:

        logger.info(
            "✓ Column names: all correct"
        )

    return failures


def check_timestamp_parseability(
    df: pd.DataFrame,
) -> list[str]:
    """
    Rule 3:
    Every utc_time value must be parseable.
    """

    failures = []

    parsed = pd.to_datetime(
        df["utc_time"],
        errors="coerce",
    )

    invalid_count = int(
        parsed.isna().sum()
    )

    if invalid_count > 0:

        failures.append(
            f"{invalid_count} rows have "
            "unparseable timestamps"
        )

    else:

        logger.info(
            "✓ Timestamps: all parseable"
        )

    return failures

def check_all_null_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Rule 4:
    No GNSS error column may be entirely empty.
    """

    failures = []

    for column in ERROR_COLUMNS:

        if df[column].isna().all():

            failures.append(
                f"Column {column} is entirely empty"
            )

    if not failures:

        logger.info(
            "✓ No all-null columns"
        )

    return failures

def check_physical_limits(
    df: pd.DataFrame,
) -> list[str]:
    """
    Rule 5:
    Verify that GNSS errors are inside the
    physically allowed project ranges.
    """

    failures = []

    for column, (
        minimum,
        maximum,
    ) in PHYSICAL_LIMITS.items():

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_mask = (
            (values < minimum)
            | (values > maximum)
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        if invalid_count > 0:

            max_seen = values[
                invalid_mask
            ].abs().max()

            failures.append(
                f"{invalid_count} rows have "
                f"values outside physical limits "
                f"in {column} "
                f"(max abs seen: "
                f"{max_seen}m)"
            )

    if not failures:

        logger.info(
            "✓ Physical limits: all within range"
        )

    return failures

def check_minimum_rows(
    df: pd.DataFrame,
) -> list[str]:
    """
    Rule 6:
    At least five rows must remain after
    duplicate removal.
    """

    failures = []

    clean_df = df.drop_duplicates()

    row_count = len(clean_df)

    if row_count < 5:

        failures.append(
            f"Only {row_count} rows after "
            "deduplication — too few to be useful"
        )

    else:

        logger.info(
            "✓ Row count: %d rows after dedup",
            row_count,
        )

    return failures

def check_timestamp_ordering(
    df: pd.DataFrame,
) -> list[str]:
    """
    Rule 7:
    After deduplication and sorting, timestamps
    must be strictly increasing.
    """

    failures = []

    clean_df = (
        df.drop_duplicates()
        .copy()
    )

    clean_df["utc_time"] = pd.to_datetime(
        clean_df["utc_time"],
        errors="coerce",
    )

    clean_df = (
        clean_df
        .dropna(subset=["utc_time"])
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    if len(clean_df) > 1:

        differences = (
            clean_df["utc_time"].diff()
        )

        invalid_order = (
            differences <= pd.Timedelta(0)
        ).sum()

        if invalid_order > 0:

            failures.append(
                "Timestamp ordering violated: "
                f"{invalid_order} non-increasing "
                "timestamps after deduplication"
            )

    if not failures:

        logger.info(
            "✓ Timestamp ordering: strictly increasing "
            "after deduplication"
        )

    return failures

def warn_duplicates(
    df: pd.DataFrame,
) -> tuple[list[str], int]:
    """
    Warning 1:
    Detect exact duplicate rows.

    Returns:
        warnings
        duplicate_count
    """

    duplicate_count = int(
        df.duplicated(keep="first").sum()
    )

    warnings = []

    if duplicate_count > 0:

        message = (
            f"Duplicate rows found: "
            f"{duplicate_count} — "
            f"duplicates will be removed"
        )

        warnings.append(message)

        logger.warning(
            "⚠ %s",
            message,
        )

    else:

        logger.info(
            "✓ No duplicate rows"
        )

    return warnings, duplicate_count


def warn_large_gaps(
    df: pd.DataFrame,
) -> list[str]:
    """
    Warning 2:
    Detect timestamp gaps greater than 24 hours.
    """

    warnings = []

    timestamps = pd.to_datetime(
        df["utc_time"],
        errors="coerce",
    )

    timestamps = (
        timestamps
        .dropna()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(timestamps) < 2:

        return warnings

    gaps = timestamps.diff()

    large_gap_indices = (
        gaps > pd.Timedelta(hours=24)
    )

    for index in large_gap_indices[
        large_gap_indices
    ].index:

        gap = gaps.iloc[index]

        previous_time = (
            timestamps.iloc[index - 1]
        )

        current_time = (
            timestamps.iloc[index]
        )

        gap_hours = (
            gap.total_seconds() / 3600
        )

        message = (
            f"Large gap detected: "
            f"{gap_hours:.1f} hours between "
            f"{previous_time} and "
            f"{current_time}"
        )

        warnings.append(message)

        logger.warning(
            "⚠ %s",
            message,
        )

    if not warnings:

        logger.info(
            "✓ No gaps greater than 24 hours"
        )

    return warnings

def warn_outlier_rate(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, float]]:
    """
    Warning 3:
    Detect outliers using a 3×IQR fence.

    A warning is generated only when more than
    20% of values in a column are flagged.
    """

    warnings = []

    outlier_pct = {}

    for column in ERROR_COLUMNS:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if values.empty:

            outlier_pct[column] = 0.0

            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_fence = (
            q1 - 3 * iqr
        )

        upper_fence = (
            q3 + 3 * iqr
        )

        outlier_mask = (
            (values < lower_fence)
            | (values > upper_fence)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        percentage = (
            outlier_count
            / len(values)
            * 100
        )

        outlier_pct[column] = round(
            percentage,
            2,
        )

        if percentage > 20:

            message = (
                f"High outlier rate in "
                f"column {column}: "
                f"{percentage:.1f}% of values flagged"
            )

            warnings.append(message)

            logger.warning(
                "⚠ %s",
                message,
            )

    if not warnings:

        logger.info(
            "✓ Outlier rates within warning threshold"
        )

    return warnings, outlier_pct


def warn_partial_nulls(
    df: pd.DataFrame,
) -> list[str]:
    """
    Warning 4:
    Detect partially missing GNSS error values.

    1%-50% nulls → warning.

    >50% is not explicitly defined as a hard failure
    in the supplied Stage 2 specification, so we do
    not invent an additional rejection rule here.
    """

    warnings = []

    total_rows = len(df)

    if total_rows == 0:

        return warnings

    for column in ERROR_COLUMNS:

        null_count = int(
            df[column].isna().sum()
        )

        percentage = (
            null_count
            / total_rows
            * 100
        )

        if 0 < percentage <= 50:

            message = (
                f"Column {column} has "
                f"{percentage:.1f}% null values "
                "— rows with nulls will be "
                "skipped at insertion"
            )

            warnings.append(message)

            logger.warning(
                "⚠ %s",
                message,
            )

    if not warnings:

        logger.info(
            "✓ No partial-null warnings"
        )

    return warnings

def warn_unknown_satellite(
    satellite_id: str | None,
) -> list[str]:
    """
    Warning 5:
    Warn when the satellite cannot be identified.
    """

    warnings = []

    if (
        satellite_id is None
        or satellite_id.upper() == "UNKNOWN"
    ):

        message = (
            "Could not detect satellite type "
            "from filename — defaulting to UNKNOWN"
        )

        warnings.append(message)

        logger.warning(
            "⚠ %s",
            message,
        )

    else:

        logger.info(
            "✓ Satellite detected: %s",
            satellite_id,
        )

    return warnings


def compute_stats(
    df: pd.DataFrame,
) -> dict:
    """
    Compute descriptive statistics for the
    validation report.

    This function does not determine PASS/FAIL.
    """

    stats = {}

    # --------------------------------------------------------
    # Basic row statistics
    # --------------------------------------------------------

    stats["rows"] = int(len(df))

    stats["duplicates"] = int(
        df.duplicated(keep="first").sum()
    )

    # --------------------------------------------------------
    # Timestamp statistics
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        df["utc_time"],
        errors="coerce",
    ).dropna()

    if len(timestamps) > 0:

        timestamps = (
            timestamps
            .sort_values()
            .reset_index(drop=True)
        )

        stats["start_time"] = (
            timestamps.iloc[0].isoformat()
        )

        stats["end_time"] = (
            timestamps.iloc[-1].isoformat()
        )

        stats["time_span_hours"] = round(
            (
                timestamps.iloc[-1]
                - timestamps.iloc[0]
            ).total_seconds()
            / 3600,
            2,
        )

        if len(timestamps) > 1:

            gaps = (
                timestamps.diff()
                .dropna()
            )

            stats["median_interval_minutes"] = round(
                gaps.dt.total_seconds()
                .median()
                / 60,
                2,
            )

            stats["max_gap_hours"] = round(
                gaps.dt.total_seconds()
                .max()
                / 3600,
                2,
            )

        else:

            stats["median_interval_minutes"] = None
            stats["max_gap_hours"] = None

    else:

        stats["start_time"] = None
        stats["end_time"] = None
        stats["time_span_hours"] = None
        stats["median_interval_minutes"] = None
        stats["max_gap_hours"] = None

    # --------------------------------------------------------
    # Null statistics
    # --------------------------------------------------------

    null_rates = {}

    for column in ERROR_COLUMNS:

        null_rate = (
            df[column].isna().mean()
            * 100
        )

        null_rates[column] = round(
            null_rate,
            2,
        )

    stats["null_rate_pct"] = null_rates

    # --------------------------------------------------------
    # Outlier statistics
    # --------------------------------------------------------

    outlier_rates = {}

    for column in ERROR_COLUMNS:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if values.empty:

            outlier_rates[column] = 0.0
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr

        outliers = (
            (values < lower)
            | (values > upper)
        )

        rate = (
            outliers.sum()
            / len(values)
            * 100
        )

        outlier_rates[column] = round(
            rate,
            2,
        )

    stats["outlier_rate_pct"] = (
        outlier_rates
    )

    return stats



def validate_file(
    df: pd.DataFrame,
    filename: str,
    satellite_id: str | None = None,
) -> dict:
    """
    Run all Stage 2 hard checks and soft warnings.

    Hard failures reject the file.

    Warnings are recorded but do not reject it.
    """

    hard_failures = []
    warnings = []

    # ========================================================
    # HARD CHECKS
    # ========================================================

    # Rule 1
    hard_failures.extend(
        check_column_count(
            df,
            filename,
        )
    )

    # Rule 2
    #
    # If column count is wrong, comparing column names
    # can produce misleading results, but we still perform
    # the check safely.
    hard_failures.extend(
        check_column_names(df)
    )

    # Rule 3
    #
    # Only run if utc_time exists.
    if "utc_time" in df.columns:

        hard_failures.extend(
            check_timestamp_parseability(df)
        )

    else:

        hard_failures.append(
            "Missing utc_time column"
        )

    # The remaining checks require the expected columns.
    # Don't crash if the schema is broken.
    if all(
        column in df.columns
        for column in ERROR_COLUMNS
    ):

        # Rule 4
        hard_failures.extend(
            check_all_null_columns(df)
        )

        # Rule 5
        hard_failures.extend(
            check_physical_limits(df)
        )

    # Rule 6
    hard_failures.extend(
        check_minimum_rows(df)
    )

    # Rule 7
    if "utc_time" in df.columns:

        hard_failures.extend(
            check_timestamp_ordering(df)
        )

    # ========================================================
    # SOFT WARNINGS
    # ========================================================

    # Warning 1
    duplicate_warnings, _ = (
        warn_duplicates(df)
    )

    warnings.extend(
        duplicate_warnings
    )

    # Warning 2
    if "utc_time" in df.columns:

        warnings.extend(
            warn_large_gaps(df)
        )

    # Warning 3
    if all(
        column in df.columns
        for column in ERROR_COLUMNS
    ):

        outlier_warnings, _ = (
            warn_outlier_rate(df)
        )

        warnings.extend(
            outlier_warnings
        )

    # Warning 4
    if all(
        column in df.columns
        for column in ERROR_COLUMNS
    ):

        warnings.extend(
            warn_partial_nulls(df)
        )

    # Warning 5
    warnings.extend(
        warn_unknown_satellite(
            satellite_id
        )
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    stats = compute_stats(df)

    # ========================================================
    # FINAL DECISION
    # ========================================================

    valid = len(hard_failures) == 0

    return {
        "valid": valid,
        "filename": filename,
        "satellite": satellite_id,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "stats": stats,
    }



def validate_and_log(
    df: pd.DataFrame,
    filename: str,
    satellite_id: str | None = None,
) -> dict:
    """
    Validate a DataFrame and write the result
    to the validation_log table.

    Returns the validation result.
    """

    result = validate_file(
        df=df,
        filename=filename,
        satellite_id=satellite_id,
    )

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    # --------------------------------------------------------
    # Create validation_log table
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
        );
        """
    )

    stats = result["stats"]

    status = (
        "PASS"
        if result["valid"]
        else "FAIL"
    )

    conn.execute(
        """
        INSERT INTO validation_log (

            run_at,
            filename,
            satellite,
            rows,
            duplicates,
            hard_failures,
            warnings,
            status,
            stats

        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(
                timezone.utc
            ).isoformat(),

            filename,

            satellite_id,

            stats.get(
                "rows",
                0,
            ),

            stats.get(
                "duplicates",
                0,
            ),

            json.dumps(
                result["hard_failures"]
            ),

            json.dumps(
                result["warnings"]
            ),

            status,

            json.dumps(
                stats
            ),
        ),
    )

    conn.commit()
    conn.close()

    return result

def save_validation_report(
    result: dict,
) -> Path:
    """
    Save one validation result as a CSV report.
    """

    report_path = (
        LOG_DIR
        / "validation_report.csv"
    )

    stats = result["stats"]

    row = {
        "run_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "filename": result[
            "filename"
        ],

        "satellite": result[
            "satellite"
        ],

        "status": (
            "PASS"
            if result["valid"]
            else "FAIL"
        ),

        "hard_failures": " | ".join(
            result["hard_failures"]
        ),

        "warnings": " | ".join(
            result["warnings"]
        ),

        "rows": stats.get(
            "rows"
        ),

        "duplicates": stats.get(
            "duplicates"
        ),

        "time_span_hours": stats.get(
            "time_span_hours"
        ),

        "median_interval_minutes": stats.get(
            "median_interval_minutes"
        ),

        "max_gap_hours": stats.get(
            "max_gap_hours"
        ),

        "null_rate_pct": json.dumps(
            stats.get(
                "null_rate_pct",
                {},
            )
        ),

        "outlier_rate_pct": json.dumps(
            stats.get(
                "outlier_rate_pct",
                {},
            )
        ),
    }

    new_row = pd.DataFrame(
        [row]
    )

    if report_path.exists():

        existing = pd.read_csv(
            report_path
        )

        combined = pd.concat(
            [
                existing,
                new_row,
            ],
            ignore_index=True,
        )

    else:

        combined = new_row

    combined.to_csv(
        report_path,
        index=False,
    )

    logger.info(
        "Validation report saved: %s",
        report_path,
    )

    return report_path


def mode_validate(
    file_path: str,
) -> None:
    """
    Standalone validation mode.

    This validates a CSV but does NOT insert it
    into gnss_errors.
    """

    path = Path(file_path)

    if not path.exists():

        print(
            f"ERROR: File does not exist: {path}"
        )

        return

    if path.suffix.lower() != ".csv":

        print(
            "ERROR: Only CSV files are supported."
        )

        return

    print()
    print("=" * 72)
    print("              GNSS STAGE 2 DATA VALIDATION")
    print("=" * 72)
    print(
        f"File: {path.name}"
    )

    try:

        df = pd.read_csv(path)

        # ----------------------------------------------------
        # Normalize columns exactly as Stage 1 does
        # ----------------------------------------------------

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.replace(
                r"\s+",
                "_",
                regex=True,
            )
            .str.lower()
        )

        column_mapping = {
            "x_error_(m)": "x_error",
            "y_error_(m)": "y_error",
            "z_error_(m)": "z_error",
            "satclockerror_(m)": "clock_error",
        }

        df = df.rename(
            columns=column_mapping
        )

        # ----------------------------------------------------
        # Satellite detection
        # ----------------------------------------------------

        satellite_id = None

        filename = path.name

        if "DATA_GEO_" in filename:

            satellite_id = "GEO_A"

        elif "DATA_MEO_Train2" in filename:

            satellite_id = "MEO_C"

        elif "DATA_MEO_Test2" in filename:

            satellite_id = "MEO_C"

        elif "DATA_MEO_" in filename:

            satellite_id = "MEO_B"

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        result = validate_and_log(
            df=df,
            filename=filename,
            satellite_id=satellite_id,
        )

        # ----------------------------------------------------
        # Save report
        # ----------------------------------------------------

        report_path = (
            save_validation_report(
                result
            )
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print()

        print(
            "RESULT:",
            "PASS"
            if result["valid"]
            else "FAIL",
        )

        print(
            f"Rows: "
            f"{result['stats']['rows']}"
        )

        print(
            f"Duplicates: "
            f"{result['stats']['duplicates']}"
        )

        # ----------------------------------------------------
        # Hard failures
        # ----------------------------------------------------

        if result["hard_failures"]:

            print()
            print(
                "HARD FAILURES:"
            )

            for failure in (
                result["hard_failures"]
            ):

                print(
                    f"  ❌ {failure}"
                )

        # ----------------------------------------------------
        # Warnings
        # ----------------------------------------------------

        if result["warnings"]:

            print()
            print(
                "WARNINGS:"
            )

            for warning in (
                result["warnings"]
            ):

                print(
                    f"  ⚠ {warning}"
                )

        print()
        print(
            f"Report: {report_path}"
        )

        print("=" * 72)
        print()

    except Exception as exc:

        logger.exception(
            "Validation failed: %s",
            exc,
        )

        print(
            f"ERROR: {exc}"
        )



def prepare_for_insertion(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare a validated DataFrame for database insertion.

    This happens ONLY after validation passes.

    Actions:
    1. Remove exact duplicate rows.
    2. Remove rows containing partial nulls.
    3. Sort chronologically.
    """

    clean_df = df.copy()

    # Remove exact duplicates
    clean_df = clean_df.drop_duplicates()

    # Remove rows containing null GNSS values
    clean_df = clean_df.dropna(
        subset=ERROR_COLUMNS
    )

    # Remove invalid timestamps
    clean_df = clean_df.dropna(
        subset=["utc_time"]
    )

    # Convert timestamp
    clean_df["utc_time"] = pd.to_datetime(
        clean_df["utc_time"],
        errors="coerce",
    )

    # Sort
    clean_df = (
        clean_df
        .sort_values("utc_time")
        .reset_index(drop=True)
    )

    return clean_df



def main():

    parser = argparse.ArgumentParser(
        description=(
            "GNSS MLOps Stage 2 - "
            "Data Validation"
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "validate",
        ],
    )

    parser.add_argument(
        "--file",
        required=True,
        help="CSV file to validate",
    )

    args = parser.parse_args()

    if args.mode == "validate":

        mode_validate(
            args.file
        )


if __name__ == "__main__":
    main()