"""
=============================================================================
  SIH GNSS ERROR PREDICTION PROJECT
  PHASE 3 — Baseline Models
=============================================================================

  WHAT THIS PHASE DOES
  ─────────────────────
  Builds 3 simple baseline models BEFORE touching the GP model.

  WHY baselines matter:
    Before building any ML model you MUST know what a dumb model scores.
    If your GP model doesn't clearly beat these baselines, then either:
      a) The GP is not working correctly, or
      b) The data is genuinely unpredictable (no model will help)
    The baselines set the floor — your GP must score above them.

  3 Baselines implemented:
    Baseline 1 — Persistence
                 "Predict the last observed training value for every
                  test timestamp."
                 This is the simplest possible model. Zero computation.
                 Represents: "the error doesn't change at all."

    Baseline 2 — Linear Extrapolation (last N points)
                 "Fit a straight line on the last 10 training points
                  and extend it into the test day."
                 This mirrors the GNSS af0 + af1×t clock model.
                 Represents: "error changes at a constant rate."

    Baseline 3 — Training Mean
                 "Predict the average error from training for every
                  test timestamp."
                 Represents: "the test day looks like an average
                  training day."

  Evaluation metric: Shapiro-Wilk W on residuals (actual − predicted)
    - Higher W = better (residuals are more Gaussian)
    - Target: W ≥ 0.9810 (the benchmark from the problem statement)
    - Also reports: RMSE, MAE, mean residual, std residual

  HOW TO RUN
  ──────────
    python src/phase3_baselines.py

  INPUT  (reads from Data/Processed/)
  ─────
    geo_train_featured.csv    ← unscaled data with t_min feature
    geo_test_featured.csv
    meo1_train_featured.csv
    meo1_test_featured.csv
    meo2_train_featured.csv
    meo2_test_featured.csv

  OUTPUT (saves to results/)
  ──────
    results/baseline_scores.csv      ← SW scores for all 3 baselines
    results/baseline_predictions/    ← prediction CSVs per satellite

  PRE-REQUISITES
  ──────────────
    Phase 1 and Phase 2 must be complete
    pip install pandas numpy scipy tabulate matplotlib
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# PATHS
# =============================================================================

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

for _d in ["Data", "data"]:
    for _p in ["Processed", "processed"]:
        _c = os.path.join(BASE_DIR, _d, _p)
        if os.path.isdir(_c):
            PROC_DIR = _c
            break

RES_DIR  = os.path.join(BASE_DIR, "results")
PRED_DIR = os.path.join(RES_DIR,  "baseline_predictions")
FIG_DIR  = os.path.join(BASE_DIR, "figures")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================

ERR_COLS = ["x_error", "y_error", "z_error", "clock_error"]
SW_BENCHMARK_W = 0.9810
SW_BENCHMARK_P = 0.5840

# How many trailing training points to use for linear extrapolation
LINEAR_FIT_POINTS = 10


# =============================================================================
# DATA LOADING
# =============================================================================

def load(filename: str) -> pd.DataFrame:
    """
    Load a featured (unscaled) CSV from Data/Processed/.

    WHY we use *_featured.csv and NOT *_ready.csv:
      *_ready.csv has SCALED error values (mean=0, std=1).
      Baselines predict in original meter units so we can directly
      compare residuals to real test values in meters.
      The SW test must run on meter-unit residuals.

    Parameters
    ----------
    filename : e.g. "geo_train_featured.csv"

    Returns
    -------
    pd.DataFrame sorted by utc_time with t_min column
    """
    path = os.path.join(PROC_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  ✗  {path}\n"
            f"     Run phase2_preprocessing.py first."
        )
    df = pd.read_csv(path)
    df["utc_time"] = pd.to_datetime(df["utc_time"])
    return df.sort_values("utc_time").reset_index(drop=True)


# =============================================================================
# BASELINE 1 — PERSISTENCE
# =============================================================================

def baseline_persistence(train_df: pd.DataFrame,
                         test_df:  pd.DataFrame) -> pd.DataFrame:
    """
    Predict the LAST observed training value for every test timestamp.

    Formula:  prediction(t) = train_data[-1]  for all t in test

    Example with GEO:
      Last training row (Sep 7, 23:41):
        x_error = 2.508   y_error = 2.415   z_error = 0.520   clock = 0.043
      Prediction for EVERY test row (Sep 8, 00:11 to Sep 8, 23:56):
        x_error = 2.508   y_error = 2.415   z_error = 0.520   clock = 0.043

    WHY this is useful as a baseline:
      If your GP model cannot beat even this, the model is broken.
      In practice, persistence is hard to beat at very short horizons
      (15-30 min) because errors don't change much in 15 minutes.
      At longer horizons (hours) the error drifts and persistence
      becomes increasingly wrong — that's where ML should help.

    Parameters
    ----------
    train_df : training dataframe with ERR_COLS
    test_df  : test dataframe with ERR_COLS

    Returns
    -------
    pd.DataFrame with utc_time + predicted ERR_COLS
    """
    last_obs  = train_df[ERR_COLS].iloc[-1].to_dict()
    n_test    = len(test_df)
    pred_vals = {col: [last_obs[col]] * n_test for col in ERR_COLS}
    preds     = pd.DataFrame(pred_vals)
    preds.insert(0, "utc_time", test_df["utc_time"].values)
    return preds


# =============================================================================
# BASELINE 2 — LINEAR EXTRAPOLATION
# =============================================================================

def baseline_linear(train_df:  pd.DataFrame,
                    test_df:   pd.DataFrame,
                    n_fit_pts: int = LINEAR_FIT_POINTS) -> pd.DataFrame:
    """
    Fit a 1st-order polynomial on the last N training points,
    extrapolate to test timestamps.

    Formula:  prediction(t) = a0 + a1 × t
      where a0 (bias) and a1 (drift) are fitted by least squares
      on the last n_fit_pts rows of training data.

    WHY this matters for GNSS:
      This is EXACTLY the same model the satellite itself uses for
      clock corrections: error = af0 + af1 × (t - toc)
      (bias + drift × time since reference epoch)

      For clock error specifically, this is the natural physics-based
      model. For orbit errors, it's a first approximation.

    WHY last N points (not all training data):
      Using all 121 GEO training rows would average across the entire
      7-day period and miss the recent trend. The last 10 points
      (≈2.5 hours in 15-min mode) capture the current drift rate
      which is most relevant for extrapolating into the test day.

      If n_fit_pts is too small (< 3), the fit is noisy.
      If too large, it averages away the recent trend.
      10 points is a good default for 15-min sampled data.

    Parameters
    ----------
    train_df  : full training dataframe
    test_df   : test dataframe
    n_fit_pts : how many trailing training points to use

    Returns
    -------
    pd.DataFrame with utc_time + predicted ERR_COLS
    """
    tail     = train_df.tail(n_fit_pts)
    t_train  = tail["t_min"].values
    t_test   = test_df["t_min"].values

    preds_dict = {"utc_time": test_df["utc_time"].values}

    for col in ERR_COLS:
        y_train = tail[col].values

        # np.polyfit(x, y, deg=1) returns [slope, intercept]
        coeffs = np.polyfit(t_train, y_train, 1)

        # np.polyval evaluates the polynomial at new x values
        y_pred = np.polyval(coeffs, t_test)

        preds_dict[col] = y_pred

    return pd.DataFrame(preds_dict)


# =============================================================================
# BASELINE 3 — TRAINING MEAN
# =============================================================================

def baseline_mean(train_df: pd.DataFrame,
                  test_df:  pd.DataFrame) -> pd.DataFrame:
    """
    Predict the training set mean for every test timestamp.

    Formula:  prediction(t) = mean(train_errors)  for all t in test

    WHY this is useful:
      If errors are roughly stationary (no trend), the mean is the
      optimal constant predictor in the MSE sense.
      Comparing persistence vs mean tells you whether the error
      has a trend at the end of training:
        persistence ≈ mean  →  no recent trend
        persistence ≠ mean  →  recent trend exists (linear extrapolation
                               should help)

    Parameters
    ----------
    train_df : training dataframe
    test_df  : test dataframe

    Returns
    -------
    pd.DataFrame with utc_time + predicted ERR_COLS
    """
    means     = train_df[ERR_COLS].mean().to_dict()
    n_test    = len(test_df)
    pred_vals = {col: [means[col]] * n_test for col in ERR_COLS}
    preds     = pd.DataFrame(pred_vals)
    preds.insert(0, "utc_time", test_df["utc_time"].values)
    return preds


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(actual_df:  pd.DataFrame,
             pred_df:    pd.DataFrame,
             label:      str = "",
             sat_name:   str = "") -> pd.DataFrame:
    """
    Compute metrics on residuals = actual - predicted.

    Metrics computed per error column:
      RMSE      : root mean squared error (sensitive to large spikes)
      MAE       : mean absolute error (more robust)
      Res mean  : mean of residuals (should be near 0 if unbiased)
      Res std   : standard deviation of residuals
      SW_W      : Shapiro-Wilk statistic  ← PRIMARY metric
      SW_p      : p-value of SW test
      H0_reject : 1 if p<0.05 (non-normal), 0 if p≥0.05 (normal)

    The AVERAGED row at the bottom is what you report to the evaluator.

    Parameters
    ----------
    actual_df : test dataframe with true ERR_COLS values
    pred_df   : predictions dataframe with ERR_COLS
    label     : name of this baseline for display
    sat_name  : satellite name for display

    Returns
    -------
    pd.DataFrame with one row per ERR_COL + one AVERAGED row
    """
    rows = []
    for col in ERR_COLS:
        actual   = actual_df[col].values
        pred     = pred_df[col].values
        residual = actual - pred

        rmse     = np.sqrt(np.mean(residual ** 2))
        mae      = np.mean(np.abs(residual))
        res_mean = residual.mean()
        res_std  = residual.std()
        w, p     = stats.shapiro(residual)
        rejected = int(p < 0.05)

        rows.append({
            "satellite"  : sat_name,
            "baseline"   : label,
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

    # Average row
    avg = {
        "satellite"  : sat_name,
        "baseline"   : label,
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
    """
    Pretty-print evaluation results for one baseline.
    """
    print(f"\n  ── {label} ──")
    rows = []
    for _, row in df_res.iterrows():
        sym = "✓" if row["h0_rejected"] == 0 else "✗"
        vs_bench = ""
        if row["column"] == "AVERAGED":
            diff = row["sw_w"] - SW_BENCHMARK_W
            vs_bench = f"  ({'↑' if diff >= 0 else '↓'}{abs(diff):.4f} vs benchmark)"
        rows.append([
            f"{sym} {row['column']}",
            f"{row['rmse']:.4f}",
            f"{row['mae']:.4f}",
            f"{row['res_mean']:+.4f}",
            f"{row['res_std']:.4f}",
            f"{row['sw_w']:.4f}{vs_bench}",
            f"{row['sw_p']:.4f}",
        ])
    hdrs = ["Column", "RMSE", "MAE", "Res mean", "Res std", "SW_W", "SW_p"]
    print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))


# =============================================================================
# VISUALISATION — time series of predictions vs actuals
# =============================================================================

def plot_baseline_comparison(train_df:   pd.DataFrame,
                             test_df:    pd.DataFrame,
                             preds_dict: dict,
                             sat_name:   str):
    """
    Plot actual test values + all 3 baseline predictions for each
    error column. Shows visually how far off each baseline is.

    Parameters
    ----------
    train_df   : unscaled training dataframe
    test_df    : unscaled test dataframe
    preds_dict : {"Persistence": df, "Linear": df, "Mean": df}
    sat_name   : 'GEO', 'MEO1', 'MEO2'
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
    fig.suptitle(f"Phase 3 Baselines vs Actual — {sat_name}",
                 fontsize=14, fontweight="bold", y=1.01)

    colors = {
        "Persistence"   : "#E06C75",
        "Linear Extrap" : "#61AFEF",
        "Training Mean" : "#98C379",
    }

    for idx, col in enumerate(ERR_COLS):
        ax = axes[idx]

        # Training data (context)
        ax.scatter(train_df["utc_time"], train_df[col],
                   color="#888", s=6, alpha=0.4, label="Train (actual)",
                   zorder=2)

        # Test actuals
        ax.scatter(test_df["utc_time"], test_df[col],
                   color="black", s=18, zorder=5,
                   label="Test (actual)", marker="o")

        # Baseline predictions
        for bname, bpreds in preds_dict.items():
            ax.plot(test_df["utc_time"], bpreds[col],
                    color=colors[bname], linewidth=1.5,
                    linestyle="--", label=f"{bname}", zorder=4, alpha=0.85)

        # Vertical line separating train from test
        if len(train_df) > 0:
            ax.axvline(train_df["utc_time"].max(),
                       color="gray", linewidth=1, linestyle=":",
                       alpha=0.6, label="Train/test split")

        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_ylabel("Error (m)", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=25)

        if idx == 0:
            ax.legend(fontsize=7, ncol=5, loc="upper right")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"phase3_baselines_{sat_name.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: figures/phase3_baselines_{sat_name.lower()}.png")
    return out


def plot_residual_distributions(test_df:    pd.DataFrame,
                                preds_dict: dict,
                                sat_name:   str):
    """
    Plot residual histograms with normal overlay for each baseline.
    This is the visual equivalent of the Shapiro-Wilk test.
    A residual distribution that matches the normal curve → good SW score.
    """
    n_baselines = len(preds_dict)
    fig, axes = plt.subplots(4, n_baselines,
                              figsize=(5 * n_baselines, 14))
    fig.suptitle(
        f"Phase 3 Residual Distributions — {sat_name}\n"
        f"(Closer to normal bell curve = better SW_W score)",
        fontsize=12, fontweight="bold")

    b_colors = {
        "Persistence"   : "#E06C75",
        "Linear Extrap" : "#61AFEF",
        "Training Mean" : "#98C379",
    }

    for col_idx, col in enumerate(ERR_COLS):
        for b_idx, (bname, bpreds) in enumerate(preds_dict.items()):
            ax       = axes[col_idx][b_idx]
            residual = test_df[col].values - bpreds[col].values
            w, p     = stats.shapiro(residual)

            # Histogram of residuals
            ax.hist(residual, bins=15, density=True,
                    color=b_colors[bname], alpha=0.6,
                    edgecolor="white")

            # Normal distribution overlay
            x_range = np.linspace(residual.min(), residual.max(), 200)
            normal_curve = stats.norm.pdf(x_range, residual.mean(),
                                           residual.std())
            ax.plot(x_range, normal_curve, color="black",
                    linewidth=1.5, linestyle="--", label="Normal")

            status = "✓ Normal" if p >= 0.05 else "✗ Not normal"
            ax.set_title(f"{bname}\n{col}\nW={w:.4f}  {status}",
                         fontsize=8, fontweight="bold")
            ax.set_xlabel("Residual (m)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(FIG_DIR,
                       f"phase3_residuals_{sat_name.lower()}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Saved: figures/phase3_residuals_{sat_name.lower()}.png")
    return out


# =============================================================================
# FULL PIPELINE — one satellite
# =============================================================================

def run_baselines_for_satellite(train_file: str,
                                 test_file:  str,
                                 sat_name:   str) -> pd.DataFrame:
    """
    Run all 3 baselines for one satellite. Print results. Save outputs.

    Parameters
    ----------
    train_file : e.g. "geo_train_featured.csv"
    test_file  : e.g. "geo_test_featured.csv"
    sat_name   : 'GEO', 'MEO1', 'MEO2'

    Returns
    -------
    pd.DataFrame with all evaluation rows for this satellite
    """
    sep = "─" * 60
    print(f"\n  {sep}")
    print(f"  Satellite: {sat_name}")
    print(f"  {sep}")

    # Load unscaled data
    train_df = load(train_file)
    test_df  = load(test_file)

    print(f"  Train: {len(train_df)} rows  |  "
          f"Test: {len(test_df)} rows")
    print(f"  Train t_min: {train_df['t_min'].min():.0f} → "
          f"{train_df['t_min'].max():.0f} min")
    print(f"  Test  t_min: {test_df['t_min'].min():.0f} → "
          f"{test_df['t_min'].max():.0f} min")

    # ── Run all 3 baselines ───────────────────────────────────────────
    preds_b1 = baseline_persistence(train_df, test_df)
    preds_b2 = baseline_linear(train_df, test_df, LINEAR_FIT_POINTS)
    preds_b3 = baseline_mean(train_df, test_df)

    preds_dict = {
        "Persistence"     : preds_b1,
        "Linear Extrap"  : preds_b2,
        "Training Mean"  : preds_b3,
    }

    # ── Evaluate each ─────────────────────────────────────────────────
    print(f"\n  SW Benchmark: W={SW_BENCHMARK_W}  p={SW_BENCHMARK_P}")
    all_eval_rows = []
    for bname, bpreds in preds_dict.items():
        df_eval = evaluate(test_df, bpreds,
                           label=bname, sat_name=sat_name)
        print_eval(df_eval, bname)
        all_eval_rows.append(df_eval)

        # Save predictions
        out = os.path.join(PRED_DIR,
                           f"{sat_name.lower()}_{bname.lower().replace(' ','_')}.csv")
        bpreds.to_csv(out, index=False)

    # ── Plots ─────────────────────────────────────────────────────────
    print()
    plot_baseline_comparison(train_df, test_df,
                              preds_dict, sat_name)
    plot_residual_distributions(test_df, preds_dict, sat_name)

    return pd.concat(all_eval_rows, ignore_index=True)


# =============================================================================
# SUMMARY — compare all baselines across all satellites
# =============================================================================

def print_summary(all_scores: pd.DataFrame):
    """
    Print a master comparison table showing SW_W for every baseline
    and every satellite, averaged across the 4 error columns.
    This is the table you use to decide which baseline to beat
    and how much room the GP model has to improve.
    """
    print("\n" + "═" * 65)
    print("  PHASE 3 SUMMARY — Baseline SW_W scores (averaged over 4 cols)")
    print("  Higher W = more Gaussian residuals = better model")
    print(f"  Benchmark target: W = {SW_BENCHMARK_W}")
    print("═" * 65)

    avg_rows = all_scores[all_scores["column"] == "AVERAGED"]
    pivot = avg_rows.pivot_table(
        index="satellite",
        columns="baseline",
        values="sw_w",
        aggfunc="first"
    )
    print()
    print(tabulate(pivot.round(4),
                   headers="keys", tablefmt="rounded_outline"))

    print()
    print("  RMSE comparison (meters):")
    pivot_rmse = avg_rows.pivot_table(
        index="satellite",
        columns="baseline",
        values="rmse",
        aggfunc="first"
    )
    print(tabulate(pivot_rmse.round(4),
                   headers="keys", tablefmt="rounded_outline"))

    # Which baseline is best per satellite
    print()
    print("  Best baseline per satellite (by SW_W):")
    for sat in avg_rows["satellite"].unique():
        sat_rows = avg_rows[avg_rows["satellite"] == sat]
        best_idx  = sat_rows["sw_w"].idxmax()
        best_row  = sat_rows.loc[best_idx]
        print(f"    {sat:<6}: {best_row['baseline']:<20} "
              f"W={best_row['sw_w']:.4f}  RMSE={best_row['rmse']:.4f}m")

    print(f"""
  HOW TO READ THIS TABLE
  ──────────────────────────────────────────────────────────
  • SW_W < 0.9810  →  baseline does NOT meet benchmark
  • SW_W ≥ 0.9810  →  baseline meets benchmark (GP must beat this)
  • If ALL baselines fail badly (W << 0.98), the test data has
    large unpredictable spikes that no model captures well.
    This is expected for GEO (upload boundary spikes ±75m).

  WHAT THE GP MODEL (Phase 4) NEEDS TO DO
  ──────────────────────────────────────────────────────────
  • Beat the BEST baseline SW_W score for each satellite
  • Ideally reach W ≥ 0.9810 on averaged residuals
  • The GP models periodic patterns the linear baseline misses
    (daily 24h cycle for GEO, 12h cycle for MEO)
  • Phase 6 (improvements) will refine the GEO spike problem
  ──────────────────────────────────────────────────────────""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "═" * 65)
    print("  SIH GNSS Error Prediction Project")
    print("  PHASE 3 — Baseline Models")
    print("═" * 65)
    print("""
  Building 3 baselines:
    1. Persistence      — last training value repeated
    2. Linear Extrap    — a0 + a1×t fitted on last 10 train points
    3. Training Mean    — average of all training values
    """)

    all_scores = []

    # ── Satellite A: GEO ─────────────────────────────────────────────
    scores = run_baselines_for_satellite(
        train_file = "geo_train_featured.csv",
        test_file  = "geo_test_featured.csv",
        sat_name   = "GEO",
    )
    all_scores.append(scores)

    # ── Satellite B: MEO1 ────────────────────────────────────────────
    scores = run_baselines_for_satellite(
        train_file = "meo1_train_featured.csv",
        test_file  = "meo1_test_featured.csv",
        sat_name   = "MEO1",
    )
    all_scores.append(scores)

    # ── Satellite C: MEO2 ────────────────────────────────────────────
    scores = run_baselines_for_satellite(
        train_file = "meo2_train_featured.csv",
        test_file  = "meo2_test_featured.csv",
        sat_name   = "MEO2",
    )
    all_scores.append(scores)

    # ── Combined scores ───────────────────────────────────────────────
    all_df = pd.concat(all_scores, ignore_index=True)

    # Save all scores to CSV
    out = os.path.join(RES_DIR, "baseline_scores.csv")
    all_df.to_csv(out, index=False)
    print(f"\n  ✓  All scores saved: results/baseline_scores.csv")

    # Summary
    print_summary(all_df)

    # Done
    print("\n" + "═" * 65)
    print("  PHASE 3 COMPLETE")
    print("═" * 65)
    print("""
  Files saved:
    results/baseline_scores.csv
    results/baseline_predictions/
      geo_persistence.csv   geo_linear_extrap.csv   geo_training_mean.csv
      meo1_...              meo1_...                meo1_...
      meo2_...              meo2_...                meo2_...
    figures/
      phase3_baselines_geo.png    phase3_residuals_geo.png
      phase3_baselines_meo1.png   phase3_residuals_meo1.png
      phase3_baselines_meo2.png   phase3_residuals_meo2.png

  Next step → run:  python src/phase4_gp_model.py
    """)


if __name__ == "__main__":
    main()