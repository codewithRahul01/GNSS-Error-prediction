"""
=============================================================================
  SIH GNSS ERROR PREDICTION PROJECT
  MASTER IMPORTS FILE  —  imports.py
=============================================================================

  PURPOSE
  ───────
  This file is the single source of truth for every library and module
  used across all 11 phases of the project.

  HOW TO USE
  ──────────
  Option A (recommended): copy-paste only the section you need into each
  phase script. Each section is clearly labelled with which phases use it.

  Option B: at the top of any phase script, write:
      from imports import *
  and every name in this file becomes available.

  HOW TO INSTALL EVERYTHING
  ─────────────────────────
  Run once before starting the project:
      pip install -r requirements.txt

  Or install manually:
      pip install pandas numpy scipy scikit-learn matplotlib seaborn \
                  tabulate statsmodels openpyxl

  ENVIRONMENT DETAILS
  ───────────────────
  Python  : 3.8 or higher  (tested on 3.12.3)
  pandas  : 3.x             (2.x also works with minor changes)
  numpy   : 2.x             (1.x also works)
  scipy   : 1.17+
  sklearn : 1.8+
=============================================================================
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PYTHON STANDARD LIBRARY
# Used in: ALL phases
# These come built-in with Python — no installation needed.
# ─────────────────────────────────────────────────────────────────────────────

import os           # file paths, directory creation (os.makedirs, os.path.join)
import sys          # exit on error (sys.exit), Python version check (sys.version_info)
import warnings     # suppress noisy 3rd-party warnings (warnings.filterwarnings)
import time         # measure how long model training takes (time.time())
import math         # basic math constants (math.pi, math.sqrt, math.inf)
import json         # save/load config files and results as JSON
import argparse     # command-line argument parsing for the final predict.py script

from pathlib import Path          # modern file path handling (alternative to os.path)

from dataclasses import (
    dataclass,                    # clean class definitions for configs and results
    field,                        # default_factory for mutable defaults in dataclasses
)

from typing import (
    List,                         # type hint: List[str], List[float]
    Tuple,                        # type hint: Tuple[np.ndarray, np.ndarray]
    Dict,                         # type hint: Dict[str, pd.DataFrame]
    Optional,                     # type hint: Optional[str] = None
    Union,                        # type hint: Union[str, Path]
)

from copy import deepcopy         # copy GP models without sharing state
import itertools                  # itertools.product() for hyperparameter grid search
import functools                  # functools.partial() for kernel factory functions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — NUMPY  (numerical computing)
# Used in: ALL phases
# ─────────────────────────────────────────────────────────────────────────────
#
# Why numpy?
#   All error values (x, y, z, clock) are stored as numpy arrays internally.
#   GP kernels, scalers, and metrics all work on numpy arrays, not DataFrames.
#
# Key things we use:
#   np.array, np.zeros, np.ones, np.linspace, np.arange
#   np.mean, np.std, np.var, np.median
#   np.sqrt, np.abs, np.exp, np.log, np.sin, np.cos, np.pi
#   np.percentile, np.quantile
#   np.concatenate, np.stack, np.vstack, np.hstack
#   np.isnan, np.isinf, np.where, np.clip
#   np.polyfit, np.polyval  (used in polynomial baseline model)
#   np.random  (for reproducibility — np.random.seed)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

from numpy.linalg import (
    norm,      # vector/matrix norm — used in error magnitude calculation
    inv,       # matrix inverse — used internally by GP
    eig,       # eigenvalues — used in GP covariance checks
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PANDAS  (data loading and manipulation)
# Used in: ALL phases (Phase 0 through Phase 11)
# ─────────────────────────────────────────────────────────────────────────────
#
# Why pandas?
#   Your data lives in CSVs with timestamps and float columns.
#   pandas is the standard tool for loading, cleaning, filtering,
#   and grouping tabular data. Every data file in this project is
#   loaded as a pd.DataFrame.
#
# Key things we use:
#   pd.read_csv()             — load the 6 GNSS error CSV files
#   pd.to_datetime()          — parse the utc_time string column
#   df.sort_values()          — sort by timestamp (always ascending)
#   df.drop_duplicates()      — remove the 101 duplicate rows in MEO2
#   df.groupby()              — group by date for daily statistics
#   df.diff()                 — compute time gaps between consecutive rows
#   df.quantile()             — compute Q1, Q3 for outlier detection
#   df.describe()             — quick summary statistics
#   df.isna(), df.fillna()    — check and handle missing values
#   df.to_csv()               — save processed / prediction outputs
#   pd.concat()               — combine train + test or multiple satellites
#   pd.Timestamp()            — create a specific datetime value
#   pd.Timedelta()            — represent time durations
#   pd.DatetimeIndex          — index of datetime values
#   pd.read_excel()           — read the SW_ReferenceData.xlsx benchmark file
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

from pandas import (
    DataFrame,       # the main 2D table type
    Series,          # a single column / 1D array with labels
    Timestamp,       # a single datetime value (pd.Timestamp("2025-09-01"))
    DatetimeIndex,   # an index of datetimes
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SCIPY  (scientific computing and statistics)
# Used in: Phase 1 (EDA), Phase 3 (baselines), Phase 8 (evaluation)
# ─────────────────────────────────────────────────────────────────────────────
#
# Why scipy?
#   scipy.stats contains the Shapiro-Wilk test — the PRIMARY evaluation
#   metric for this competition. Every submission is judged on this.
#   scipy also provides signal analysis tools (for finding dominant periods),
#   interpolation (for imputing the MEO2 data gap), and curve fitting
#   (for the polynomial baseline model).
#
# Key things we use:
#
# --- STATS (most important for this project) ---
#   shapiro(x)        → (W, p)  ← THE evaluation metric
#   normaltest(x)     → alternative normality test (D'Agostino–Pearson)
#   kstest(x, 'norm') → Kolmogorov-Smirnov test (3rd normality check)
#   anderson(x)       → Anderson-Darling test (4th normality check)
#   skew(x)           → skewness (0 = symmetric, used in EDA section 1.5)
#   kurtosis(x)       → kurtosis (3 = normal, >3 = heavy tails)
#   norm.ppf(q)       → inverse normal CDF, used to build Q-Q plot points
#   norm.pdf(x)       → normal PDF, used to overlay normal curve on histograms
#
# --- INTERPOLATION ---
#   interp1d          → linear / cubic interpolation for MEO2 gap imputation
#   CubicSpline       → smooth cubic interpolation (better than linear)
#
# --- SIGNAL ANALYSIS ---
#   periodogram       → finds dominant frequencies in error time series
#   welch             → power spectral density (smoother than periodogram)
#
# --- OPTIMIZATION ---
#   curve_fit         → fit custom functions (upload sawtooth model for GEO)
#   minimize          → general optimization (alternative kernel tuning)
# ─────────────────────────────────────────────────────────────────────────────

from scipy import stats

from scipy.stats import (
    shapiro,          # THE evaluation metric: Shapiro-Wilk normality test
    normaltest,       # D'Agostino-Pearson normality test (backup check)
    kstest,           # Kolmogorov-Smirnov test (backup check)
    anderson,         # Anderson-Darling test (backup check)
    skew,             # skewness of a distribution
    kurtosis,         # kurtosis of a distribution
    norm as scipy_norm,  # normal distribution object: .ppf(), .pdf(), .cdf()
)

from scipy.interpolate import (
    interp1d,         # general interpolation: linear, cubic, nearest
    CubicSpline,      # smooth cubic spline interpolation (better for MEO2 gap)
)

from scipy.signal import (
    periodogram,      # raw power spectrum — finds dominant periods in errors
    welch,            # Welch's method — smoother power spectrum estimate
)

from scipy.optimize import (
    curve_fit,        # fit sawtooth / polynomial models to GEO upload pattern
    minimize,         # general function minimization
)

from scipy.linalg import solve    # solve linear systems (used internally by GP)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — STATSMODELS  (advanced time series analysis)
# Used in: Phase 1 (EDA), Phase 3 (baselines)
# ─────────────────────────────────────────────────────────────────────────────
#
# Why statsmodels?
#   scipy gives you basic normality tests.
#   statsmodels adds proper time series diagnostic tools:
#     - ADF test: is the series stationary? (critical before modeling)
#     - KPSS test: double-check stationarity
#     - ACF/PACF: how far back does autocorrelation go? (validates GP kernel
#       length scale choice and tells you if a 7-day lookback makes sense)
#     - Seasonal decompose: separates trend + seasonality + residual visually
# ─────────────────────────────────────────────────────────────────────────────

from statsmodels.tsa.stattools import (
    adfuller,         # Augmented Dickey-Fuller stationarity test
                      # H0: series has a unit root (non-stationary)
                      # p < 0.05 → reject H0 → series IS stationary
    kpss,             # KPSS test (complement to ADF, different null hypothesis)
                      # H0: series IS stationary
                      # p < 0.05 → reject H0 → series is NOT stationary
    acf,              # autocorrelation function values (for analysis)
    pacf,             # partial autocorrelation function values
)

from statsmodels.tsa.seasonal import (
    seasonal_decompose,  # decompose time series into: trend + seasonal + residual
)

from statsmodels.graphics.tsaplots import (
    plot_acf,         # plot autocorrelation function
    plot_pacf,        # plot partial autocorrelation function
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SCIKIT-LEARN: PREPROCESSING
# Used in: Phase 2 (Preprocessing), Phase 4 (GP model), Phase 5 (Tuning)
# ─────────────────────────────────────────────────────────────────────────────
#
# Why do we need scaling?
#   GP kernel hyperparameters (length scale, amplitude) are defined in the
#   SAME units as the input and output data. If t_min ranges from 0 to 9000
#   and x_error ranges from -75 to +58, the optimizer struggles to find
#   sensible hyperparameters. Scaling brings everything to unit variance,
#   making optimization much more reliable.
#
# Which scaler for which purpose:
#   StandardScaler  → mean=0, std=1. Use for t_min (input) and all error
#                     columns (output). Best for GP which assumes Gaussian data.
#
#   RobustScaler    → uses median and IQR instead of mean and std.
#                     More resistant to the large GEO outliers.
#                     Use as an alternative for GEO's heavily-spiked data.
#
#   MinMaxScaler    → scales to [0, 1] range. Only use if you have a
#                     specific reason (not recommended for GP).
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import (
    StandardScaler,       # zero-mean, unit-variance scaling  ← primary scaler
    MinMaxScaler,         # scale to [0, 1]  (rarely needed here)
    RobustScaler,         # median/IQR scaling  ← good alternative for GEO
    PolynomialFeatures,   # create polynomial terms: t, t², t³ for baseline
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SCIKIT-LEARN: GAUSSIAN PROCESS  (the main model)
# Used in: Phase 4 (GP design), Phase 5 (Kernel tuning), Phase 7 (Prediction)
# ─────────────────────────────────────────────────────────────────────────────
#
# The GP model has two parts: the regressor and the kernel.
# You build a kernel, pass it to the regressor, and the regressor optimizes
# the kernel's hyperparameters during fit().
#
# KERNEL COMPONENTS:
#
#   RBF (Radial Basis Function)
#   ────────────────────────────
#   Also called "squared exponential". Produces infinitely smooth functions.
#   Use: captures slow, smooth drift in errors between upload cycles.
#   Parameter: length_scale — how quickly correlation drops with time.
#              Small length_scale = wiggly/fast. Large = smooth/slow.
#   k(t1,t2) = exp(-||t1-t2||² / (2 × length_scale²))
#
#   Matern (nu=1.5 or nu=2.5)
#   ──────────────────────────
#   Similar to RBF but allows rougher functions. nu=1.5 is once-differentiable,
#   nu=2.5 is twice-differentiable. More realistic for real-world signals.
#   Use: MEO orbit error trend (not perfectly smooth).
#
#   ExpSineSquared (Periodic kernel)
#   ─────────────────────────────────
#   Produces periodic functions that repeat with a learnable period.
#   Use: the dominant periodic component of errors.
#        GEO  → periodicity=1440 (24h daily cycle)
#        MEO  → periodicity=720  (12h half-day cycle) + 1440 (daily)
#   Parameters:
#     length_scale — how quickly correlation drops within a period
#     periodicity  — the period in input units (minutes here)
#   k(t1,t2) = exp(-2 × sin²(π|t1-t2|/periodicity) / length_scale²)
#
#   WhiteKernel (Noise)
#   ────────────────────
#   Models observation noise: unexplained random variance.
#   Use: upload-boundary spikes in GEO that aren't predictable from trends.
#   k(t1,t2) = noise_level × δ(t1,t2)
#   IMPORTANT: always include this. Without it, GP interpolates exactly
#              through every noisy training point (overfitting).
#
#   ConstantKernel (Amplitude)
#   ───────────────────────────
#   A scalar multiplier applied to any kernel: k_new = C * k_old.
#   Use: wrapping RBF/Matern to allow the optimizer to scale the amplitude.
#   Without it, the kernel amplitude is fixed — the optimizer can't find
#   the right signal variance.
#
#   RationalQuadratic
#   ──────────────────
#   Equivalent to a mixture of RBF kernels at different scales.
#   Use: when errors show structure at multiple time scales simultaneously.
#
#   DotProduct
#   ───────────
#   Produces polynomial-like functions. Rarely used for this project,
#   but available if you want to model a long-term trend explicitly.
#
# HOW TO COMBINE KERNELS:
#   Kernels are added (+) or multiplied (*).
#   Addition (+) means: "the output has BOTH components"
#     → trend + periodicity + noise
#   Multiplication (*) means: "the correlation has BOTH properties"
#     → use C * RBF to add amplitude scaling
#
# EXAMPLE (GEO):
#   kernel = (
#       ConstantKernel(1.0) * RBF(length_scale=240)    # smooth drift
#     + ConstantKernel(1.0) * ExpSineSquared(1440)      # daily cycle
#     + WhiteKernel(1.0)                                # spike noise
#   )
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import (
    RBF,                  # smooth long-range trend kernel
    Matern,               # rougher trend kernel (nu=1.5 or 2.5)
    ExpSineSquared,       # periodic kernel — the most important for this project
    WhiteKernel,          # observation noise kernel — always include this
    ConstantKernel,       # amplitude multiplier — wrap other kernels with this
    DotProduct,           # polynomial-like kernel (rarely needed here)
    RationalQuadratic,    # multi-scale kernel (alternative to RBF)
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SCIKIT-LEARN: MODEL SELECTION AND METRICS
# Used in: Phase 3 (baselines), Phase 5 (cross-validation), Phase 8 (eval)
# ─────────────────────────────────────────────────────────────────────────────
#
# Cross-validation for GP:
#   We cannot use random k-fold CV because time series data must stay ordered.
#   LeaveOneOut is the best option for small datasets: remove one point,
#   train on the rest, predict the removed point, repeat for all points.
#   For 46-143 training points this is computationally feasible.
#
# Metrics:
#   The PRIMARY metric is Shapiro-Wilk W (not RMSE), but we still compute
#   RMSE and MAE to understand how close predictions are numerically.
#   R² tells us what fraction of variance our model explains.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import (
    LeaveOneOut,          # remove 1 point at a time — best for small datasets
    KFold,                # standard k-fold (use TimeSeriesSplit instead for TS)
    cross_val_score,      # run CV and collect scores automatically
    TimeSeriesSplit,      # time-respecting CV: train on past, test on future
)

from sklearn.metrics import (
    mean_squared_error,   # MSE — take sqrt for RMSE: np.sqrt(MSE)
    mean_absolute_error,  # MAE — robust to outliers (no squaring)
    r2_score,             # R² — fraction of variance explained (1.0 = perfect)
)

from sklearn.pipeline import Pipeline  # chain scaler + GP into one object


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SCIKIT-LEARN: LINEAR MODELS  (baseline models)
# Used in: Phase 3 (baselines)
# ─────────────────────────────────────────────────────────────────────────────
#
# Three baseline models we build before the GP:
#   1. Persistence: last observed value repeated (no sklearn needed)
#   2. Linear extrapolation: LinearRegression on last N points
#   3. Polynomial fit: LinearRegression with PolynomialFeatures (degree 2-3)
#
# Ridge is LinearRegression with L2 regularization — more stable when
# fitting polynomials to small datasets (prevents wild extrapolation).
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import (
    LinearRegression,     # ordinary least squares — simplest baseline
    Ridge,                # L2-regularized regression — better for small data
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MATPLOTLIB  (core plotting library)
# Used in: Phase 1 (EDA), Phase 8 (eval), Phase 10 (visualizations)
# ─────────────────────────────────────────────────────────────────────────────
#
# matplotlib is the foundation of all plots in this project.
# Every figure — time series, histograms, Q-Q plots, boxplots — is built
# with matplotlib, sometimes with seaborn as a higher-level layer on top.
#
# Key things we use:
#   plt.figure(), plt.subplots()     — create figure and axes
#   plt.subplot(), GridSpec          — complex multi-panel layouts
#   ax.plot(), ax.scatter()          — line and scatter plots
#   ax.hist()                        — histograms of residuals
#   ax.boxplot()                     — boxplots for outlier comparison
#   ax.axhline(), ax.axvline()       — horizontal/vertical reference lines
#   ax.fill_between()                — GP confidence interval shading
#   ax.set_title(), ax.set_xlabel()  — labels
#   ax.legend(), ax.grid()           — legend and grid
#   plt.tight_layout()               — prevent label overlap
#   plt.savefig(path, dpi=150)       — save to file
#   plt.close()                      — free memory after saving
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")          # non-interactive: saves to file without a display
                               # Change to "TkAgg" or remove this line if you want
                               # pop-up plot windows on your local machine

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec        # complex subplot layouts
import matplotlib.patches as mpatches        # legend colour patches
from matplotlib.lines import Line2D          # custom legend line entries
from matplotlib.ticker import MaxNLocator    # clean integer tick marks


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — SEABORN  (statistical visualization)
# Used in: Phase 1 (EDA), Phase 8 (eval), Phase 10 (visualizations)
# ─────────────────────────────────────────────────────────────────────────────
#
# seaborn sits on top of matplotlib and provides:
#   sns.heatmap()         — correlation matrices (Phase 1 Fig 4)
#   sns.histplot()        — histograms with KDE overlay (residual distributions)
#   sns.kdeplot()         — kernel density estimate curves
#   sns.boxplot()         — styled boxplots
#   sns.scatterplot()     — styled scatter plots with hue/size/style options
#   sns.set_style()       — apply a consistent visual theme to all plots
#   sns.set_palette()     — set a colour palette
# ─────────────────────────────────────────────────────────────────────────────

import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — TABULATE  (terminal tables)
# Used in: Phase 1 (EDA), Phase 8 (evaluation report)
# ─────────────────────────────────────────────────────────────────────────────
#
# Prints clean, formatted tables in the terminal.
# The evaluator will see your printed output — clean tables look professional.
# Formats used: "rounded_outline", "simple", "github" (for README tables)
# ─────────────────────────────────────────────────────────────────────────────

from tabulate import tabulate


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — OPENPYXL  (Excel file reading)
# Used in: Phase 1 (EDA) — reading SW_ReferenceData.xlsx
# ─────────────────────────────────────────────────────────────────────────────
#
# pandas uses openpyxl as the backend for pd.read_excel().
# You don't call openpyxl directly — just having it installed lets
# pd.read_excel() work. Included here for documentation completeness.
# ─────────────────────────────────────────────────────────────────────────────

import openpyxl   # pandas uses this as backend for pd.read_excel()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — PROJECT-WIDE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
#
# Define once here. Import from this file in every phase script.
# This prevents typos and makes it easy to update (change one place, done).
# ─────────────────────────────────────────────────────────────────────────────

# Speed of light (for unit conversion reference — data is already in meters)
C_LIGHT = 299_792_458.0       # meters per second

# Standard column names in every GNSS error CSV file
COLS     = ["utc_time", "x_error", "y_error", "z_error", "clock_error"]
ERR_COLS = ["x_error", "y_error", "z_error", "clock_error"]

# Shapiro-Wilk benchmark scores from the problem statement
SW_BENCHMARK_W   = 0.9810
SW_BENCHMARK_P   = 0.5840
SW_ALPHA         = 0.05      # significance level for H0 rejection

# Outlier detection threshold (k × IQR)
OUTLIER_K = 3.0

# GEO upload mode switch: only use data from this date for training
GEO_MODE_SWITCH_DATE = "2025-09-03"

# File names (relative to data/raw/)
FILES = {
    "GEO_Train"  : "DATA_GEO_Train.csv",
    "GEO_Test"   : "DATA_GEO_Test.csv",
    "MEO1_Train" : "DATA_MEO_Train.csv",
    "MEO1_Test"  : "DATA_MEO_Test.csv",
    "MEO2_Train" : "DATA_MEO_Train2.csv",
    "MEO2_Test"  : "DATA_MEO_Test2.csv",
    "SW_REF"     : "SW_ReferenceData.xlsx",
}

# Consistent colours across all figures
SAT_COLORS = {
    "GEO"  : "#E06C75",   # soft red
    "MEO1" : "#61AFEF",   # blue
    "MEO2" : "#98C379",   # green
}

COL_COLORS = {
    "x_error"     : "#E5C07B",   # amber
    "y_error"     : "#61AFEF",   # blue
    "z_error"     : "#98C379",   # green
    "clock_error" : "#C678DD",   # purple
}

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — GLOBAL CONFIGURATION (paths)
# ─────────────────────────────────────────────────────────────────────────────
#
# Call  setup_paths(__file__)  at the top of every phase script to get
# the correct absolute paths regardless of where you run the script from.
# ─────────────────────────────────────────────────────────────────────────────

def setup_paths(caller_file: str = __file__) -> Dict[str, Path]:
    """
    Returns a dict of absolute paths for the project folders.

    Usage (in any phase script):
        from imports import setup_paths
        PATHS = setup_paths(__file__)
        df = pd.read_csv(PATHS["raw"] / "DATA_GEO_Train.csv")
    """
    base = Path(caller_file).resolve().parent.parent
    paths = {
        "base"      : base,
        "raw"       : base / "data" / "raw",
        "processed" : base / "data" / "processed",
        "src"       : base / "src",
        "results"   : base / "results",
        "figures"   : base / "figures",
        "report"    : base / "report",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — QUICK-REFERENCE: WHAT GOES WHERE
# ─────────────────────────────────────────────────────────────────────────────
#
#  PHASE          SECTIONS NEEDED
#  ─────────────────────────────────────────────────────────
#  Phase 0+1 EDA           1, 2, 3, 4, 10, 11, 12, 13, 14, 15
#  Phase 2 Preprocessing   1, 2, 3, 6, 14, 15
#  Phase 3 Baselines       1, 2, 3, 4, 8, 9, 10, 14, 15
#  Phase 4 GP Model        1, 2, 3, 4, 6, 7, 14, 15
#  Phase 5 Kernel Tuning   1, 2, 3, 4, 7, 8, 14, 15
#  Phase 6 Improvements    1, 2, 3, 4, 5, 6, 7, 14, 15
#  Phase 7 Prediction      1, 2, 3, 6, 7, 14, 15
#  Phase 8 Evaluation      1, 2, 3, 4, 10, 11, 12, 14, 15
#  Phase 9 Submission      1, 2, 3, 7, 14, 15
#  Phase 10 Visualization  1, 2, 3, 10, 11, 12, 14, 15
#  Phase 11 GitHub         14, 15 (just constants and paths)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST — run this file directly to verify everything imports correctly
#   python src/imports.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  SIH GNSS Project — Master Imports Self-Test")
    print("═" * 60)

    checks = {
        "numpy"        : f"version {np.__version__}",
        "pandas"       : f"version {pd.__version__}",
        "scipy.stats"  : "shapiro, normaltest, kstest, skew, kurtosis",
        "scipy.interp" : "interp1d, CubicSpline",
        "scipy.signal" : "periodogram, welch",
        "scipy.optim"  : "curve_fit",
        "statsmodels"  : "adfuller, kpss, acf, pacf, seasonal_decompose",
        "sklearn.GP"   : "GaussianProcessRegressor + all kernels",
        "sklearn.pre"  : "StandardScaler, RobustScaler, PolynomialFeatures",
        "sklearn.sel"  : "LeaveOneOut, TimeSeriesSplit, cross_val_score",
        "sklearn.met"  : "MSE, MAE, R2",
        "sklearn.lin"  : "LinearRegression, Ridge",
        "matplotlib"   : f"version {matplotlib.__version__}  [backend: {matplotlib.get_backend()}]",
        "seaborn"      : f"version {sns.__version__}",
        "tabulate"     : "ok",
        "openpyxl"     : f"version {openpyxl.__version__}",
    }

    for lib, detail in checks.items():
        print(f"  ✓  {lib:<20} {detail}")

    print("\n  Project constants:")
    print(f"    ERR_COLS   = {ERR_COLS}")
    print(f"    GEO mode switch date = {GEO_MODE_SWITCH_DATE}")
    print(f"    SW benchmark  W={SW_BENCHMARK_W}  p={SW_BENCHMARK_P}  α={SW_ALPHA}")
    print(f"    Outlier rule  {OUTLIER_K}×IQR")
    print(f"    Random seed   {RANDOM_SEED}")

    PATHS = setup_paths()
    print("\n  Project paths:")
    for name, path in PATHS.items():
        print(f"    {name:<12} {path}")

    print("\n  ✓  ALL IMPORTS OK — ready to build all phases\n")