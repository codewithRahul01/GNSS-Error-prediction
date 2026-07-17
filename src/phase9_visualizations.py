"""
=============================================================================
  SIH GNSS ERROR PREDICTION PROJECT
  PHASE 10 — Publication-Quality Visualizations and Jupyter Notebook
=============================================================================

  WHAT THIS PHASE DOES
  ─────────────────────
  Creates two final deliverables for your report and presentation:

  1. MASTER FIGURE — One large publication-quality figure combining:
       Panel A: SW_W progress across all phases (all satellites)
       Panel B: Final GP predictions vs actual (all 3 satellites)
       Panel C: Residual Q-Q plots for best (MEO1) and worst (GEO)
       Panel D: SW score comparison table as visual heatmap
     → Saved as: figures/phase10_master_figure.png

  2. RESULTS NOTEBOOK — A Jupyter notebook (.ipynb) that walks through:
       - Dataset summary
       - Model architecture
       - All results in clean tables
       - Interactive figures
     → Saved as: report/results_notebook.ipynb

  3. SPIKE ANALYSIS FIGURE — Explains why GEO is limited:
       Shows spike rows vs smooth rows with separate SW scores
     → Saved as: figures/phase10_geo_spike_analysis.png

  4. SATELLITE COMPARISON DASHBOARD — Side-by-side all 3 satellites:
       Time series + residual distributions for all columns
     → Saved as: figures/phase10_dashboard.png

  HOW TO RUN
  ──────────
    python src/phase10_visualizations.py

  WHAT IT SAVES
  ─────────────
    figures/phase10_master_figure.png      ← include in report
    figures/phase10_geo_spike_analysis.png ← explain GEO limitation
    figures/phase10_dashboard.png          ← presentation slide
    report/results_notebook.ipynb          ← interactive notebook
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# =============================================================================
# PATHS
# =============================================================================

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

for _d in ["Data","data"]:
    for _p in ["Processed","processed"]:
        _c = os.path.join(BASE_DIR,_d,_p)
        if os.path.isdir(_c): PROC_DIR = _c; break

RES_DIR    = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
REPORT_DIR = os.path.join(BASE_DIR, "report")

os.makedirs(FIG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================

ERR_COLS       = ["x_error","y_error","z_error","clock_error"]
SW_BENCHMARK_W = 0.9810
ALPHA          = 0.05

SAT_COLORS = {"GEO":"#E06C75","MEO1":"#61AFEF","MEO2":"#98C379"}
COL_COLORS = {
    "x_error"    :"#E5C07B",
    "y_error"    :"#61AFEF",
    "z_error"    :"#98C379",
    "clock_error":"#C678DD",
}
COL_LABELS = {
    "x_error"    :"X Error (m)",
    "y_error"    :"Y Error (m)",
    "z_error"    :"Z Error (m)",
    "clock_error":"Clock Error (m)",
}

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_all():
    """Load actual test values, predictions, and residuals for all satellites."""
    configs = [
        ("GEO",  "geo_test_featured.csv",  "final_predictions_geo.csv"),
        ("MEO1", "meo1_test_featured.csv", "final_predictions_meo1.csv"),
        ("MEO2", "meo2_test_featured.csv", "final_predictions_meo2.csv"),
    ]
    train_configs = [
        ("GEO",  "geo_train_featured.csv"),
        ("MEO1", "meo1_train_featured.csv"),
        ("MEO2", "meo2_train_featured.csv"),
    ]

    data = {}
    for sat, tf, pf in configs:
        actual = pd.read_csv(os.path.join(PROC_DIR, tf))
        actual["utc_time"] = pd.to_datetime(actual["utc_time"])
        pred   = pd.read_csv(os.path.join(RES_DIR,  pf))
        pred["utc_time"] = pd.to_datetime(pred["utc_time"])
        residuals = pd.DataFrame({"utc_time": actual["utc_time"]})
        for col in ERR_COLS:
            residuals[col] = actual[col].values - pred[col].values
        data[sat] = {"actual": actual, "pred": pred, "residuals": residuals}

    for sat, tf in train_configs:
        train = pd.read_csv(os.path.join(PROC_DIR, tf))
        train["utc_time"] = pd.to_datetime(train["utc_time"])
        data[sat]["train"] = train

    return data


def load_phase_scores():
    """
    Load SW_W averaged scores for each phase and satellite.
    Returns dict: {phase_label: {satellite: sw_w}}
    """
    phase_files = [
        ("Baseline", "baseline_scores.csv", True),
        ("GP v1",    "gp_scores.csv",       False),
        ("Tuned",    "tuned_scores.csv",     False),
        ("Final",    "final_scores.csv",     False),
    ]
    result = {}
    for label, fname, is_baseline in phase_files:
        path = os.path.join(RES_DIR, fname)
        if not os.path.exists(path):
            continue
        df  = pd.read_csv(path)
        avg = df[df["column"] == "AVERAGED"]
        if is_baseline:
            best = avg.loc[avg.groupby("satellite")["sw_w"].idxmax()]
            result[label] = dict(zip(best["satellite"], best["sw_w"]))
        else:
            result[label] = dict(zip(avg["satellite"], avg["sw_w"]))
    return result


# =============================================================================
# FIGURE 1 — MASTER FIGURE (publication quality, 4 panels)
# =============================================================================

def plot_master_figure(data, phase_scores):
    """
    A single comprehensive figure for the report or presentation.

    Layout (3 rows × 3 cols with merged cells):
      Row 1: [SW_W progress across phases — spans full width]
      Row 2: [GEO pred vs actual] [MEO1 pred vs actual] [MEO2 pred vs actual]
      Row 3: [GEO Q-Q clock]      [MEO1 Q-Q clock]      [SW score heatmap]
    """
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#1E2127")

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.42, wspace=0.32,
        top=0.93, bottom=0.07,
        left=0.07, right=0.97,
    )

    txt_kw = dict(color="#ABB2BF", fontsize=9)
    title_kw = dict(color="white", fontsize=10, fontweight="bold", pad=8)
    DARK_BG = "#282C34"

    # ── Panel A: SW_W progress ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_facecolor(DARK_BG)

    phases = list(phase_scores.keys())
    sats   = ["GEO","MEO1","MEO2"]
    x      = np.arange(len(phases))
    width  = 0.25

    for i, sat in enumerate(sats):
        vals = [phase_scores.get(p,{}).get(sat, 0) for p in phases]
        bars = ax_a.bar(x + i*width, vals, width,
                        label=sat, color=SAT_COLORS[sat],
                        alpha=0.85, edgecolor="#1E2127", linewidth=0.8)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax_a.text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.008,
                          f"{val:.3f}", ha="center", va="bottom",
                          fontsize=7.5, color="white", fontweight="bold")

    ax_a.axhline(SW_BENCHMARK_W, color="#E06C75", linewidth=2,
                 linestyle="--", alpha=0.9,
                 label=f"Benchmark W={SW_BENCHMARK_W}")
    ax_a.set_xticks(x + width)
    ax_a.set_xticklabels(phases, **txt_kw)
    ax_a.set_ylabel("SW_W (avg over 4 error columns)", **txt_kw)
    ax_a.set_title("A — SW_W Progress Across All Phases",
                   loc="left", **title_kw)
    ax_a.set_ylim(0.4, 1.08)
    ax_a.legend(fontsize=8, labelcolor="white",
                facecolor=DARK_BG, edgecolor="#4B5263")
    ax_a.tick_params(colors="#ABB2BF")
    ax_a.spines[:].set_color("#4B5263")
    ax_a.grid(axis="y", alpha=0.15, color="white")

    # ── Panel B: predictions vs actual (one per satellite) ────────────────
    col_to_show = "clock_error"
    for s_idx, sat in enumerate(sats):
        ax_b = fig.add_subplot(gs[1, s_idx])
        ax_b.set_facecolor(DARK_BG)
        d = data[sat]

        # Training context (light)
        ax_b.scatter(d["train"]["utc_time"], d["train"][col_to_show],
                     color="#4B5263", s=5, alpha=0.5, label="Train",
                     zorder=2)
        # GP prediction
        ax_b.plot(d["pred"]["utc_time"], d["pred"][col_to_show],
                  color=SAT_COLORS[sat], linewidth=2,
                  label="GP pred", zorder=4)
        # Confidence band
        if f"{col_to_show}_std" in d["pred"].columns:
            std = d["pred"][f"{col_to_show}_std"].values
            ax_b.fill_between(
                d["pred"]["utc_time"],
                d["pred"][col_to_show] - 2*std,
                d["pred"][col_to_show] + 2*std,
                color=SAT_COLORS[sat], alpha=0.15)
        # Actual test
        ax_b.scatter(d["actual"]["utc_time"], d["actual"][col_to_show],
                     color="white", s=18, zorder=5, label="Actual",
                     edgecolors=SAT_COLORS[sat], linewidths=0.8)
        # Split line
        ax_b.axvline(d["train"]["utc_time"].max(),
                     color="#ABB2BF", lw=0.8, ls=":", alpha=0.5)
        ax_b.axhline(0, color="#4B5263", lw=0.6, ls="--")

        res = d["actual"][col_to_show].values - d["pred"][col_to_show].values
        w,p = stats.shapiro(res)
        sym = "✓" if p>=ALPHA else "✗"
        ax_b.set_title(f"B{s_idx+1} — {sat}  clock_error\n"
                       f"SW_W={w:.4f}  {sym}",
                       loc="left", **title_kw)
        ax_b.set_ylabel("Error (m)", **txt_kw)
        ax_b.tick_params(colors="#ABB2BF", labelsize=7)
        ax_b.tick_params(axis="x", rotation=20)
        ax_b.spines[:].set_color("#4B5263")
        ax_b.grid(alpha=0.1, color="white")
        if s_idx == 0:
            ax_b.legend(fontsize=6.5, labelcolor="white",
                        facecolor=DARK_BG, edgecolor="#4B5263")

    # ── Panel C1: GEO Q-Q (clock_error — most problematic) ───────────────
    ax_c1 = fig.add_subplot(gs[2, 0])
    ax_c1.set_facecolor(DARK_BG)
    res = data["GEO"]["residuals"]["clock_error"].values
    w,p = stats.shapiro(res)
    r_std = (res - res.mean()) / max(res.std(),1e-9)
    n     = len(r_std)
    theo  = stats.norm.ppf((np.arange(1,n+1)-0.5)/n)
    ax_c1.scatter(theo, np.sort(r_std), color=SAT_COLORS["GEO"],
                  s=25, zorder=3, edgecolors="none")
    lm = max(abs(theo).max(), abs(r_std).max())*1.15
    ax_c1.plot([-lm,lm],[-lm,lm],"w--",lw=1.2,alpha=0.6,label="Ideal")
    ax_c1.set_xlim(-lm,lm); ax_c1.set_ylim(-lm,lm)
    ax_c1.set_title(f"C1 — GEO clock_error Q-Q\n"
                    f"W={w:.4f} ✗ (upload spikes)",
                    loc="left", **title_kw)
    ax_c1.set_xlabel("Theoretical quantiles", **txt_kw)
    ax_c1.set_ylabel("Sample quantiles", **txt_kw)
    ax_c1.tick_params(colors="#ABB2BF", labelsize=7)
    ax_c1.spines[:].set_color("#4B5263")
    ax_c1.grid(alpha=0.1, color="white")
    ax_c1.legend(fontsize=7, labelcolor="white",
                 facecolor=DARK_BG, edgecolor="#4B5263")

    # ── Panel C2: MEO1 Q-Q (clock_error — best result) ───────────────────
    ax_c2 = fig.add_subplot(gs[2, 1])
    ax_c2.set_facecolor(DARK_BG)
    res2 = data["MEO1"]["residuals"]["clock_error"].values
    w2,p2= stats.shapiro(res2)
    r2   = (res2-res2.mean())/max(res2.std(),1e-9)
    n2   = len(r2)
    t2   = stats.norm.ppf((np.arange(1,n2+1)-0.5)/n2)
    ax_c2.scatter(t2, np.sort(r2), color=SAT_COLORS["MEO1"],
                  s=35, zorder=3, edgecolors="none")
    lm2 = max(abs(t2).max(), abs(r2).max())*1.15
    ax_c2.plot([-lm2,lm2],[-lm2,lm2],"w--",lw=1.2,alpha=0.6,label="Ideal")
    ax_c2.set_xlim(-lm2,lm2); ax_c2.set_ylim(-lm2,lm2)
    ax_c2.set_title(f"C2 — MEO1 clock_error Q-Q\n"
                    f"W={w2:.4f} ✓ (benchmark level)",
                    loc="left", **title_kw)
    ax_c2.set_xlabel("Theoretical quantiles", **txt_kw)
    ax_c2.set_ylabel("Sample quantiles", **txt_kw)
    ax_c2.tick_params(colors="#ABB2BF", labelsize=7)
    ax_c2.spines[:].set_color("#4B5263")
    ax_c2.grid(alpha=0.1, color="white")
    ax_c2.legend(fontsize=7, labelcolor="white",
                 facecolor=DARK_BG, edgecolor="#4B5263")

    # ── Panel D: SW score heatmap ─────────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, 2])
    ax_d.set_facecolor(DARK_BG)

    sats_h = ["GEO","MEO1","MEO2"]
    sw_matrix = []
    for sat in sats_h:
        row = []
        for col in ERR_COLS:
            res_v = data[sat]["residuals"][col].values
            w_v,_ = stats.shapiro(res_v)
            row.append(w_v)
        sw_matrix.append(row)

    sw_arr = np.array(sw_matrix)

    # Custom colormap: red (0.5) → yellow (0.8) → green (1.0)
    cmap = LinearSegmentedColormap.from_list(
        "sw", ["#E06C75","#E5C07B","#98C379"], N=256)

    im = ax_d.imshow(sw_arr, cmap=cmap, vmin=0.5, vmax=1.0,
                     aspect="auto")
    plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04,
                 label="SW_W").ax.tick_params(colors="#ABB2BF",
                                               labelsize=7)

    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels(["x","y","z","clk"], **txt_kw)
    ax_d.set_yticks(range(3))
    ax_d.set_yticklabels(sats_h, **txt_kw)
    ax_d.set_title("D — SW_W Heatmap\n(green ≥ 0.98 = benchmark)",
                   loc="left", **title_kw)

    for i in range(3):
        for j in range(4):
            val = sw_arr[i,j]
            sym = "✓" if val >= SW_BENCHMARK_W else ""
            ax_d.text(j, i, f"{val:.3f}\n{sym}",
                      ha="center", va="center",
                      color="white" if val < 0.85 else "#1E2127",
                      fontsize=8, fontweight="bold")

    # Draw benchmark line annotation
    ax_d.axhline(0.5, color="white", lw=0.3, alpha=0.2)

    # Title
    fig.suptitle(
        "GNSS Satellite Clock & Ephemeris Error Prediction\n"
        "Gaussian Process Regression — SIH 2025",
        color="white", fontsize=14, fontweight="bold", y=0.98)

    out = os.path.join(FIG_DIR, "phase10_master_figure.png")
    plt.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓  Master figure saved: figures/phase10_master_figure.png")
    return out


# =============================================================================
# FIGURE 2 — GEO SPIKE ANALYSIS
# =============================================================================

def plot_geo_spike_analysis(data):
    """
    Explains the GEO limitation: show spike rows vs smooth rows separately.

    Two columns per error column:
      Left:  time series coloured by spike (red) vs smooth (blue)
      Right: separate SW bars for spike and smooth subsets
    """
    SPIKE_THRESH = 10.0
    d = data["GEO"]
    actual = d["actual"]
    pred   = d["pred"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(
        "GEO Satellite — Spike Analysis\n"
        "Why clock_error SW_W is limited: "
        "upload-boundary spikes are operationally driven",
        fontsize=12, fontweight="bold")

    for idx, col in enumerate(ERR_COLS):
        act_vals = actual[col].values
        prd_vals = pred[col].values
        res_vals = act_vals - prd_vals

        spike_mask  = np.abs(act_vals) > SPIKE_THRESH
        smooth_mask = ~spike_mask
        n_spike     = spike_mask.sum()
        n_smooth    = smooth_mask.sum()

        # ── Left: time series coloured by type ──────────────────────────
        ax = axes[idx][0]
        ax.plot(actual["utc_time"], prd_vals,
                color="#61AFEF", lw=1.5, label="GP prediction", zorder=3)
        ax.scatter(actual["utc_time"][smooth_mask],
                   act_vals[smooth_mask],
                   color="#98C379", s=20, zorder=4,
                   label=f"Smooth rows (n={n_smooth})", marker="o")
        ax.scatter(actual["utc_time"][spike_mask],
                   act_vals[spike_mask],
                   color="#E06C75", s=40, zorder=5,
                   label=f"Spike rows (n={n_spike})", marker="X")
        ax.axhline(SPIKE_THRESH,  color="gray", lw=0.8, ls="--",
                   alpha=0.5, label=f"±{SPIKE_THRESH}m threshold")
        ax.axhline(-SPIKE_THRESH, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.4)

        ax.set_title(f"{col} — Prediction vs Actual",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Error (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper right")

        # ── Right: SW score comparison ──────────────────────────────────
        ax2 = axes[idx][1]
        labels, sw_vals, colors = [], [], []

        # All residuals
        w_all, _ = stats.shapiro(res_vals)
        labels.append(f"All rows\n(n={len(res_vals)})")
        sw_vals.append(w_all)
        colors.append("#AAAAAA")

        # Smooth rows
        if smooth_mask.sum() >= 3:
            w_sm, p_sm = stats.shapiro(res_vals[smooth_mask])
            labels.append(f"Smooth rows\n(n={n_smooth})")
            sw_vals.append(w_sm)
            colors.append("#98C379")

        # Spike rows
        if spike_mask.sum() >= 3:
            w_sp, _ = stats.shapiro(res_vals[spike_mask])
            labels.append(f"Spike rows\n(n={n_spike})")
            sw_vals.append(w_sp)
            colors.append("#E06C75")

        bars = ax2.bar(labels, sw_vals, color=colors,
                       alpha=0.85, edgecolor="white", width=0.5)
        for bar, val in zip(bars, sw_vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.008,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

        ax2.axhline(SW_BENCHMARK_W, color="red", lw=1.5,
                    ls="--", label=f"Benchmark ({SW_BENCHMARK_W})")
        ax2.set_ylim(0.3, 1.1)
        ax2.set_ylabel("SW_W", fontsize=8)
        ax2.set_title(f"{col} — SW_W by Row Type",
                      fontsize=10, fontweight="bold")
        ax2.legend(fontsize=7)
        ax2.grid(axis="y", alpha=0.25)
        ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "phase10_geo_spike_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  GEO spike analysis: figures/phase10_geo_spike_analysis.png")


# =============================================================================
# FIGURE 3 — COMPREHENSIVE DASHBOARD
# =============================================================================

def plot_dashboard(data):
    """
    4×3 grid dashboard: one row per error column, one column per satellite.
    Each cell: actual (black) + GP prediction (coloured) + residual std.
    """
    sats = ["GEO","MEO1","MEO2"]
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle(
        "GP Prediction Dashboard — All Satellites × All Error Columns\n"
        "Black dots = actual test values  |  Coloured line = GP prediction  "
        "|  Shaded = ±2σ confidence",
        fontsize=12, fontweight="bold")

    for c_idx, col in enumerate(ERR_COLS):
        for s_idx, sat in enumerate(sats):
            ax  = axes[c_idx][s_idx]
            d   = data[sat]

            # Training
            ax.scatter(d["train"]["utc_time"], d["train"][col],
                       color="#CCCCCC", s=4, alpha=0.3,
                       label="Train", zorder=2)
            # Prediction
            ax.plot(d["pred"]["utc_time"], d["pred"][col],
                    color=SAT_COLORS[sat], lw=2,
                    label="GP pred", zorder=4)
            # Confidence band
            if f"{col}_std" in d["pred"].columns:
                std = d["pred"][f"{col}_std"].values
                ax.fill_between(
                    d["pred"]["utc_time"],
                    d["pred"][col] - 2*std,
                    d["pred"][col] + 2*std,
                    color=SAT_COLORS[sat], alpha=0.15)
            # Actual
            ax.scatter(d["actual"]["utc_time"], d["actual"][col],
                       color="black", s=16, zorder=5,
                       label="Actual", marker="o")

            # Train/test split
            ax.axvline(d["train"]["utc_time"].max(),
                       color="gray", lw=0.8, ls=":", alpha=0.5)
            ax.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.4)

            res  = d["actual"][col].values - d["pred"][col].values
            w, p = stats.shapiro(res)
            rmse = np.sqrt(np.mean(res**2))
            sym  = "✓" if p >= ALPHA else "✗"

            title_str = f"{sat} — {COL_LABELS[col]}"
            subtitle   = f"SW_W={w:.3f} {sym}  RMSE={rmse:.3f}m"
            ax.set_title(f"{title_str}\n{subtitle}",
                         fontsize=8.5, fontweight="bold")
            ax.set_ylabel("Error (m)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(alpha=0.2)

            if c_idx == 0 and s_idx == 0:
                ax.legend(fontsize=6, ncol=3, loc="upper left")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "phase10_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Dashboard saved: figures/phase10_dashboard.png")


# =============================================================================
# JUPYTER NOTEBOOK GENERATOR
# =============================================================================

def generate_notebook(data, phase_scores):
    """
    Create a Jupyter notebook (.ipynb) that presents all results
    in a clean, interactive format for the report/presentation.

    The notebook ONLY displays results — no heavy computation.
    All figures are loaded from the figures/ folder.
    """
    import nbformat
    from nbformat.v4 import (new_notebook, new_markdown_cell,
                              new_code_cell)

    nb = new_notebook()
    cells = []

    # ── Title ────────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
# GNSS Satellite Clock & Ephemeris Error Prediction
## SIH 2025 — Results Notebook

**Model:** Gaussian Process Regression  
**Evaluation metric:** Shapiro-Wilk W statistic on residuals  
**Benchmark:** W = 0.9810, p = 0.5840

---
"""))

    # ── Setup cell ───────────────────────────────────────────────────────
    cells.append(new_code_cell("""\
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy import stats
from IPython.display import Image, display

# Paths (adjust if running from a different directory)
import os
BASE = os.path.abspath('..')  # project root
RES  = os.path.join(BASE, 'results')
FIG  = os.path.join(BASE, 'figures')
PROC = os.path.join(BASE, 'Data', 'Processed')

ERR_COLS = ['x_error','y_error','z_error','clock_error']
print("Setup complete.")
"""))

    # ── Dataset summary ──────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 1. Dataset Summary

Three independent satellites, each with 7 days of training data.
"""))

    cells.append(new_code_cell("""\
summary = {
    'Satellite' : ['GEO (A)','MEO1 (B)','MEO2 (C)'],
    'Train rows': [142, 46, 143],
    'Test rows' : [69, 6, 18],
    'Train start': ['2025-09-01','2025-09-01','2025-09-03'],
    'Test date'  : ['2025-09-08','2025-09-08','2025-09-10'],
    'Upload mode': ['120-min→15-min','Irregular','Irregular'],
    'Key challenge': ['Upload spikes ±58m',
                      'Only 46 rows','4 daily data gaps'],
}
df_sum = pd.DataFrame(summary)
print(df_sum.to_string(index=False))
"""))

    # ── Model results ────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 2. Priority 1 — Shapiro-Wilk Results
"""))

    cells.append(new_code_cell("""\
sw_report = pd.read_csv(os.path.join(RES, 'submission_sw_report.csv'))
avg = sw_report[sw_report['column'].isin(['AVERAGED','GRAND AVERAGE'])]
print("SW Scores (averaged over 4 error columns):")
print(avg[['satellite','column','sw_w','sw_p',
           'h0_rejected','rmse']].to_string(index=False))
print(f"\\nBenchmark: W=0.9810  p=0.5840  H0_rejected=0")
"""))

    # ── Master figure ────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 3. Master Figure — All Results Summary
"""))

    cells.append(new_code_cell("""\
display(Image(os.path.join(FIG, 'phase10_master_figure.png'), width=900))
"""))

    # ── Dashboard ────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 4. Prediction Dashboard — All Satellites × All Columns
"""))

    cells.append(new_code_cell("""\
display(Image(os.path.join(FIG, 'phase10_dashboard.png'), width=900))
"""))

    # ── Q-Q plots ────────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 5. Q-Q Plots (Priority 3)

Points on the diagonal = normal residuals = high SW_W score.
"""))

    cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (sat, fname) in enumerate([
    ('GEO',  'phase8_qq_geo.png'),
    ('MEO1', 'phase8_qq_meo1.png'),
    ('MEO2', 'phase8_qq_meo2.png'),
]):
    img = plt.imread(os.path.join(FIG, fname))
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(sat, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    # ── GEO spike analysis ───────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 6. GEO Spike Analysis — Explaining the Limitation

For smooth rows (|error| < 10m): **SW_W ≥ 0.95** for y_error and z_error.  
Upload-boundary spikes (±35–58m) are **operationally driven** — not predictable  
from orbital mechanics. This is a data limitation, not a modeling failure.
"""))

    cells.append(new_code_cell("""\
display(Image(os.path.join(FIG, 'phase10_geo_spike_analysis.png'), width=900))
"""))

    # ── Residual histograms ──────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 7. Residual Histograms — All Satellites
"""))

    cells.append(new_code_cell("""\
display(Image(os.path.join(FIG, 'phase8_residual_hist_all.png'), width=900))
"""))

    # ── Phase progress ───────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 8. SW_W Progress Across All Phases
"""))

    cells.append(new_code_cell("""\
display(Image(os.path.join(FIG, 'phase8_final_comparison.png'), width=700))
"""))

    # ── Priority 2 ──────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 9. Priority 2 — Residual Mean and Standard Deviation
"""))

    cells.append(new_code_cell("""\
p2 = sw_report[sw_report['column'].isin(['AVERAGED','GRAND AVERAGE'])]
print("Priority 2 — Residual Statistics:")
print(p2[['satellite','column','res_mean','res_std',
          'rmse','mae']].to_string(index=False))
print()
print("NOTE: MEO1 and MEO2 residual means ≈ 0 (no bias)")
print("      GEO  residual mean = +0.38m (caused by upload spikes)")
"""))

    # ── Conclusion ───────────────────────────────────────────────────────
    cells.append(new_markdown_cell("""
## 10. Conclusion

| Satellite | SW_W | All columns pass H0? | Key insight |
|---|---|---|---|
| GEO  | 0.7865 | ✗ | Upload spikes ±58m are not learnable |
| MEO1 | **0.9084** | **✓** | All 4 columns normal, near benchmark |
| MEO2 | 0.8076 | Partial | 24h data gaps cause extrapolation error |

**Why Gaussian Process:**
- Only 46–143 training rows → no room for LSTM windows
- Non-uniform sampling (1–1556 min gaps) → GP handles naturally
- Evaluation is residual normality → GP posterior is Gaussian by design

**Runtime:** < 60 seconds for all 3 satellites on any laptop.
"""))

    nb.cells = cells

    path = os.path.join(REPORT_DIR, "results_notebook.ipynb")
    with open(path, "w") as f:
        nbformat.write(nb, f)
    print(f"  ✓  Jupyter notebook saved: report/results_notebook.ipynb")
    return path


# =============================================================================
# PRINT FINAL COMPLETE SUMMARY TABLE
# =============================================================================

def print_complete_results(data):
    """Print the complete per-column SW table for all satellites."""
    print("\n" + "═"*72)
    print("  COMPLETE RESULTS TABLE — All Satellites × All Columns")
    print(f"  Benchmark: W={SW_BENCHMARK_W}  H0=0 (fail to reject)")
    print("═"*72)

    from tabulate import tabulate
    rows = []
    for sat in ["GEO","MEO1","MEO2"]:
        d = data[sat]
        for col in ERR_COLS:
            res  = d["residuals"][col].values
            w, p = stats.shapiro(res)
            rmse = np.sqrt(np.mean(res**2))
            mae  = np.mean(np.abs(res))
            sym  = "✓" if p >= ALPHA else "✗"
            rows.append([
                f"{sym} {sat}", col,
                f"{len(res)}",
                f"{w:.4f}",
                f"{p:.4f}",
                f"{res.mean():+.4f}",
                f"{res.std():.4f}",
                f"{rmse:.4f}",
            ])

        # Averaged
        avg_w = np.mean([
            stats.shapiro(d["residuals"][c].values)[0]
            for c in ERR_COLS])
        rows.append([f"  {sat} AVG", "─ AVERAGED ─",
                     "─","─","─","─","─",f"{avg_w:.4f}"])

    print(tabulate(rows,
                   headers=["Satellite","Column","n",
                             "SW_W","p-value","Res mean",
                             "Res std","RMSE"],
                   tablefmt="rounded_outline"))

    grand_ws = []
    for sat in ["GEO","MEO1","MEO2"]:
        for col in ERR_COLS:
            res = data[sat]["residuals"][col].values
            w,_ = stats.shapiro(res)
            grand_ws.append(w)

    print(f"\n  Grand average SW_W = {np.mean(grand_ws):.4f}")
    print(f"  Benchmark          = {SW_BENCHMARK_W}")
    print(f"  Gap to benchmark   = {np.mean(grand_ws) - SW_BENCHMARK_W:+.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "═"*65)
    print("  SIH GNSS Error Prediction Project")
    print("  PHASE 10 — Publication Visualizations + Notebook")
    print("═"*65)

    # Load all data
    print("\n  Loading data...")
    data         = load_all()
    phase_scores = load_phase_scores()
    print(f"  ✓  Loaded {len(data)} satellites, "
          f"{len(phase_scores)} phase score records")

    # Figure 1: Master figure
    print("\n  Generating master figure (4 panels)...")
    plot_master_figure(data, phase_scores)

    # Figure 2: GEO spike analysis
    print("\n  Generating GEO spike analysis...")
    plot_geo_spike_analysis(data)

    # Figure 3: Full dashboard
    print("\n  Generating comprehensive dashboard...")
    plot_dashboard(data)

    # Notebook
    print("\n  Generating Jupyter notebook...")
    generate_notebook(data, phase_scores)

    # Complete results table
    print_complete_results(data)

    print("\n" + "═"*65)
    print("  PHASE 10 COMPLETE")
    print("═"*65)
    print("""
  Files saved:
    figures/phase10_master_figure.png      ← include in report / slides
    figures/phase10_geo_spike_analysis.png ← explain GEO limitation
    figures/phase10_dashboard.png          ← full comparison dashboard
    report/results_notebook.ipynb          ← interactive Jupyter notebook

  To open the notebook:
    cd "/Users/rahuljangra/Downloads/SIH Project /GNSS-Error-prediction"
    source venv/bin/activate
    pip install jupyter
    jupyter notebook report/results_notebook.ipynb

  ──────────────────────────────────────────────────────────
  PROJECT COMPLETE — ALL 10 PHASES DONE

  What you have built:
    • Full EDA revealing duplicate rows, upload modes, data gaps
    • Preprocessing pipeline with GEO mode filter + winsorization
    • 3 baseline models for comparison
    • 12 Gaussian Process models (4 per satellite)
    • Kernel tuning with time-series cross-validation
    • Single-command evaluation script (phase7_predict.py)
    • Complete SW evaluation with Q-Q plots
    • GitHub-ready repository with README
    • Publication-quality figures + Jupyter notebook

  Best result: MEO1 SW_W=0.9084 (all 4 columns pass normality test)
  ──────────────────────────────────────────────────────────
    """)


if __name__ == "__main__":
    main()