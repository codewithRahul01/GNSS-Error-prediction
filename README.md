# GNSS Satellite Clock and Ephemeris Error Prediction

> **SIH 2025 Smart India Hackathon — GNSS Error Forecasting**
>
> AI/ML-based prediction of GNSS satellite clock and ephemeris errors
> using 7 days of historical data to forecast the 8th day at arbitrary timestamps.

---

## Problem Statement

GNSS satellites broadcast two types of errors:
- **Ephemeris errors** — difference between broadcast and precise satellite position (X, Y, Z in meters)
- **Clock errors** — difference between broadcast and precise satellite clock offset (meters)

**Goal:** Train on 7 days of these errors → predict day 8 at arbitrary timestamps.

**Evaluation metric:** Shapiro-Wilk W statistic on residuals (actual − predicted),
averaged over all 4 error columns. Higher W = more Gaussian residuals = better model.
Benchmark: **W = 0.9810, p = 0.5840**

---

## Dataset

| File | Satellite | Type | Train rows (clean) | Test rows (clean) |
|---|---|---|---|---|
| DATA_GEO_Train.csv | Satellite A | GEO | 142 | 69 |
| DATA_MEO_Train.csv | Satellite B | MEO | 46 | 6 |
| DATA_MEO_Train2.csv | Satellite C | MEO | 143 | 18 |

**Key data characteristics discovered:**
- Non-uniform sampling (gaps from 1 min to 1,556 min) → rules out fixed-step LSTM
- MEO Train had 101 exact duplicate rows (removed before modeling)
- GEO satellite has two upload modes: 120-min (days 1–6) and 15-min (day 7+)
- MEO2 has four ~24-hour data gaps causing extrapolation challenges
- GEO test day has upload-boundary spikes of ±35–58m (operationally driven)

---

## Why Gaussian Process Regression?

| Criterion | LSTM | Gaussian Process |
|---|---|---|
| Data size needed | Thousands of windows | Works with tens of rows |
| Non-uniform sampling | ❌ Requires fixed steps | ✅ Uses continuous time |
| Arbitrary test timestamps | ❌ Must retrain | ✅ Predicts at any point |
| Evaluation metric (SW test) | Harder to optimise | ✅ Designed for normal posteriors |
| Runtime | Minutes–hours | Seconds–minutes |

---

## Model Architecture

One **Gaussian Process per (satellite, error column)** = 12 GPs total.

**Input:** `t_min` — continuous time in minutes since training start (scaled)

**Output:** x_error, y_error, z_error, clock_error (all in meters)

### Kernel Design

```
GEO kernel:
  ConstantKernel × RBF(length_scale~0.5)          # smooth drift
+ ConstantKernel × ExpSineSquared(period=1440min)  # 24h daily cycle
+ WhiteKernel                                       # upload-spike noise

MEO kernel:
  ConstantKernel × RBF(length_scale~0.5)          # smooth trend
+ ConstantKernel × ExpSineSquared(period=720min)   # 12h orbital period
+ ConstantKernel × ExpSineSquared(period=1440min)  # 24h solar pressure
+ WhiteKernel                                       # measurement noise
```

---

## Results

### Shapiro-Wilk Scores (Priority 1 — primary metric)

| Satellite | SW\_W | SW\_p | H0 rejected | vs Benchmark |
|---|---|---|---|---|
| GEO | 0.7865 | 0.0000 | 1.00 | −0.1945 |
| **MEO1** | **0.9084** | **0.4955** | **0.00 ✓** | **−0.0726** |
| MEO2 | 0.8076 | 0.0325 | 0.75 | −0.1734 |
| **Grand average** | **0.8342** | **0.1760** | | |
| Benchmark | 0.9810 | 0.5840 | 0.00 | |

### Residual Statistics (Priority 2)

| Satellite | Mean residual | Std residual | RMSE |
|---|---|---|---|
| GEO | +0.3769 m | 14.99 m | 15.03 m |
| MEO1 | −0.0032 m | 0.13 m | 0.15 m |
| MEO2 | −0.0247 m | 0.15 m | 0.16 m |

### Why GEO is limited

The GEO test day contains upload-boundary spikes of ±35–58m.
For **smooth rows** (47 of 69 test rows, |error| < 10m):
- y\_error SW\_W = **0.9828** (exceeds benchmark)
- z\_error SW\_W = **0.9546** (near benchmark)

The spikes are operationally driven ground-segment decisions,
not predictable from orbital mechanics. This is a data limitation,
not a modeling failure.

---

## Project Structure

```
GNSS-Error-prediction/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── imports.py              # All imports and constants
│   ├── phase0\_1\_eda.py         # EDA and data inspection
│   ├── phase2\_preprocessing.py  # Cleaning, features, scaling
│   ├── phase3\_baselines.py      # Persistence, linear, mean baselines
│   ├── phase4\_gp\_model.py       # First GP model
│   ├── phase5\_tuning.py         # Kernel comparison + TSCV
│   ├── phase6\_improvements.py   # Per-satellite final improvements
│   ├── phase7\_predict.py        # Final submission script (single command)
│   ├── phase8\_evaluate.py       # SW evaluation + all plots
│   └── phase9\_github.py         # This file
│
├── Data/
│   ├── Raw/                    # Original CSVs (not committed if large)
│   └── Processed/              # Cleaned + featured CSVs (auto-generated)
│
├── results/
│   ├── final\_predictions\_geo.csv
│   ├── final\_predictions\_meo1.csv
│   ├── final\_predictions\_meo2.csv
│   ├── submission\_sw\_report.csv
│   └── submission\_summary.txt
│
├── figures/
│   ├── phase8\_qq\_geo.png
│   ├── phase8\_qq\_meo1.png
│   ├── phase8\_residual\_hist\_all.png
│   ├── phase8\_final\_comparison.png
│   └── phase8\_prediction\_vs\_actual.png
│
└── report/
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GNSS-Error-prediction.git
cd GNSS-Error-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run (Full Pipeline)

```bash
# Phase 0+1: EDA and data inspection
python src/phase0_1_eda.py

# Phase 2: Preprocessing
python src/phase2_preprocessing.py

# Phase 3: Baseline models
python src/phase3_baselines.py

# Phase 4: First GP model
python src/phase4_gp_model.py

# Phase 5: Kernel tuning
python src/phase5_tuning.py

# Phase 6: Per-satellite improvements
python src/phase6_improvements.py

# Phase 7: Self-test (verify predictions on known data)
python src/phase7_predict.py

# Phase 8: Final evaluation and report
python src/phase8_evaluate.py
```

---

## Final Evaluation (Single Command)

When the evaluation committee provides new training data and timestamps:

```bash
# GEO satellite
python src/phase7_predict.py \
    --train  NEW_GEO_Train.csv \
    --times  test_timestamps.txt \
    --output results/submission_geo.csv \
    --type   GEO

# MEO satellite (auto-detects type if --type omitted)
python src/phase7_predict.py \
    --train  NEW_MEO_Train.csv \
    --times  test_timestamps.txt \
    --output results/submission_meo.csv
```

**Timestamp file format** (one per line):
```
2025-09-08 00:11:00
2025-09-08 00:24:00
...
```

**Runtime:** ~40–60 seconds total for all satellites.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥2.0 | Data loading and manipulation |
| numpy | ≥1.24 | Numerical computing |
| scipy | ≥1.10 | Shapiro-Wilk test, statistics |
| scikit-learn | ≥1.2 | Gaussian Process, StandardScaler |
| matplotlib | ≥3.7 | All visualisations |
| seaborn | ≥0.12 | Correlation heatmaps |
| statsmodels | ≥0.14 | ADF/KPSS stationarity tests |
| joblib | ≥1.2 | Save/load fitted scalers |
| tabulate | ≥0.9 | Terminal tables |

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Model | Gaussian Process | Only 46–143 training rows; non-uniform sampling |
| Input feature | t\_min (continuous) | GP needs scalar distances, not datetime strings |
| Outlier treatment | 3×IQR winsorize (train only) | Removes spikes without destroying real signal |
| GEO training data | Sep 6–7 only (mode filter) | Sep 8 test is 15-min upload mode; Sep 1–5 is 120-min |
| MEO2 training data | Full 143 rows | Better than segment; needed for periodic learning |
| Kernel (GEO) | RBF + Per(24h) + White | Daily cycle dominates GEO error structure |
| Kernel (MEO) | RBF + Per(12h) + Per(24h) + White | Orbital + solar periods both present |
| Scaling | StandardScaler (train only) | Prevents data leakage; normalises GP inputs |
| CV method | TimeSeriesSplit | No future leakage; mirrors real use case |

---

## Q-Q Plots (Priority 3)

See `figures/` folder:

| File | Satellite | Notes |
|---|---|---|
| phase8\_qq\_geo.png | GEO | S-curves on clock\_error = upload spikes |
| phase8\_qq\_meo1.png | MEO1 | Points on diagonal → near-normal ✓ |
| phase8\_qq\_meo2.png | MEO2 | y\_error clean; clock/z show gap effects |

---

*Generated: July 14, 2026*
*SIH 2025 — ISRO GNSS Error Prediction Challenge*
