# 🌍 Climate Trend Analyzer

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industry-style, beginner-friendly Python data science platform for analyzing historical climate records, discovering temperature trajectories, detecting unusual extreme anomalies, visualizing seasonal patterns, and projecting short-term educational forecasts.

Designed as a showcase student portfolio project for GitHub, internships, placements, and technical data science interviews.

---

## 🌟 Key Features

1. **Dual Data Sourcing**:
   - Out-of-the-box **Reproducible Synthetic Climate Generator** with physical seasonality, long-term warming trends, autoregressive weather noise, and controlled anomalies.
   - Support for **Custom User CSV Uploads** with automatic column mapping (`Date`, `Temperature`, `Rainfall`, `Humidity`).

2. **Rigorous Data Preprocessing**:
   - Automatic column alias resolution and chronological sorting.
   - Comprehensive data quality audit (missing counts/percentages, duplicate timestamp removal).
   - Configurable imputation strategies (`linear_interpolation`, `seasonal_median`, `forward_fill`, `drop`).
   - Physical boundary bounds enforcement (non-negative rainfall, $[0\%, 100\%]$ relative humidity).

3. **Statistical Trend Quantification**:
   - 12-month and 36-month moving averages with confidence intervals.
   - Ordinary Least Squares (OLS) linear regression reporting rate per decade ($^\circ\text{C}/\text{decade}$), $R^2$, $p$-values, and standard error.
   - Non-parametric **Mann-Kendall** rank correlation trend test.
   - Monthly climatological anomaly baseline calculations.

4. **Dual-Method Anomaly Detection**:
   - **Statistical Engine**: Climatological month-specific Z-Score and Interquartile Range (IQR).
   - **Machine Learning Engine**: Unsupervised **Isolation Forest** tree ensembles over multidimensional climate space.
   - Plain-English anomaly classifications (*Heatwave*, *Cold Snap*, *Extreme Downpour*).
   - Evaluation benchmarking (Precision, Recall, F1) against injected synthetic ground truth.

5. **Educational Time-Series Forecasting**:
   - Holt-Winters Seasonal Exponential Smoothing with holdout validation (last 12 months held out for testing).
   - Rigorous evaluation metrics: MAE, RMSE, MAPE %, and $R^2$.
   - Interactive projection horizon with $95\%$ prediction uncertainty bands.

6. **Interactive Streamlit Dashboard & Export Center**:
   - Modern, responsive scientific UI with KPI metric cards, Plotly interactive charts, and data filters.
   - One-click exports of cleaned CSV datasets, detected anomalies, and executive markdown reports.

---

## 📁 Repository Structure

```text
Climate-Trend-Analyzer/
├── app/
│   ├── __init__.py
│   └── dashboard.py               # Interactive Streamlit Web Dashboard
├── data/
│   ├── raw/                       # Raw user-supplied climate files
│   ├── sample/                    # Pre-generated reproducible synthetic monthly CSV
│   └── processed/                 # Cleaned, imputed, and feature-engineered records
├── docs/
│   └── architecture.md            # Mathematical methods & system design
├── outputs/
│   ├── figures/                   # High-res static figures (PNG)
│   ├── tables/                    # Exported CSV summaries and anomaly logs
│   └── reports/                   # Automated Markdown executive reports
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Physics-informed synthetic climate simulation
│   ├── preprocessing.py           # Data quality audit, deduplication, imputation
│   ├── eda.py                     # Climatology, seasonal summaries, correlations
│   ├── trend_analysis.py          # Moving averages, OLS regression, Mann-Kendall
│   ├── anomaly_detection.py       # Z-score + Isolation Forest dual detection
│   ├── forecasting.py             # Holt-Winters backtesting & future projection
│   └── insights.py                # Plain-language narrative findings generator
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py      # Unit tests for cleaning and boundary checks
│   └── test_analysis.py           # Unit tests for trends, anomalies, and forecasting
├── main.py                        # CLI pipeline runner (executes end-to-end)
├── requirements.txt               # Pinned project dependencies
├── .gitignore                     # Git ignore rules for caches & virtual environments
└── README.md                      # Comprehensive project guide
```

---

## 🚀 Quick Start Guide

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/climate-trend-analyzer.git
cd climate-trend-analyzer
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# On Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Automated CLI Pipeline
To run the full data generation, cleaning, analysis, and report generation via command line:
```bash
python main.py
```
This generates cleaned datasets in `data/processed/`, tables in `outputs/tables/`, figures in `outputs/figures/`, and a markdown report in `outputs/reports/`.

### Step 5: Launch the Interactive Streamlit Dashboard
```bash
# Recommended on Windows (avoids PATH lookup issues):
python -m streamlit run app/dashboard.py

# Or directly if Streamlit is in your PATH:
streamlit run app/dashboard.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

### Step 6: Run the Automated Tests
```bash
python -m pytest -v
```

---

## 🔬 Scientific Integrity & Disclaimers

- **Simulated vs. Measured Data**: All generated datasets include an `is_synthetic = True` flag. Synthetic data serves as an educational and testing benchmark and must never be represented as actual station observations.
- **Correlation vs. Causation**: Local correlations between climate variables do not imply standalone causation.
- **Forecasting Scope**: Statistical extrapolations illustrate historical pattern continuation for learning purposes; they do not simulate physical greenhouse gas concentration pathways (IPCC SSPs).

---

## 💼 Interview & Portfolio Talking Points

When presenting this project to interviewers or hiring managers:

1. **System Architecture**: Explain the separation of concerns between preprocessing, statistical inference, machine learning, and visualization rather than relying on a single monolithic notebook.
2. **Data Quality First**: Mention how missing dates, duplicate timestamps, and physical bounds (e.g. non-negative rainfall) are audited before running models.
3. **Dual-Model Anomaly Detection**: Discuss why you chose both a domain-specific statistical test (month-by-month climatological Z-score) and an unsupervised ML model (Isolation Forest) to capture both univariate extremes and multivariate pattern anomalies.
4. **Validation Rigor**: Highlight chronological holdout backtesting (avoiding data leakage in time series) and benchmark evaluation (Precision, Recall, F1) against known synthetic ground truth.
