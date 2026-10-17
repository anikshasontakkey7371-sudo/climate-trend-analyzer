"""
trend_analysis.py
=================
Quantifies long-term climatic trends through rolling averages,
Ordinary Least Squares (OLS) regression, and Mann-Kendall monotonic trend tests.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression


def compute_rolling_averages(
    df: pd.DataFrame,
    windows: Tuple[int, ...] = (12, 36),
) -> pd.DataFrame:
    """
    Compute rolling averages and rolling standard deviations for climate variables.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned climate DataFrame sorted by date.
    windows : Tuple[int, ...]
        Window lengths in steps (e.g., 12 for 12-month, 36 for 3-year rolling average).

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with rolling mean and rolling std columns.
    """
    res = df.copy()
    target_cols = [c for c in ["temperature_c", "rainfall_mm"] if c in res.columns]

    for col in target_cols:
        for w in windows:
            res[f"{col}_roll_{w}"] = res[col].rolling(window=w, min_periods=max(1, w // 2)).mean().round(2)
            res[f"{col}_roll_std_{w}"] = res[col].rolling(window=w, min_periods=max(1, w // 2)).std().round(2)

    return res


def estimate_linear_trend(
    df: pd.DataFrame,
    target_col: str = "temperature_c",
) -> Dict[str, float]:
    """
    Calculate OLS linear trend, slope per decade, R², p-value, and standard error.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'date' and target_col.
    target_col : str
        Column to regress against time (default 'temperature_c').

    Returns
    -------
    dict
        Trend statistics including slope_per_year, slope_per_decade,
        intercept, r_squared, p_value, std_err, and trend_significance.
    """
    valid = df.dropna(subset=["date", target_col]).copy()
    if len(valid) < 3:
        raise ValueError("Insufficient data points to fit linear trend.")

    # Time represented in years from start date
    t_start = valid["date"].iloc[0]
    years_elapsed = np.array((valid["date"] - t_start).dt.days / 365.25, dtype=float)
    y = valid[target_col].values

    # Scipy linregress for rigorous p-values and standard errors
    slope, intercept, r_value, p_value, std_err = stats.linregress(years_elapsed, y)

    slope_per_decade = slope * 10.0
    r_squared = r_value ** 2

    # Scikit-learn trendline coordinates for plotting
    trend_fitted = intercept + slope * years_elapsed

    is_significant = p_value < 0.05

    # Non-parametric Mann-Kendall tau test using scipy.stats.kendalltau
    tau, mk_p_value = stats.kendalltau(years_elapsed, y)

    return {
        "slope_per_year": round(float(slope), 4),
        "slope_per_decade": round(float(slope_per_decade), 4),
        "intercept": round(float(intercept), 3),
        "r_squared": round(float(r_squared), 4),
        "p_value": float(p_value),
        "std_err": round(float(std_err), 4),
        "is_statistically_significant": is_significant,
        "mann_kendall_tau": round(float(tau), 4),
        "mann_kendall_p_value": float(mk_p_value),
        "trend_direction": "Warming" if slope > 0 else "Cooling",
        "fitted_values": trend_fitted,
    }


def compute_monthly_temperature_anomalies(
    df: pd.DataFrame,
    baseline_years: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Calculate monthly temperature anomalies relative to a baseline climatology period.

    Parameters
    ----------
    df : pd.DataFrame
        Input climate DataFrame with 'date', 'month', and 'temperature_c'.
    baseline_years : Optional[Tuple[int, int]]
        Start and end years for the reference baseline. If None, uses first 15 years.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'temp_anomaly_c' and 'temp_baseline_c' columns.
    """
    res = df.copy()

    if baseline_years is None:
        min_yr = res["year"].min()
        baseline_years = (min_yr, min(min_yr + 14, res["year"].max()))

    baseline_mask = (res["year"] >= baseline_years[0]) & (res["year"] <= baseline_years[1])
    baseline_df = res[baseline_mask]

    if baseline_df.empty:
        baseline_df = res

    # Monthly baseline means
    monthly_baselines = baseline_df.groupby("month")["temperature_c"].mean().to_dict()

    res["temp_baseline_c"] = res["month"].map(monthly_baselines).round(2)
    res["temp_anomaly_c"] = (res["temperature_c"] - res["temp_baseline_c"]).round(2)
    return res


# --- Plotly Interactive Charts ---

def create_temperature_trend_chart(
    df: pd.DataFrame,
    trend_stats: Dict[str, float],
) -> go.Figure:
    """Create interactive temperature time series with rolling average and OLS trendline."""
    fig = go.Figure()

    # Raw observed series
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["temperature_c"],
            mode="lines",
            name="Monthly Temperature",
            line=dict(color="#cbd5e1", width=1),
            opacity=0.7,
        )
    )

    # 12-Month Rolling Mean
    if "temperature_c_roll_12" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["temperature_c_roll_12"],
                mode="lines",
                name="12-Month Moving Avg",
                line=dict(color="#2563eb", width=2.5),
            )
        )

    # OLS Trendline
    if "fitted_values" in trend_stats:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=trend_stats["fitted_values"],
                mode="lines",
                name=f"OLS Trend (+{trend_stats['slope_per_decade']:.2f}°C/decade)",
                line=dict(color="#dc2626", width=2.5, dash="dash"),
            )
        )

    sig_text = "p < 0.05 (Statistically Significant)" if trend_stats.get("is_statistically_significant") else "Not Significant (p >= 0.05)"
    fig.update_layout(
        title=f"Long-Term Temperature Trend (Rate: {trend_stats['slope_per_decade']:+.2f}°C / decade | {sig_text})",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Temperature (°C)"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_annual_anomaly_bars(yearly_df: pd.DataFrame) -> go.Figure:
    """Create annual temperature anomaly bar chart with warm/cool diverging colors."""
    overall_mean = yearly_df["mean_temp_c"].mean()
    yearly_df = yearly_df.copy()
    yearly_df["anomaly"] = (yearly_df["mean_temp_c"] - overall_mean).round(2)
    yearly_df["color"] = np.where(yearly_df["anomaly"] >= 0, "#ef4444", "#3b82f6")

    fig = go.Figure(
        go.Bar(
            x=yearly_df["year"],
            y=yearly_df["anomaly"],
            marker_color=yearly_df["color"],
            name="Annual Anomaly",
        )
    )

    fig.update_layout(
        title=f"Annual Temperature Anomaly Relative to {overall_mean:.1f}°C Period Average",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Anomaly (°C)"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_rainfall_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create interactive rainfall chart with 12-month rolling sum/mean."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["rainfall_mm"],
            name="Monthly Rainfall (mm)",
            marker_color="#93c5fd",
            opacity=0.65,
        )
    )

    if "rainfall_mm_roll_12" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rainfall_mm_roll_12"],
                mode="lines",
                name="12-Month Moving Avg",
                line=dict(color="#1d4ed8", width=2.5),
            )
        )

    fig.update_layout(
        title="Monthly Rainfall and 12-Month Rolling Average",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Rainfall (mm)"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


# --- Matplotlib Static Export ---

def save_trend_static_figures(
    df: pd.DataFrame,
    trend_stats: Dict[str, float],
    output_dir: Optional[Path] = None,
) -> Path:
    """Save static trend charts for reporting."""
    if output_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_dir = base_dir / "outputs" / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["date"], df["temperature_c"], color="#94a3b8", label="Observed Monthly Temp", alpha=0.6, lw=0.9)
    if "temperature_c_roll_12" in df.columns:
        ax.plot(df["date"], df["temperature_c_roll_12"], color="#1d4ed8", label="12-Month Moving Avg", lw=2)
    if "fitted_values" in trend_stats:
        ax.plot(df["date"], trend_stats["fitted_values"], color="#dc2626", linestyle="--", label=f"OLS Trend ({trend_stats['slope_per_decade']:+.2f}°C/dec)", lw=2)

    ax.set_title("Temperature Trend and Rolling Average", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_trend_line.png", dpi=300)
    plt.close()

    return output_dir
