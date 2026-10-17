"""
eda.py
======
Exploratory Data Analysis (EDA) module for climate data.
Computes descriptive statistics, annual/monthly/seasonal summaries,
correlation analysis, and produces both Matplotlib and Plotly figures.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for numeric climate variables.
    Includes Mean, Std Dev, Min, 25%, Median, 75%, Max, Skewness.
    """
    numeric_cols = [c for c in ["temperature_c", "rainfall_mm", "humidity_pct"] if c in df.columns]
    desc = df[numeric_cols].describe().T
    desc["skewness"] = df[numeric_cols].skew()
    desc = desc.round(2)
    desc = desc.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std Dev",
            "min": "Min",
            "25%": "25th Pct",
            "50%": "Median",
            "75%": "75th Pct",
            "max": "Max",
            "skewness": "Skewness",
        }
    )
    return desc


def compute_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual aggregate metrics:
    - Mean Temperature (°C)
    - Total Rainfall (mm)
    - Mean Humidity (%)
    """
    agg_dict = {"temperature_c": "mean"}
    if "rainfall_mm" in df.columns:
        agg_dict["rainfall_mm"] = "sum"
    if "humidity_pct" in df.columns:
        agg_dict["humidity_pct"] = "mean"

    yearly = df.groupby("year").agg(agg_dict).round(2).reset_index()
    yearly = yearly.rename(
        columns={
            "temperature_c": "mean_temp_c",
            "rainfall_mm": "total_rainfall_mm",
            "humidity_pct": "mean_humidity_pct",
        }
    )
    return yearly


def compute_monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute climatological baseline for each calendar month (Jan - Dec)
    averaged across the entire historical period.
    """
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    climatology = df.groupby(["month", "month_name"]).agg(
        temp_mean=("temperature_c", "mean"),
        temp_std=("temperature_c", "std"),
        temp_min=("temperature_c", "min"),
        temp_max=("temperature_c", "max"),
        rain_mean=("rainfall_mm", "mean") if "rainfall_mm" in df.columns else ("temperature_c", "count"),
        rain_std=("rainfall_mm", "std") if "rainfall_mm" in df.columns else ("temperature_c", "count"),
    ).round(2).reset_index()

    # Re-order by month index
    climatology = climatology.sort_values(by="month").reset_index(drop=True)
    return climatology


def compute_seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize temperature and rainfall distributions across meteorological seasons.
    """
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    seasonal = df.groupby("season").agg(
        mean_temperature=("temperature_c", "mean"),
        std_temperature=("temperature_c", "std"),
        mean_rainfall=("rainfall_mm", "mean") if "rainfall_mm" in df.columns else ("temperature_c", "count"),
        total_rainfall=("rainfall_mm", "sum") if "rainfall_mm" in df.columns else ("temperature_c", "count"),
        records_count=("temperature_c", "count"),
    ).round(2)

    # Reindex for intuitive seasonal flow
    seasonal = seasonal.reindex([s for s in season_order if s in seasonal.index]).reset_index()
    return seasonal


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix between climate variables."""
    numeric_cols = [c for c in ["temperature_c", "rainfall_mm", "humidity_pct"] if c in df.columns]
    return df[numeric_cols].corr().round(3)


# --- Plotly Interactive Visualizations (for Streamlit & Notebooks) ---

def create_monthly_climatology_chart(climatology_df: pd.DataFrame) -> go.Figure:
    """Create interactive dual-axis chart for monthly temperature and rainfall climatology."""
    fig = go.Figure()

    # Rainfall Bars
    if "rain_mean" in climatology_df.columns:
        fig.add_trace(
            go.Bar(
                x=climatology_df["month_name"],
                y=climatology_df["rain_mean"],
                name="Mean Rainfall (mm)",
                marker_color="#3b82f6",
                opacity=0.75,
                yaxis="y2",
            )
        )

    # Temperature Line
    fig.add_trace(
        go.Scatter(
            x=climatology_df["month_name"],
            y=climatology_df["temp_mean"],
            name="Mean Temp (°C)",
            mode="lines+markers",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8, color="#dc2626"),
            yaxis="y1",
        )
    )

    fig.update_layout(
        title="Climatological Monthly Baseline (Annual Cycle)",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Temperature (°C)", side="left", showgrid=True),
        yaxis2=dict(
            title="Rainfall (mm)",
            side="right",
            overlaying="y",
            showgrid=False,
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.6)"),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_seasonal_boxplot(df: pd.DataFrame) -> go.Figure:
    """Create interactive seasonal temperature boxplots."""
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    fig = px.box(
        df,
        x="season",
        y="temperature_c",
        color="season",
        category_orders={"season": season_order},
        title="Temperature Distribution by Meteorological Season",
        labels={"season": "Season", "temperature_c": "Temperature (°C)"},
        color_discrete_map={
            "Winter": "#60a5fa",
            "Spring": "#34d399",
            "Summer": "#f87171",
            "Autumn": "#fbbf24",
        },
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def create_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Create interactive correlation heatmap."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title="Climate Variables Correlation Matrix (Pearson)",
        labels=dict(color="Correlation"),
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


# --- Matplotlib Static Export for CLI & Reports ---

def save_eda_static_figures(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Path:
    """Generate and save publication-quality static EDA charts."""
    if output_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_dir = base_dir / "outputs" / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Temperature Distribution & KDE
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["temperature_c"], kde=True, color="#e11d48", ax=ax, bins=25)
    ax.set_title("Temperature Distribution with Kernel Density Estimate", fontsize=12, fontweight="bold")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_distribution.png", dpi=300)
    plt.close()

    # 2. Monthly Boxplot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(x="month", y="temperature_c", data=df, hue="month", palette="coolwarm", legend=False, ax=ax)
    ax.set_title("Monthly Temperature Variations Across All Years", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month of Year")
    ax.set_ylabel("Temperature (°C)")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_temperature_variation.png", dpi=300)
    plt.close()

    return output_dir
