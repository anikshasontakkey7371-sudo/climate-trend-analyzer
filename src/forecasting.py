"""
forecasting.py
==============
Educational time-series forecasting for climate variables using
Exponential Smoothing (Holt-Winters) and Harmonic Regression.

Educational & Scientific Note:
-----------------------------
This module demonstrates statistical pattern extrapolation for learning
and demonstration purposes. Statistical time-series forecasts do NOT account
for complex Earth-system feedbacks, radiative forcing scenarios (SSPs/RCPs),
or coupled ocean-atmosphere dynamics used in official IPCC climate projections.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def train_test_forecast_split(
    df: pd.DataFrame,
    test_periods: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split holding out the most recent periods for validation.
    """
    if len(df) <= test_periods * 2:
        raise ValueError("Dataset is too short for the requested test split.")
    train_df = df.iloc[:-test_periods].copy()
    test_df = df.iloc[-test_periods:].copy()
    return train_df, test_df


def fit_holt_winters_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "temperature_c",
    forecast_horizon: int = 24,
    seasonal_periods: int = 12,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    Fit Holt-Winters Exponential Smoothing with additive trend and additive seasonality.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame
        Test data for validation.
    target_col : str
        Target variable (default 'temperature_c').
    forecast_horizon : int
        Number of future steps beyond test set to project.
    seasonal_periods : int
        Number of steps per seasonal cycle (12 for monthly data).

    Returns
    -------
    Tuple[pd.Series, pd.Series, Dict[str, float]]
        test_predictions, future_forecast, and evaluation metrics dictionary.
    """
    train_series = train_df[target_col].values
    test_series = test_df[target_col].values
    n_test = len(test_series)

    # Fit Holt-Winters model
    model = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fitted_model = model.fit(optimized=True)

    # Predict test period
    test_preds = fitted_model.forecast(n_test)

    # Metrics on hold-out test set
    mae = float(mean_absolute_error(test_series, test_preds))
    rmse = float(np.sqrt(mean_squared_error(test_series, test_preds)))
    # Avoid zero division in MAPE
    non_zero = test_series != 0
    mape = float(np.mean(np.abs((test_series[non_zero] - test_preds[non_zero]) / test_series[non_zero])) * 100)
    r2 = float(r2_score(test_series, test_preds))

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_pct": round(mape, 2),
        "R2": round(r2, 3),
        "model_type": "Holt-Winters Exponential Smoothing (Additive)",
    }

    # Now refit on full data for future projection
    full_series = np.concatenate([train_series, test_series])
    full_model = ExponentialSmoothing(
        full_series,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)

    future_preds = full_model.forecast(forecast_horizon)

    return pd.Series(test_preds, index=test_df.index), pd.Series(future_preds), metrics


def generate_forecast_dates(last_date: pd.Timestamp, horizon: int, freq: str = "MS") -> pd.DatetimeIndex:
    """Generate future date index starting one step after last historical date."""
    return pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq=freq)


def create_forecast_chart(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_preds: pd.Series,
    future_preds: pd.Series,
    future_dates: pd.DatetimeIndex,
    metrics: Dict[str, float],
    target_col: str = "temperature_c",
) -> go.Figure:
    """Create interactive Plotly forecast chart with historical actuals, test validation, and future projection."""
    fig = go.Figure()

    # Historical Training Data (last 5 years for visual clarity)
    recent_train = train_df.tail(60)
    fig.add_trace(
        go.Scatter(
            x=recent_train["date"],
            y=recent_train[target_col],
            mode="lines",
            name="Historical (Training)",
            line=dict(color="#64748b", width=1.5),
        )
    )

    # Test Actuals
    fig.add_trace(
        go.Scatter(
            x=test_df["date"],
            y=test_df[target_col],
            mode="lines+markers",
            name="Validation Actuals",
            line=dict(color="#1e293b", width=2),
            marker=dict(size=6, color="#0f172a"),
        )
    )

    # Test Model Predictions
    fig.add_trace(
        go.Scatter(
            x=test_df["date"],
            y=test_preds.values,
            mode="lines+markers",
            name=f"Validation Forecast (RMSE: {metrics['RMSE']:.2f}°C)",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=6, color="#d97706"),
        )
    )

    # Future Projection
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_preds.values,
            mode="lines+markers",
            name="Future Horizon Projection",
            line=dict(color="#8b5cf6", width=2.5),
            marker=dict(size=5, color="#7c3aed"),
        )
    )

    # Estimated Uncertainty Ribbon around Future Projection
    rmse = metrics["RMSE"]
    upper_bound = future_preds.values + 1.96 * rmse
    lower_bound = future_preds.values - 1.96 * rmse

    fig.add_trace(
        go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill="toself",
            fillcolor="rgba(139, 92, 246, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Band",
        )
    )

    fig.update_layout(
        title=f"Educational Climate Forecast ({metrics['model_type']} | Test MAE: {metrics['MAE']}°C)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Temperature (°C)"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
