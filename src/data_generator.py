"""
data_generator.py
=================
Generates reproducible, physics-informed synthetic climate datasets
for research, education, benchmarking, and demonstration purposes.

Scientific Note:
----------------
All generated records are marked with `is_synthetic = True`.
Simulated trends and anomalies are controlled for educational purposes
and do NOT represent measured climate observations for any specific real-world station.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def get_season(month: int) -> str:
    """Map calendar month (1-12) to a meteorological season."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Autumn"


def generate_climate_data(
    start_year: int = 1990,
    end_year: int = 2023,
    freq: str = "MS",  # Month Start
    base_temp: float = 16.5,
    temp_annual_amp: float = 9.5,
    temp_trend_per_decade: float = 0.28,
    base_rainfall: float = 65.0,
    rainfall_seasonal_amp: float = 45.0,
    noise_std_temp: float = 1.1,
    anomaly_fraction: float = 0.03,
    missing_fraction: float = 0.015,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic climate time-series DataFrame.

    Parameters
    ----------
    start_year : int
        Starting calendar year.
    end_year : int
        Ending calendar year (inclusive).
    freq : str
        Frequency string ('MS' for Month Start, 'D' for Daily).
    base_temp : float
        Mean baseline temperature in degrees Celsius.
    temp_annual_amp : float
        Amplitude of seasonal temperature cycle in degrees Celsius.
    temp_trend_per_decade : float
        Long-term warming trend in degrees Celsius per decade.
    base_rainfall : float
        Base monthly rainfall in millimeters.
    rainfall_seasonal_amp : float
        Amplitude of seasonal precipitation cycle in millimeters.
    noise_std_temp : float
        Standard deviation of Gaussian noise for temperature.
    anomaly_fraction : float
        Proportion of observations injected with extreme anomalies.
    missing_fraction : float
        Proportion of observations with simulated sensor missingness (NaN).
    random_seed : int
        Seed for NumPy random generator to ensure exact reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with date, temperature_c, rainfall_mm, humidity_pct,
        calendar features, and ground-truth anomaly/synthetic flags.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Date Range Generation
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_steps = len(dates)

    # Time in years elapsed from start
    years_elapsed = np.array((dates - dates[0]).days / 365.25, dtype=float)

    # 2. Temperature Simulation:
    # Baseline + Decadal Trend + Annual Sinusoid + Autoregressive (AR1) noise
    warming_trend = (temp_trend_per_decade / 10.0) * years_elapsed

    # Seasonal sinusoidal cycle (peak around July / Month 7 in Northern Hemisphere)
    day_of_year = np.array(dates.dayofyear, dtype=float)
    seasonal_cycle = temp_annual_amp * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)

    # AR(1) short-term weather persistence
    ar_noise = np.zeros(n_steps)
    phi = 0.35  # persistence coefficient
    white_noise = rng.normal(0, noise_std_temp, size=n_steps)
    for i in range(1, n_steps):
        ar_noise[i] = phi * ar_noise[i - 1] + white_noise[i]

    temperature = base_temp + warming_trend + seasonal_cycle + ar_noise

    # 3. Rainfall Simulation:
    # Seasonal peak (e.g. Summer/Late Summer storms or monsoon) + Gamma distribution
    rain_seasonality = np.maximum(
        0.1,
        1.0 + (rainfall_seasonal_amp / base_rainfall) * np.sin(2 * np.pi * (day_of_year - 120) / 365.25),
    )
    # Gamma distribution parameters: shape (k) and scale (theta)
    k_shape = 2.5
    theta_scale = (base_rainfall * rain_seasonality) / k_shape
    rainfall = rng.gamma(shape=k_shape, scale=theta_scale)

    # 4. Relative Humidity Simulation (%):
    # Physically coupled: higher with rainfall, inversely coupled with dry heat
    base_humidity = 62.0
    humid_from_rain = 0.22 * rainfall
    humid_from_temp = -0.65 * (temperature - base_temp)
    humid_noise = rng.normal(0, 4.0, size=n_steps)
    humidity = base_humidity + humid_from_rain + humid_from_temp + humid_noise
    humidity = np.clip(humidity, 15.0, 98.0)

    # 5. Inject Controlled Anomalies with Ground Truth Labels
    anomaly_flags = np.zeros(n_steps, dtype=bool)
    n_anomalies = max(2, int(n_steps * anomaly_fraction))
    anomaly_indices = rng.choice(n_steps, size=n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_flags[idx] = True
        anomaly_type = rng.choice(["heatwave", "cold_snap", "extreme_downpour"])
        if anomaly_type == "heatwave":
            temperature[idx] += rng.uniform(4.5, 7.5)
            humidity[idx] = max(18.0, humidity[idx] - 15.0)
        elif anomaly_type == "cold_snap":
            temperature[idx] -= rng.uniform(4.0, 7.0)
        elif anomaly_type == "extreme_downpour":
            rainfall[idx] += rng.uniform(110.0, 190.0)
            humidity[idx] = min(98.0, humidity[idx] + 20.0)

    # 6. Construct Clean Base DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "temperature_c": np.round(temperature, 2),
            "rainfall_mm": np.round(rainfall, 2),
            "humidity_pct": np.round(humidity, 1),
            "year": dates.year,
            "month": dates.month,
            "month_name": dates.strftime("%B"),
            "season": [get_season(m) for m in dates.month],
            "is_anomaly_ground_truth": anomaly_flags,
            "is_synthetic": True,
        }
    )

    # 7. Inject Controlled Missing Values for Preprocessing Demonstrations
    if missing_fraction > 0.0:
        n_missing = int(n_steps * missing_fraction)
        if n_missing > 0:
            missing_temp_idx = rng.choice(n_steps, size=n_missing, replace=False)
            missing_rain_idx = rng.choice(n_steps, size=n_missing, replace=False)
            df.loc[missing_temp_idx, "temperature_c"] = np.nan
            df.loc[missing_rain_idx, "rainfall_mm"] = np.nan

    return df


def save_synthetic_sample(
    output_path: Optional[Path] = None,
    random_seed: int = 42,
) -> Path:
    """
    Generate and save the default monthly synthetic climate dataset to data/sample/.

    Parameters
    ----------
    output_path : Optional[Path]
        Target CSV file path. If None, saves to 'data/sample/climate_monthly_synthetic.csv'.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    Path
        Absolute path to the saved CSV file.
    """
    if output_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_path = base_dir / "data" / "sample" / "climate_monthly_synthetic.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_climate_data(
        start_year=1990,
        end_year=2023,
        freq="MS",
        random_seed=random_seed,
    )
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved_file = save_synthetic_sample()
    print(f"Successfully generated synthetic sample dataset at: {saved_file}")
    sample_df = pd.read_csv(saved_file)
    print(f"Dataset shape: {sample_df.shape[0]} rows, {sample_df.shape[1]} columns")
    print("Preview:\n", sample_df.head(3))
