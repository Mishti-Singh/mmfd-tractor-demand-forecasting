import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==============================================================================
# 1. LOAD AND CLEAN DATA
# ==============================================================================
file_path = "/content/Part3_Industry_plus_MMFD.xlsx"

# Read the first two columns without forcing header assumption
df = pd.read_excel(file_path, usecols=[0, 1])

# Rename columns to standard names
df.columns = ["Month", "Demand"]

# Convert to numeric/datetime and strip out header text or instruction rows
df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

# Drop any rows where Month or Demand couldn't be parsed
df = df.dropna(subset=["Month", "Demand"]).copy()

# Sort by date and set Monthly Start index
df.sort_values("Month", inplace=True)
df.set_index("Month", inplace=True)
df.index.freq = "MS"

print("--- Historical Data Overview ---")
print(
    f"Data range: {df.index.min().strftime('%Y-%m')} to"
    f" {df.index.max().strftime('%Y-%m')}"
)
print(f"Total historical observations: {len(df)}\n")

# ==============================================================================
# 2. FIT HOLT-WINTERS MULTIPLICATIVE MODEL
# ==============================================================================
model = ExponentialSmoothing(
    df["Demand"],
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
    initialization_method="estimated",
)

fit_model = model.fit()
df["Fitted_Forecast"] = fit_model.fittedvalues

# ==============================================================================
# 3. GENERATE 12-MONTH FUTURE FORECAST
# ==============================================================================
forecast_horizon = 12
future_forecast = fit_model.forecast(forecast_horizon)

future_dates = pd.date_range(
    start=df.index.max() + pd.DateOffset(months=1),
    periods=forecast_horizon,
    freq="MS",
)
df_future = pd.DataFrame(
    {"Forecast": future_forecast.values}, index=future_dates
)

# ==============================================================================
# 4. CALCULATE METRICS
# ==============================================================================
errors = df["Demand"] - df["Fitted_Forecast"]
abs_errors = np.abs(errors)

mae = np.mean(abs_errors)
rmse = np.sqrt(np.mean(errors**2))
mape = np.mean(abs_errors / df["Demand"]) * 100

print("--- Model Accuracy Metrics ---")
print(f"Mean Absolute Error (MAE):       {mae:.2f}")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f}")
print(f"Mean Absolute Percentage (MAPE): {mape:.2f}%\n")

# ==============================================================================
# 5. VISUALIZATION & SAVE
# ==============================================================================
plt.figure(figsize=(12, 6), dpi=300)

plt.plot(
    df.index,
    df["Demand"],
    label="Actual MMFD Demand",
    color="#1f77b4",
    linewidth=2,
)
plt.plot(
    df.index,
    df["Fitted_Forecast"],
    label="Holt-Winters Fitted Model",
    color="#ff7f0e",
    linestyle="--",
    linewidth=1.8,
)
plt.plot(
    df_future.index,
    df_future["Forecast"],
    label="12-Month Out-of-Sample Forecast",
    color="#d62728",
    linewidth=2.5,
    marker="o",
)

# Safely annotate peak demand
if pd.Timestamp("2012-10-01") in df_future.index:
    oct_val = df_future.loc["2012-10-01", "Forecast"]
    plt.annotate(
        f"Oct 2012 Peak: {oct_val:,.0f}",
        xy=(pd.Timestamp("2012-10-01"), oct_val),
        xytext=(pd.Timestamp("2011-06-01"), oct_val + 3000),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=10,
        fontweight="bold",
    )

plt.title(
    "MMFD Tractor Demand Forecasting (Holt-Winters Model)",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Demand (Units)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left")
plt.tight_layout()

# Save plot for GitHub
plt.savefig("holt_winters_python_forecast.png")
print("Successfully generated and saved 'holt_winters_python_forecast.png'!")
plt.show()
