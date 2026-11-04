import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


file_path = "DAMS 301 only MMFD Demand Part 3.xlsx"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"File '{file_path}' not found. Please ensure it is in the working directory."
    )

# Read Excel file
df = pd.read_excel(file_path)

# Rename columns to standard lowercase format if needed
df.columns = ["Month", "Demand"]

# Ensure proper datetime format and set frequency to Monthly Start (MS)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df.index.freq = "MS"

print("--- Historical Data Overview ---")
print(f"Data range: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
print(f"Total historical observations: {len(df)}\n")

# ==============================================================================
# 2. BUILD & FIT HOLT-WINTERS MULTIPLICATIVE SEASONAL MODEL
# ==============================================================================
# Configure Holt-Winters model matching Part 3 setup:
# - Additive Trend
# - Multiplicative Seasonality (12-month period)
model = ExponentialSmoothing(
    df["Demand"],
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
    initialization_method="estimated",
)

# Fit model (Statsmodels automatically optimizes alpha, beta, gamma using GRG/Maximum Likelihood)
fit_model = model.fit()

# Extract fitted historical values
df["Fitted_Forecast"] = fit_model.fittedvalues

# ==============================================================================
# 3. GENERATE 12-MONTH FUTURE FORECAST
# ==============================================================================
forecast_horizon = 12
future_forecast = fit_model.forecast(forecast_horizon)

# Create DataFrame for future forecast periods (April 2012 - March 2013)
future_dates = pd.date_range(
    start=df.index.max() + pd.DateOffset(months=1), periods=forecast_horizon, freq="MS"
)
df_future = pd.DataFrame(
    {"Forecast": future_forecast.values}, index=future_dates
)

# ==============================================================================
# 4. EVALUATE ERROR METRICS
# ==============================================================================
# Compute error terms on historical fitted values
errors = df["Demand"] - df["Fitted_Forecast"]
abs_errors = np.abs(errors)
sq_errors = errors**2
pct_errors = abs_errors / df["Demand"]

mae = np.mean(abs_errors)
rmse = np.sqrt(np.mean(sq_errors))
mape = np.mean(pct_errors) * 100
mean_error = np.mean(errors)

print("--- Model Accuracy Metrics ---")
print(f"Mean Error (ME):                 {mean_error:.2f}")
print(f"Mean Absolute Error (MAE):       {mae:.2f}")
print(f"Root Mean Squared Error (RMSE):  {rmse:.2f}")
print(f"Mean Absolute Percentage (MAPE): {mape:.2f}%\n")

print("--- 12-Month Out-of-Sample Forecast (Apr 2012 - Mar 2013) ---")
for date, val in zip(df_future.index, df_future["Forecast"]):
    print(f"{date.strftime('%Y-%m')}: {val:,.2f} units")

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(14, 7), dpi=300)

# Plot historical actuals, fitted values, and future forecast
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

# Highlight October 2012 Peak
oct_2012_val = df_future.loc["2012-10-01", "Forecast"]
plt.annotate(
    f"Oct 2012 Peak: {oct_2012_val:,.0f}",
    xy=(pd.Timestamp("2012-10-01"), oct_2012_val),
    xytext=(pd.Timestamp("2011-06-01"), oct_2012_val + 3000),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    fontsize=10,
    fontweight="bold",
)

# Graph Formatting
plt.title(
    "MMFD Tractor Demand Forecasting using Holt-Winters Multiplicative Method",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Demand (Units)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left", frameon=True)
plt.tight_layout()

# Save image for GitHub repository README
plt.savefig("holt_winters_python_forecast.png")
print("\nPlot saved successfully as 'holt_winters_python_forecast.png'.")
plt.show()
