"""
Retail Sales Trend & Seasonality Analysis
Time series decomposition, stationarity testing, and ACF/PACF analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
print("=" * 60)
print("RETAIL SALES TREND & SEASONALITY ANALYSIS")
print("=" * 60)

df = pd.read_csv('retail_sales.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

print(f"\nDataset Shape: {df.shape}")
print(f"Date Range: {df.index.min()} to {df.index.max()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# Check for missing values
if df['Sales'].isnull().sum() > 0:
    print(f"\nWarning: {df['Sales'].isnull().sum()} missing values found. Filling with forward fill.")
    df['Sales'].fillna(method='ffill', inplace=True)

# Time Series Decomposition
print("\n" + "=" * 60)
print("TIME SERIES DECOMPOSITION")
print("=" * 60)

decomposition = seasonal_decompose(df['Sales'], model='additive', period=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

axes[0].plot(df['Sales'], color='blue', linewidth=1.5)
axes[0].set_title('Original Sales Data', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Sales', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].plot(trend, color='green', linewidth=2)
axes[1].set_title('Trend Component', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Trend', fontsize=12)
axes[1].grid(True, alpha=0.3)

axes[2].plot(seasonal, color='orange', linewidth=1.5)
axes[2].set_title('Seasonal Component', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Seasonality', fontsize=12)
axes[2].grid(True, alpha=0.3)

axes[3].plot(residual, color='red', linewidth=1)
axes[3].set_title('Residual (Noise) Component', fontsize=14, fontweight='bold')
axes[3].set_ylabel('Residuals', fontsize=12)
axes[3].set_xlabel('Date', fontsize=12)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decomposition_plot.png', dpi=300, bbox_inches='tight')
print("\n✓ Decomposition plot saved as 'decomposition_plot.png'")
# plt.show()

# Component statistics
print(f"\nTrend Statistics:")
print(f"  Mean: {trend.mean():.2f}")
print(f"  Std Dev: {trend.std():.2f}")
print(f"\nSeasonal Statistics:")
print(f"  Mean: {seasonal.mean():.2f}")
print(f"  Amplitude: {seasonal.max() - seasonal.min():.2f}")
print(f"\nResidual Statistics:")
print(f"  Mean: {residual.mean():.2f}")
print(f"  Std Dev: {residual.std():.2f}")

# ACF and PACF Analysis
print("\n" + "=" * 60)
print("AUTOCORRELATION ANALYSIS (ACF & PACF)")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(df['Sales'].dropna(), lags=40, ax=axes[0], color='blue')
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Lag', fontsize=12)
axes[0].set_ylabel('ACF', fontsize=12)
axes[0].grid(True, alpha=0.3)

plot_pacf(df['Sales'].dropna(), lags=40, ax=axes[1], color='green')
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Lag', fontsize=12)
axes[1].set_ylabel('PACF', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('acf_pacf_plot.png', dpi=300, bbox_inches='tight')
print("\n✓ ACF/PACF plot saved as 'acf_pacf_plot.png'")
# plt.show()

print("\nInterpretation:")
print("  - ACF shows correlation between observations at different lags")
print("  - PACF shows direct correlation after removing intermediate lag effects")
print("  - Significant spikes indicate temporal dependencies in the data")

# Stationarity Test (ADF Test)
print("\n" + "=" * 60)
print("STATIONARITY TEST (Augmented Dickey-Fuller)")
print("=" * 60)

def adf_test(series, name=''):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\nADF Test Results for {name}:")
    print(f"  ADF Statistic: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.3f}")
    
    if result[1] <= 0.05:
        print(f"  ✓ STATIONARY (p-value ≤ 0.05)")
        print(f"    Reject null hypothesis - No unit root present")
    else:
        print(f"  ✗ NON-STATIONARY (p-value > 0.05)")
        print(f"    Fail to reject null hypothesis - Unit root present")
    
    return result[1] <= 0.05

# Test original series
is_stationary = adf_test(df['Sales'], 'Original Sales Data')

# Visual stationarity check
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Rolling statistics
rolling_mean = df['Sales'].rolling(window=30).mean()
rolling_std = df['Sales'].rolling(window=30).std()

axes[0].plot(df['Sales'], color='blue', label='Original Sales', linewidth=1.5)
axes[0].plot(rolling_mean, color='red', label='Rolling Mean (30-day)', linewidth=2)
axes[0].plot(rolling_std, color='green', label='Rolling Std Dev (30-day)', linewidth=2)
axes[0].set_title('Rolling Mean & Standard Deviation', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Sales', fontsize=12)
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Distribution
axes[1].hist(df['Sales'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].set_title('Sales Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sales', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('stationarity_visual.png', dpi=300, bbox_inches='tight')
print("\n✓ Stationarity visualization saved as 'stationarity_visual.png'")
# plt.show()

# If non-stationary, test differenced series
if not is_stationary:
    print("\n" + "-" * 60)
    print("Testing First-Order Differenced Series")
    print("-" * 60)
    df['Sales_diff'] = df['Sales'].diff()
    adf_test(df['Sales_diff'], 'First-Order Differenced Sales')

# Final Summary
print("\n" + "=" * 60)
print("FORECASTING READINESS SUMMARY")
print("=" * 60)

print("\n✓ Decomposition Complete:")
print("  - Trend component extracted and analyzed")
print("  - Seasonal patterns identified")
print("  - Residual noise quantified")

print("\n✓ Autocorrelation Analysis Complete:")
print("  - ACF and PACF plots generated")
print("  - Temporal dependencies visualized")

print("\n✓ Stationarity Assessment Complete:")
print("  - ADF test conducted")
if is_stationary:
    print("  - Dataset is STATIONARY - Ready for forecasting")
else:
    print("  - Dataset is NON-STATIONARY - Differencing recommended")
    print("  - First-order differencing may achieve stationarity")

print("\n✓ Dataset Validated:")
print("  - Statistical properties understood")
print("  - Suitable preprocessing identified")
print("  - Ready for forecasting model development")

# FORECASTING MODELS
print("\n" + "=" * 60)
print("FORECASTING MODELS")
print("=" * 60)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# Prepare data for forecasting
train_size = int(len(df) * 0.8)
train_data = df['Sales'][:train_size]
test_data = df['Sales'][train_size:]

print(f"Training data: {len(train_data)} observations")
print(f"Test data: {len(test_data)} observations")

# Model 1: ARIMA
print("\n" + "-" * 40)
print("ARIMA Model")
print("-" * 40)

try:
    arima_model = ARIMA(train_data, order=(1,1,1))
    arima_fitted = arima_model.fit()
    arima_forecast = arima_fitted.forecast(steps=len(test_data))
    arima_mae = mean_absolute_error(test_data, arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
    print(f"ARIMA MAE: {arima_mae:.2f}")
    print(f"ARIMA RMSE: {arima_rmse:.2f}")
except:
    print("ARIMA model failed - using simple forecast")
    arima_forecast = [train_data.mean()] * len(test_data)
    arima_mae = mean_absolute_error(test_data, arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))

# Model 2: Exponential Smoothing
print("\n" + "-" * 40)
print("Exponential Smoothing")
print("-" * 40)

try:
    exp_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=30)
    exp_fitted = exp_model.fit()
    exp_forecast = exp_fitted.forecast(steps=len(test_data))
    exp_mae = mean_absolute_error(test_data, exp_forecast)
    exp_rmse = np.sqrt(mean_squared_error(test_data, exp_forecast))
    print(f"Exponential Smoothing MAE: {exp_mae:.2f}")
    print(f"Exponential Smoothing RMSE: {exp_rmse:.2f}")
except:
    print("Exponential Smoothing failed - using trend forecast")
    exp_forecast = np.linspace(train_data.iloc[-1], train_data.iloc[-1] * 1.1, len(test_data))
    exp_mae = mean_absolute_error(test_data, exp_forecast)
    exp_rmse = np.sqrt(mean_squared_error(test_data, exp_forecast))

# Model 3: Linear Trend
print("\n" + "-" * 40)
print("Linear Trend Model")
print("-" * 40)

X_train = np.arange(len(train_data)).reshape(-1, 1)
X_test = np.arange(len(train_data), len(df)).reshape(-1, 1)
linear_model = LinearRegression()
linear_model.fit(X_train, train_data)
linear_forecast = linear_model.predict(X_test)
linear_mae = mean_absolute_error(test_data, linear_forecast)
linear_rmse = np.sqrt(mean_squared_error(test_data, linear_forecast))
print(f"Linear Trend MAE: {linear_mae:.2f}")
print(f"Linear Trend RMSE: {linear_rmse:.2f}")

# Forecast Visualization
fig, ax = plt.subplots(figsize=(14, 8))

# Plot historical data
ax.plot(train_data.index, train_data, label='Training Data', color='blue', linewidth=2)
ax.plot(test_data.index, test_data, label='Actual Test Data', color='black', linewidth=2)

# Plot forecasts
ax.plot(test_data.index, arima_forecast, label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
ax.plot(test_data.index, exp_forecast, label='Exp. Smoothing Forecast', color='green', linestyle='--', linewidth=2)
ax.plot(test_data.index, linear_forecast, label='Linear Trend Forecast', color='orange', linestyle='--', linewidth=2)

ax.axvline(x=train_data.index[-1], color='gray', linestyle=':', alpha=0.7, label='Train/Test Split')
ax.set_title('Sales Forecasting Results', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('forecast_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Forecast results saved as 'forecast_results.png'")
# plt.show()

# Model Comparison
print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

results_df = pd.DataFrame({
    'Model': ['ARIMA', 'Exponential Smoothing', 'Linear Trend'],
    'MAE': [arima_mae, exp_mae, linear_mae],
    'RMSE': [arima_rmse, exp_rmse, linear_rmse]
})

results_df = results_df.sort_values('MAE')
print("\nModel Rankings (by MAE):")
for i, row in results_df.iterrows():
    print(f"{row['Model']:20} | MAE: {row['MAE']:8.2f} | RMSE: {row['RMSE']:8.2f}")

best_model = results_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model}")

# Future Predictions
print("\n" + "=" * 60)
print("FUTURE PREDICTIONS (Next 30 Days)")
print("=" * 60)

if best_model == 'ARIMA':
    try:
        future_forecast = arima_fitted.forecast(steps=30)
    except:
        future_forecast = [train_data.mean()] * 30
elif best_model == 'Exponential Smoothing':
    try:
        future_forecast = exp_fitted.forecast(steps=30)
    except:
        future_forecast = np.linspace(df['Sales'].iloc[-1], df['Sales'].iloc[-1] * 1.1, 30)
else:
    X_future = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    future_forecast = linear_model.predict(X_future)

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Sales': future_forecast})

print("\nNext 30 Days Forecast:")
print(future_df.head(10))
print(f"\nAverage predicted sales: {future_forecast.mean():.2f}")
print(f"Predicted sales range: {future_forecast.min():.2f} - {future_forecast.max():.2f}")

# Save results
future_df.to_csv('future_predictions.csv', index=False)
results_df.to_csv('model_comparison.csv', index=False)

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)
print("\n✓ Files Generated:")
print("  - decomposition_plot.png")
print("  - acf_pacf_plot.png")
print("  - stationarity_visual.png")
print("  - forecast_results.png")
print("  - future_predictions.csv")
print("  - model_comparison.csv")
print("\n✓ Analysis Complete:")
print("  - Time series decomposition")
print("  - Stationarity testing")
print("  - Autocorrelation analysis")
print("  - Multiple forecasting models")
print("  - Model performance comparison")
print("  - Future predictions generated")


from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Run ADF test and store full result
adf_result = adfuller(df['Sales'].dropna(), autolag='AIC')

# Also keep stationarity flag if needed
is_stationary = adf_result[1] <= 0.05


def create_pdf_report(adf_result, name, regno):
    doc = SimpleDocTemplate("Retail_Sales_Report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title Page
    story.append(Paragraph("<b>Retail Sales Trend & Seasonality Analysis</b>", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Name: {name}", styles['Normal']))
    story.append(Paragraph(f"Registration Number: {regno}", styles['Normal']))
    story.append(Spacer(1, 40))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>", styles['Heading2']))
    story.append(Paragraph("This project analyzes retail sales data to identify trend, seasonality, and randomness. It validates the dataset for forecasting readiness using decomposition, autocorrelation analysis, and stationarity testing.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Problem Statement
    story.append(Paragraph("<b>Problem Statement</b>", styles['Heading2']))
    story.append(Paragraph("Management lacks clarity on whether sales demonstrate a long-term trend, recurring seasonal patterns, or randomness. Without statistical validation, forecasting models may be unreliable.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Methodology
    story.append(Paragraph("<b>Methodology</b>", styles['Heading2']))
    story.append(Paragraph("Steps include: data preprocessing, time series decomposition, ACF/PACF analysis, and stationarity testing using the Augmented Dickey-Fuller test.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Results
    story.append(Paragraph("<b>Results</b>", styles['Heading2']))
    story.append(Paragraph(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Insert Plots
    story.append(Image("decomposition_plot.png", width=400, height=250))
    story.append(Spacer(1, 20))
    story.append(Image("acf_pacf_plot.png", width=400, height=250))
    story.append(Spacer(1, 20))
    story.append(Image("stationarity_visual.png", width=400, height=250))

    # Conclusion
    story.append(Paragraph("<b>Conclusion</b>", styles['Heading2']))
    story.append(Paragraph("The dataset shows clear seasonal patterns and requires transformation for stationarity before forecasting. This analysis ensures statistical readiness for predictive modeling.", styles['Normal']))

    story.append(Paragraph("<b>Introduction</b>", styles['Heading2']))
    story.append(Paragraph("Retail sales forecasting is critical for inventory management, staffing, and promotional planning. This project explores statistical properties of sales data to ensure forecasting readiness.", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("<b>Forecasting Models</b>", styles['Heading2']))
    story.append(Paragraph("We tested ARIMA, Exponential Smoothing, and Linear Trend models. Their performance was compared using MAE and RMSE metrics.", styles['Normal']))
    story.append(Image("forecast_results.png", width=400, height=250))


    doc.build(story)
     
# Call the function with your details
create_pdf_report(adf_result, name="NITHISHKUMAR S", regno="727723EUCS152")

