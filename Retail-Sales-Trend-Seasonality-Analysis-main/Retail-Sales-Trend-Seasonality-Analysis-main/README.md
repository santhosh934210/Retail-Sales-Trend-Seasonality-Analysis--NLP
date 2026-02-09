# Retail Sales Analysis & Forecasting Project

A comprehensive time series analysis project for retail sales data, including trend analysis, seasonality detection, stationarity testing, and multiple forecasting models.

## Features

- **Time Series Decomposition**: Separates sales data into trend, seasonal, and residual components
- **Stationarity Testing**: Uses Augmented Dickey-Fuller test to assess data stationarity
- **Autocorrelation Analysis**: ACF and PACF plots to identify temporal dependencies
- **Multiple Forecasting Models**: ARIMA, Exponential Smoothing, and Linear Trend models
- **Model Comparison**: Performance evaluation using MAE and RMSE metrics
- **Future Predictions**: 30-day sales forecasts using the best-performing model

## Files

- `retail_sales_analysis.py` - Main analysis script
- `generate_sample_data.py` - Creates sample retail sales dataset
- `requirements.txt` - Python dependencies
- `retail_sales.csv` - Sample sales data (generated)

## Generated Outputs

- `decomposition_plot.png` - Time series decomposition visualization
- `acf_pacf_plot.png` - Autocorrelation and partial autocorrelation plots
- `stationarity_visual.png` - Rolling statistics and distribution plots
- `forecast_results.png` - Model forecasting comparison
- `future_predictions.csv` - 30-day sales predictions
- `model_comparison.csv` - Model performance metrics

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate sample data (optional):
   ```bash
   python generate_sample_data.py
   ```

3. Run the analysis:
   ```bash
   python retail_sales_analysis.py
   ```

## Analysis Components

### 1. Data Preprocessing
- Date parsing and indexing
- Missing value handling
- Basic statistical summary

### 2. Time Series Decomposition
- Additive decomposition model
- Trend extraction and analysis
- Seasonal pattern identification
- Residual noise quantification

### 3. Stationarity Assessment
- Augmented Dickey-Fuller test
- Rolling mean and standard deviation
- Visual stationarity checks
- First-order differencing if needed

### 4. Autocorrelation Analysis
- ACF plots for lag correlation
- PACF plots for direct correlation
- Temporal dependency identification

### 5. Forecasting Models
- **ARIMA(1,1,1)**: Auto-regressive integrated moving average
- **Exponential Smoothing**: Triple exponential smoothing with trend and seasonality
- **Linear Trend**: Simple linear regression model

### 6. Model Evaluation
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Performance ranking and best model selection

### 7. Future Predictions
- 30-day sales forecasts
- Prediction statistics and ranges
- CSV export for further analysis

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn

## Sample Output

The analysis provides:
- Comprehensive statistical summaries
- High-quality visualizations
- Model performance comparisons
- Future sales predictions
- Detailed interpretation of results

Perfect for retail analytics, business forecasting, and time series learning projects.