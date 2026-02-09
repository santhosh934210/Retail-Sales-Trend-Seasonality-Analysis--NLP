"""
Generate sample retail sales data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate sales data with trend, seasonality, and noise
n_days = len(dates)
trend = np.linspace(1000, 1500, n_days)
seasonal = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
weekly_pattern = 100 * np.sin(2 * np.pi * np.arange(n_days) / 7)
noise = np.random.normal(0, 50, n_days)

sales = trend + seasonal + weekly_pattern + noise
sales = np.maximum(sales, 100)  # Ensure positive values

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Sales': sales.round(2)
})

# Save to CSV
df.to_csv('retail_sales.csv', index=False)
print(f"Generated {len(df)} days of retail sales data")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Sales range: ${df['Sales'].min():.2f} to ${df['Sales'].max():.2f}")
print("File saved as 'retail_sales.csv'")