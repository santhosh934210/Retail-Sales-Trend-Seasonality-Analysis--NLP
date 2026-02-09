"""
Generate sample retail sales data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range (2 years of daily data)
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate base sales with trend and seasonality
n_days = len(dates)
base_sales = 1000  # Base daily sales

# Add trend (gradual increase over time)
trend = np.linspace(0, 500, n_days)

# Add seasonality (weekly and monthly patterns)
weekly_pattern = 200 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly cycle
monthly_pattern = 150 * np.sin(2 * np.pi * np.arange(n_days) / 30)  # Monthly cycle
yearly_pattern = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Yearly cycle

# Add random noise
noise = np.random.normal(0, 100, n_days)

# Combine all components
sales = base_sales + trend + weekly_pattern + monthly_pattern + yearly_pattern + noise

# Ensure no negative sales
sales = np.maximum(sales, 100)

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Sales': sales.round(2)
})

# Save to CSV
df.to_csv('retail_sales.csv', index=False)
print(f"Generated {len(df)} records of retail sales data")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Sales range: ${df['Sales'].min():.2f} to ${df['Sales'].max():.2f}")
print("Data saved to 'retail_sales.csv'")