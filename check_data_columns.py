import pandas as pd
import sys
import os

# Add project to path
sys.path.insert(0, '/data/Sajjan_Singh/DSP_Project')

from src.utils.config import Config

print("Checking data columns...")

# Load train data
train_path = Config.get_data_path('split', 1, 'train')
print(f"\nLoading: {train_path}")

train_data = pd.read_csv(train_path)

print(f"\nShape: {train_data.shape}")
print(f"\nAll columns ({len(train_data.columns)}):")
for i, col in enumerate(train_data.columns, 1):
    print(f"  {i:3d}. {col}")

print("\n" + "="*80)
print("Weather-related columns:")
weather_cols = [col for col in train_data.columns if any(word in col.upper() 
                for word in ['TEMP', 'IRRAD', 'WEATHER', 'SOLAR', 'CLEAR'])]
for col in weather_cols:
    print(f"  - {col}")

print("\n" + "="*80)
print("Power-related columns:")
power_cols = [col for col in train_data.columns if 'POWER' in col.upper() or 'DC' in col or 'AC' in col]
for col in power_cols:
    print(f"  - {col}")

print("\n" + "="*80)
print("First few rows of key columns:")
key_cols = ['DATE_TIME', 'TOTAL_AC_POWER', 'TOTAL_DC_POWER']
existing_key_cols = [col for col in key_cols if col in train_data.columns]

# Add some weather columns if they exist
weather_check = ['TOTAL_IRRADIATION', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 
                'MODULE_TEMPERATURE', 'solar_elevation']
for col in weather_check:
    if col in train_data.columns:
        existing_key_cols.append(col)
        if len(existing_key_cols) >= 7:
            break

print(train_data[existing_key_cols].head(10))

print("\n" + "="*80)
print("Data types:")
for col in existing_key_cols:
    print(f"  {col}: {train_data[col].dtype}")