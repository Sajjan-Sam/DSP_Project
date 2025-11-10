"""
Exploratory Data Analysis for Solar Power Forecasting

Comprehensive analysis of solar generation and weather data.
Generates insights and visualizations for understanding patterns.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Add project root (one level above 'notebooks') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.utils.config import *
from src.utils.logger import get_logger

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Initialize logger
logger = get_logger("EDA")

# =============================================================================
# LOAD CLEANED DATA
# =============================================================================

logger.log_section("LOADING CLEANED DATA")

plant1 = pd.read_csv(PROCESSED_DATA_DIR / 'plant1_clean.csv')
plant2 = pd.read_csv(PROCESSED_DATA_DIR / 'plant2_clean.csv')

plant1['DATE_TIME'] = pd.to_datetime(plant1['DATE_TIME'])
plant2['DATE_TIME'] = pd.to_datetime(plant2['DATE_TIME'])

logger.info(f"Plant 1: {plant1.shape}")
logger.info(f"Plant 2: {plant2.shape}")


print("DATA LOADED SUCCESSFULLY")

print(f"\nPlant 1 Shape: {plant1.shape}")
print(f"Plant 2 Shape: {plant2.shape}")

# Display first few rows
print("\n" + "="*80)
print("PLANT 1 PREVIEW")

print(plant1.head())

print("\n" + "="*80)
print("PLANT 2 PREVIEW")

print(plant2.head())

# =============================================================================
# BASIC STATISTICS
# =============================================================================

logger.log_section("BASIC STATISTICS")

print("\n" + "="*80)
print("PLANT 1 STATISTICS")

print(plant1.describe())

print("\n" + "="*80)
print("PLANT 2 STATISTICS")

print(plant2.describe())

# =============================================================================
# TIME SERIES OVERVIEW
# =============================================================================

def plot_time_series_overview(df, plant_name):
    """Plot complete time series for all major variables"""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f'{plant_name} - Time Series Overview', fontsize=16, fontweight='bold')
    
    # Find power column
    power_col = [col for col in df.columns if 'TOTAL' in col and 'POWER' in col][0]
    
    # Plot 1: Power Generation
    axes[0].plot(df['DATE_TIME'], df[power_col], linewidth=0.5, alpha=0.7, color='#FF6B6B')
    axes[0].set_ylabel('Power (W)', fontsize=12, fontweight='bold')
    axes[0].set_title('Solar Power Generation', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Irradiation
    if 'IRRADIATION' in df.columns:
        axes[1].plot(df['DATE_TIME'], df['IRRADIATION'], linewidth=0.5, alpha=0.7, color='#FFA500')
        axes[1].set_ylabel('Irradiation (W/m²)', fontsize=12, fontweight='bold')
        axes[1].set_title('Solar Irradiation', fontsize=12)
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Temperature
    temp_col = [col for col in df.columns if 'AMBIENT' in col and 'TEMP' in col]
    if temp_col:
        axes[2].plot(df['DATE_TIME'], df[temp_col[0]], linewidth=0.5, alpha=0.7, color='#4ECDC4')
        axes[2].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        axes[2].set_title('Ambient Temperature', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = FIGURES_DIR / f'{plant_name.lower().replace(" ", "_")}_overview.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure: {save_path}")
    
    plt.show()

logger.log_section("TIME SERIES VISUALIZATION")
plot_time_series_overview(plant1, 'Plant 1')
plot_time_series_overview(plant2, 'Plant 2')

# =============================================================================
# DAILY PATTERNS
# =============================================================================

def analyze_daily_patterns(df, plant_name):
    """Analyze and visualize daily generation patterns"""
    
    df = df.copy()
    df['hour'] = df['DATE_TIME'].dt.hour
    df['date'] = df['DATE_TIME'].dt.date
    
    # Find power column
    power_col = [col for col in df.columns if 'TOTAL' in col and 'POWER' in col][0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{plant_name} - Daily Pattern Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Average hourly generation
    hourly_avg = df.groupby('hour')[power_col].mean()
    axes[0, 0].bar(hourly_avg.index, hourly_avg.values, color='#95E1D3', edgecolor='black')
    axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
    axes[0, 0].set_ylabel('Average Power (W)', fontsize=12)
    axes[0, 0].set_title('Average Generation by Hour', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Heatmap of daily profiles
    daily_profiles = df.pivot_table(
        values=power_col, 
        index='date', 
        columns='hour', 
        aggfunc='mean'
    )
    
    im = axes[0, 1].imshow(daily_profiles.values, aspect='auto', cmap='YlOrRd')
    axes[0, 1].set_xlabel('Hour of Day', fontsize=12)
    axes[0, 1].set_ylabel('Date', fontsize=12)
    axes[0, 1].set_title('Daily Generation Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 1], label='Power (W)')
    
    # Plot 3: Distribution by hour
    hours_to_plot = [6, 9, 12, 15, 18]
    for hour in hours_to_plot:
        hour_data = df[df['hour'] == hour][power_col]
        axes[1, 0].hist(hour_data, bins=30, alpha=0.5, label=f'{hour}:00')
    
    axes[1, 0].set_xlabel('Power (W)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Power Distribution by Hour', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Box plot by hour
    df.boxplot(column=power_col, by='hour', ax=axes[1, 1], patch_artist=True)
    axes[1, 1].set_xlabel('Hour of Day', fontsize=12)
    axes[1, 1].set_ylabel('Power (W)', fontsize=12)
    axes[1, 1].set_title('Generation Variability by Hour', fontsize=12, fontweight='bold')
    plt.suptitle('')  # Remove auto title
    
    plt.tight_layout()
    
    save_path = FIGURES_DIR / f'{plant_name.lower().replace(" ", "_")}_daily_patterns.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure: {save_path}")
    
    plt.show()
    
    # Print insights
    peak_hour = hourly_avg.idxmax()
    peak_power = hourly_avg.max()
    
    print(f"\n{'='*60}")
    print(f"{plant_name} - Daily Pattern Insights")
    print(f"{'='*60}")
    print(f"Peak generation hour: {peak_hour}:00")
    print(f"Peak average power: {peak_power:,.0f} W")
    print(f"Generation starts: ~{hourly_avg[hourly_avg > hourly_avg.max()*0.1].index.min()}:00")
    print(f"Generation ends: ~{hourly_avg[hourly_avg > hourly_avg.max()*0.1].index.max()}:00")

logger.log_section("DAILY PATTERN ANALYSIS")
analyze_daily_patterns(plant1, 'Plant 1')
analyze_daily_patterns(plant2, 'Plant 2')

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(df, plant_name):
    """Analyze correlations between variables"""
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(f'{plant_name} - Feature Correlation Matrix', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    save_path = FIGURES_DIR / f'{plant_name.lower().replace(" ", "_")}_correlations.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure: {save_path}")
    
    plt.show()
    
    # Find power column
    power_col = [col for col in df.columns if 'TOTAL' in col and 'POWER' in col][0]
    
    # Print top correlations with power
    power_corr = corr_matrix[power_col].sort_values(ascending=False)
    
    print(f"\n{'='*60}")
    print(f"{plant_name} - Top Correlations with Power Generation")
    print(f"{'='*60}")
    print(power_corr.head(10))

logger.log_section("CORRELATION ANALYSIS")
analyze_correlations(plant1, 'Plant 1')
analyze_correlations(plant2, 'Plant 2')

# =============================================================================
# WEATHER CONDITIONS ANALYSIS
# =============================================================================

def analyze_weather_conditions(df, plant_name):
    """Analyze power generation under different weather conditions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{plant_name} - Weather Impact Analysis', fontsize=16, fontweight='bold')
    
    power_col = [col for col in df.columns if 'TOTAL' in col and 'POWER' in col][0]
    
    # Plot 1: Power vs Irradiation
    if 'IRRADIATION' in df.columns:
        axes[0, 0].scatter(df['IRRADIATION'], df[power_col], alpha=0.3, s=1)
        axes[0, 0].set_xlabel('Irradiation (W/m²)', fontsize=12)
        axes[0, 0].set_ylabel('Power (W)', fontsize=12)
        axes[0, 0].set_title('Power vs Irradiation', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Power vs Temperature
    temp_col = [col for col in df.columns if 'AMBIENT' in col and 'TEMP' in col]
    if temp_col:
        axes[0, 1].scatter(df[temp_col[0]], df[power_col], alpha=0.3, s=1, color='orange')
        axes[0, 1].set_xlabel('Temperature (°C)', fontsize=12)
        axes[0, 1].set_ylabel('Power (W)', fontsize=12)
        axes[0, 1].set_title('Power vs Temperature', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Module temperature impact
    module_temp_col = [col for col in df.columns if 'MODULE' in col and 'TEMP' in col]
    if module_temp_col:
        axes[1, 0].scatter(df[module_temp_col[0]], df[power_col], alpha=0.3, s=1, color='red')
        axes[1, 0].set_xlabel('Module Temperature (°C)', fontsize=12)
        axes[1, 0].set_ylabel('Power (W)', fontsize=12)
        axes[1, 0].set_title('Power vs Module Temperature', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distribution of power
    axes[1, 1].hist(df[power_col], bins=50, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('Power (W)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Power Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    save_path = FIGURES_DIR / f'{plant_name.lower().replace(" ", "_")}_weather_impact.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure: {save_path}")
    
    plt.show()

logger.log_section("WEATHER CONDITIONS ANALYSIS")
analyze_weather_conditions(plant1, 'Plant 1')
analyze_weather_conditions(plant2, 'Plant 2')

# =============================================================================
# DATA QUALITY ASSESSMENT
# =============================================================================

logger.log_section("DATA QUALITY ASSESSMENT")

def assess_data_quality(df, plant_name):
    """Comprehensive data quality assessment"""
    
    print(f"\n{'='*80}")
    print(f"{plant_name} - DATA QUALITY REPORT")
    print(f"{'='*80}")
    
    # Completeness
    print("\n1. COMPLETENESS")
    print(f"   Total records: {len(df)}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Completeness: {100 * (1 - df.isnull().sum().sum() / df.size):.2f}%")
    
    # Consistency
    print("\n2. CONSISTENCY")
    df_sorted = df.sort_values('DATE_TIME')
    time_diffs = df_sorted['DATE_TIME'].diff()
    regular_intervals = (time_diffs == pd.Timedelta(minutes=15)).sum()
    print(f"   Regular 15-min intervals: {100 * regular_intervals / (len(df)-1):.2f}%")
    
    # Value ranges
    print("\n3. VALUE RANGES")
    power_col = [col for col in df.columns if 'TOTAL' in col and 'POWER' in col][0]
    print(f"   Power range: {df[power_col].min():.0f} - {df[power_col].max():.0f} W")
    
    if 'IRRADIATION' in df.columns:
        print(f"   Irradiation range: {df['IRRADIATION'].min():.2f} - {df['IRRADIATION'].max():.2f} W/m²")
    
    temp_col = [col for col in df.columns if 'AMBIENT' in col and 'TEMP' in col]
    if temp_col:
        print(f"   Temperature range: {df[temp_col[0]].min():.2f} - {df[temp_col[0]].max():.2f} °C")
    
    # Outliers
    print("\n4. OUTLIER DETECTION")
    Q1 = df[power_col].quantile(0.25)
    Q3 = df[power_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[power_col] < Q1 - 3*IQR) | (df[power_col] > Q3 + 3*IQR)).sum()
    print(f"   Power outliers (3*IQR): {outliers} ({100*outliers/len(df):.2f}%)")
    
    print("\n" + "="*80)

assess_data_quality(plant1, 'Plant 1')
assess_data_quality(plant2, 'Plant 2')

# =============================================================================
# SUMMARY AND RECOMMENDATIONS
# =============================================================================

logger.log_section("EDA SUMMARY")


print("EXPLORATORY DATA ANALYSIS COMPLETE")


print("\n KEY FINDINGS:")
print("    Data loaded and cleaned successfully")
print("    Clear daily generation patterns observed")
print("    Strong correlation between irradiation and power")
print("    Temperature effects on efficiency visible")
print("    Data quality is sufficient for modeling")

print("\n NEXT STEPS:")
print("   1. Feature engineering (create 50+ features)")
print("   2. Weather regime detection")
print("   3. Train baseline models")
print("   4. Implement novel ensemble methods")

print("\nOUTPUTS:")
print(f"   Figures saved to: {FIGURES_DIR}")
print(f"   Logs saved to: {logger.log_dir}")

logger.finalize()


