import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.feature_engineering import SolarFeatureEngineer
from src.data.weather_regimes import WeatherRegimeDetector
from src.utils.logger import setup_logger
from datetime import datetime
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

logger = setup_logger('feature_engineering', 'logs/feature_engineering.log')

def load_cleaned_data():
    """Load preprocessed clean data"""
    logger.info("Loading cleaned datasets...")
    
    data_dir = "/data/Sajjan_Singh/DSP_Project/data/processed"
    plant1 = pd.read_csv(os.path.join(data_dir, "plant1_clean.csv"))
    plant2 = pd.read_csv(os.path.join(data_dir, "plant2_clean.csv"))
    
    logger.info(f"Plant 1 shape: {plant1.shape}")
    logger.info(f"Plant 2 shape: {plant2.shape}")
    
    return plant1, plant2

def engineer_features(plant1, plant2):
    """Apply feature engineering pipeline"""
    logger.info("\nStarting feature engineering pipeline...")
    
    engineer = SolarFeatureEngineer()
    
    # Plant 1
    logger.info("\nProcessing Plant 1...")
    plant1_featured = engineer.create_all_features(plant1, target_col='TOTAL_AC_POWER')
    logger.info(f"Plant 1 featured shape: {plant1_featured.shape}")
    
    # Plant 2
    logger.info("\nProcessing Plant 2...")
    plant2_featured = engineer.create_all_features(plant2, target_col='TOTAL_AC_POWER')
    logger.info(f"Plant 2 featured shape: {plant2_featured.shape}")
    
    # Print feature groups
    groups = engineer.get_feature_groups()
    logger.info("\nFeature composition:")
    for group_name, features in groups.items():
        logger.info(f"  {group_name}: {len(features)} features")
    logger.info(f"  Total features: {len(engineer.feature_names)}")
    
    return plant1_featured, plant2_featured, engineer

def detect_weather_regimes(plant1, plant2):
    """Detect weather regimes for adaptive modeling"""
    logger.info("\nDetecting weather regimes...")
    
    # Plant 1 regimes
    logger.info("\nPlant 1 regime detection:")
    detector1 = WeatherRegimeDetector(n_regimes=4, method='kmeans')
    plant1_regimes = detector1.detect_regimes(plant1)
    detector1.print_regime_summary()
    
    # Plant 2 regimes
    logger.info("\nPlant 2 regime detection:")
    detector2 = WeatherRegimeDetector(n_regimes=4, method='kmeans')
    plant2_regimes = detector2.detect_regimes(plant2)
    detector2.print_regime_summary()
    
    # Visualize regimes
    fig_dir = "/data/Sajjan_Singh/DSP_Project/results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    detector1.visualize_regimes(plant1_regimes, 
                               save_path=os.path.join(fig_dir, "weather_regimes_plant1.png"))
    detector2.visualize_regimes(plant2_regimes, 
                               save_path=os.path.join(fig_dir, "weather_regimes_plant2.png"))
    
    return plant1_regimes, plant2_regimes, detector1, detector2

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """
    Temporal split for time series data
    Maintains chronological order
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train)} samples ({len(train)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val)} samples ({len(val)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test)} samples ({len(test)/n*100:.1f}%)")
    
    if 'DATE_TIME' in df.columns:
        logger.info(f"\nTemporal ranges:")
        logger.info(f"  Train: {train['DATE_TIME'].min()} to {train['DATE_TIME'].max()}")
        logger.info(f"  Val:   {val['DATE_TIME'].min()} to {val['DATE_TIME'].max()}")
        logger.info(f"  Test:  {test['DATE_TIME'].min()} to {test['DATE_TIME'].max()}")
    
    return train, val, test

def save_splits(plant1, plant2, suffix=""):
    """Save train/val/test splits"""
    logger.info("\nSaving data splits...")
    
    splits_dir = "/data/Sajjan_Singh/DSP_Project/data/splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    # Split Plant 1
    p1_train, p1_val, p1_test = split_data(plant1)
    p1_train.to_csv(os.path.join(splits_dir, f"plant1_train{suffix}.csv"), index=False)
    p1_val.to_csv(os.path.join(splits_dir, f"plant1_val{suffix}.csv"), index=False)
    p1_test.to_csv(os.path.join(splits_dir, f"plant1_test{suffix}.csv"), index=False)
    
    # Split Plant 2
    p2_train, p2_val, p2_test = split_data(plant2)
    p2_train.to_csv(os.path.join(splits_dir, f"plant2_train{suffix}.csv"), index=False)
    p2_val.to_csv(os.path.join(splits_dir, f"plant2_val{suffix}.csv"), index=False)
    p2_test.to_csv(os.path.join(splits_dir, f"plant2_test{suffix}.csv"), index=False)
    
    logger.info(f"Data splits saved to {splits_dir}")

def analyze_feature_importance_preliminary(df, target='TOTAL_AC_POWER'):
    """
    Quick feature importance using correlation
    """
    logger.info("\nPreliminary feature importance analysis...")
    
    # Calculate correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in [target, 'DATE_TIME']]
    
    correlations = {}
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(df[target])
            if not np.isnan(corr):
                correlations[col] = abs(corr)
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nTop 20 most correlated features:")
    for i, (feat, corr) in enumerate(sorted_corr[:20], 1):
        logger.info(f"  {i:2d}. {feat:50s} | {corr:.4f}")
    
    # Visualize top correlations
    top_features = [f[0] for f in sorted_corr[:20]]
    top_corrs = [f[1] for f in sorted_corr[:20]]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_features)), top_corrs)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel('Absolute Correlation with Target')
    ax.set_title('Top 20 Features by Correlation')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    fig_dir = "/data/Sajjan_Singh/DSP_Project/results/figures"
    plt.savefig(os.path.join(fig_dir, "feature_correlations.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    return sorted_corr

def main():
    """Main execution pipeline"""
    start_time = datetime.now()
    logger.info("Starting feature engineering experiment")
    logger.info(f"Timestamp: {start_time}")
    
    metrics = {
        'start_time': str(start_time),
        'stages': {}
    }
    
    try:
        # Stage 1: Load data
        plant1, plant2 = load_cleaned_data()
        
        # Stage 2: Feature engineering
        plant1_featured, plant2_featured, engineer = engineer_features(plant1, plant2)
        metrics['stages']['feature_engineering'] = {
            'plant1_features': len(plant1_featured.columns),
            'plant2_features': len(plant2_featured.columns)
        }
        
        # Stage 3: Weather regime detection
        plant1_final, plant2_final, det1, det2 = detect_weather_regimes(
            plant1_featured, plant2_featured)
        
        metrics['stages']['weather_regimes'] = {
            'plant1_regimes': det1.n_regimes,
            'plant2_regimes': det2.n_regimes
        }
        
        # Stage 4: Save full featured datasets
        processed_dir = "/data/Sajjan_Singh/DSP_Project/data/processed"
        plant1_final.to_csv(os.path.join(processed_dir, "plant1_featured.csv"), 
                          index=False)
        plant2_final.to_csv(os.path.join(processed_dir, "plant2_featured.csv"), 
                          index=False)
        logger.info("\nFull featured datasets saved")
        
        # Stage 5: Create train/val/test splits
        save_splits(plant1_final, plant2_final)
        
        # Stage 6: Preliminary feature analysis
        logger.info("\nAnalyzing Plant 1 features...")
        p1_corr = analyze_feature_importance_preliminary(plant1_final)
        
        logger.info("\nAnalyzing Plant 2 features...")
        p2_corr = analyze_feature_importance_preliminary(plant2_final)
        
        # Save metrics
        end_time = datetime.now()
        metrics['end_time'] = str(end_time)
        metrics['duration_seconds'] = (end_time - start_time).total_seconds()
        
        metrics_path = "logs/feature_engineering_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"\nFeature engineering complete!")
        logger.info(f"Duration: {metrics['duration_seconds']:.2f} seconds")
        logger.info(f"Metrics saved to {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise

if __name__ == "__main__":
    main()