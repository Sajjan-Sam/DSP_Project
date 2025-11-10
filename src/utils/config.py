import os
from pathlib import Path

PROJECT_ROOT = Path("/data/Sajjan_Singh/DSP_Project")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"

for directory in [DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, 
                 RESULTS_DIR, MODELS_DIR, PREDICTIONS_DIR, 
                 FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "TOTAL_AC_POWER"
TIMESTAMP_COLUMN = "DATE_TIME"
FREQUENCY = "15T"

WEATHER_FEATURES = [
    "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
    "temp_difference", "temp_ratio", "irrad_temp_interaction"
]

TEMPORAL_FEATURES = [
    "hour", "minute", "day_of_year", "day_of_week",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_daytime", "is_peak_sun"
]

SOLAR_FEATURES = [
    "solar_elevation", "solar_declination", 
    "theoretical_irradiance"
]

HORIZONS = {
    "short": 4,
    "medium": 16,
    "long": 96
}

PROPHET_CONFIG = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "multiplicative",
    "yearly_seasonality": False,
    "weekly_seasonality": True,
    "daily_seasonality": True,
    "interval_width": 0.95,
    "n_changepoints": 25
}

SARIMA_CONFIG = {
    "order": (2, 1, 2),
    "seasonal_order": (1, 1, 1, 96),
    "trend": "c",
    "enforce_stationarity": False,
    "enforce_invertibility": False,
    "maxiter": 200,
    "method": "lbfgs"
}

LSTM_CONFIG = {
    "sequence_length": 48,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.15,
    "learning_rate": 1.5e-4,
    "batch_size": 128,
    "epochs": 120,
    "early_stopping_patience": 15
}

QRF_CONFIG = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42
}

ENSEMBLE_CONFIG = {
    "base_models": ["prophet", "sarima", "lstm", "qrf"],
    "attention_hidden_size": 64,
    "attention_dropout": 0.3,
    "learning_rate": 1.5e-4,
    "epochs": 50,
    "batch_size": 128,
    "regime_adaptive": True
}

TRAINING_CONFIG = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42
}

EVALUATION_METRICS = [
    "MAE", "RMSE", "MAPE", "R2", "CRPS"
]

VIZ_CONFIG = {
    "figure_size": (15, 10),
    "dpi": 300,
    "font_size": 10
}