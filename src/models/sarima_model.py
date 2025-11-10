import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from src.models.base_model import BaseForecastModel

class SARIMAForecastModel(BaseForecastModel):
    def __init__(self, config: Dict):
        super().__init__(config, "SARIMA")
        self.exog_features: List[str] = []
        self.fitted_model = None

    def _select_exog_features(self, X: pd.DataFrame, max_features: int = 2) -> List[str]:
        patterns = ['IRRADIATION', 'IRRAD', 'GHI', 'AMBIENT_TEMPERATURE', 'TEMP',
                    'solar_elevation', 'sun_elevation', 'clearness_index', 'clear_sky_index',
                    'hour_sin', 'hour_cos']
        selected: List[str] = []
        for p in patterns:
            if len(selected) >= max_features:
                break
            matches = [c for c in X.columns if p in c]
            if matches:
                selected.append(matches[0])
        return selected[:max_features]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        self.validate_data(X_train, y_train)
        self.exog_features = self._select_exog_features(X_train, max_features=5)
        exog_train = X_train[self.exog_features].ffill().bfill() if self.exog_features else None
        order = self.config.get('order', (2, 1, 2))
        seasonal_order = self.config.get('seasonal_order', (1, 1, 1, 96))
        try:
            self.model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                                 simple_differencing=True, concentrate_scale=True,
                                 enforce_stationarity=self.config.get('enforce_stationarity', False),
                                 enforce_invertibility=self.config.get('enforce_invertibility', False))
            self.fitted_model = self.model.fit(disp=False, maxiter=self.config.get('maxiter', 50),
                                               method=self.config.get('method', 'lbfgs'))
            self.is_trained = True
            self.feature_names = list(X_train.columns)
        except Exception:
            self.model = SARIMAX(y_train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 96),
                                 simple_differencing=True, concentrate_scale=True,
                                 enforce_stationarity=False, enforce_invertibility=False)
            self.fitted_model = self.model.fit(disp=False, maxiter=100)
            self.is_trained = True
            self.feature_names = list(X_train.columns)

    def predict(self, X: pd.DataFrame, return_std: bool = False):
        if not self.is_trained or self.fitted_model is None:
            raise ValueError("Model must be trained before prediction.")
        exog = X[self.exog_features].ffill().bfill() if self.exog_features else None
        n_steps = len(X)
        fc = self.fitted_model.get_forecast(steps=n_steps, exog=exog)
        mean = np.maximum(fc.predicted_mean.values, 0)
        if return_std:
            ci = fc.conf_int(alpha=0.05)
            std = (ci.iloc[:, 1] - ci.iloc[:, 0]) / (2 * 1.96)
            return mean, std.values
        return mean

    def save(self, filepath):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        from pathlib import Path
        import pickle
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model": self.model, "fitted_model": self.fitted_model, "config": self.config,
            "model_name": self.model_name, "feature_names": self.feature_names,
            "exog_features": self.exog_features, "training_history": self.training_history,
            "is_trained": self.is_trained
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Model saved: {filepath}")

    def load(self, filepath):
        from pathlib import Path
        import pickle
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)
        self.model = save_dict["model"]
        self.fitted_model = save_dict["fitted_model"]
        self.config = save_dict["config"]
        self.model_name = save_dict["model_name"]
        self.feature_names = save_dict["feature_names"]
        self.exog_features = save_dict.get("exog_features", [])
        self.training_history = save_dict.get("training_history", {})
        self.is_trained = save_dict["is_trained"]
        print(f"Model loaded: {filepath}")

    def __repr__(self) -> str:
        return f"SARIMAForecastModel ({'trained' if self.is_trained else 'untrained'}, {len(self.exog_features)} exog)"
