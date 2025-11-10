import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from prophet import Prophet
from src.models.base_model import BaseForecastModel
import warnings
warnings.filterwarnings('ignore')

class ProphetForecastModel(BaseForecastModel):
    def __init__(self, config: Dict):
        super().__init__(config, "Prophet")
        self.regressors: List[str] = []

    def _detect_available_regressors(self, X: pd.DataFrame) -> List[str]:
        patterns = {
            'irradiation': ['IRRADIATION', 'IRRAD', 'GHI', 'TOTAL_IRRADIATION'],
            'ambient_temp': ['AMBIENT_TEMPERATURE', 'AMBIENT_TEMP', 'TEMP_AMBIENT'],
            'module_temp': ['MODULE_TEMPERATURE', 'MODULE_TEMP', 'TEMP_MODULE'],
            'clearness': ['clearness_index', 'CLEARNESS_INDEX', 'clear_sky_index'],
            'solar_elevation': ['solar_elevation', 'SOLAR_ELEVATION', 'sun_elevation'],
            'theoretical_irrad': ['theoretical_irradiance', 'THEORETICAL_IRRADIANCE']
        }
        available = []
        for _, names in patterns.items():
            for n in names:
                if n in X.columns:
                    available.append(n)
                    break
        return available

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        self.validate_data(X_train, y_train)
        self.regressors = self._detect_available_regressors(X_train)

        self.model = Prophet(
            changepoint_prior_scale=self.config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.config.get('seasonality_prior_scale', 10.0),
            seasonality_mode=self.config.get('seasonality_mode', 'multiplicative'),
            yearly_seasonality=self.config.get('yearly_seasonality', False),
            weekly_seasonality=self.config.get('weekly_seasonality', True),
            daily_seasonality=self.config.get('daily_seasonality', True),
            interval_width=self.config.get('interval_width', 0.95)
        )
        self.model.add_seasonality(name='hourly', period=1, fourier_order=8)
        for r in self.regressors:
            self.model.add_regressor(r, standardize=True)

        train_df = self._prepare_prophet_data(X_train, y_train)
        # IMPORTANT: older/newer Prophet builds don't accept show_progress/progress_bar
        self.model.fit(train_df)

        self.is_trained = True
        self.feature_names = list(X_train.columns)

    def _prepare_prophet_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        df = pd.DataFrame()
        if 'DATE_TIME' in X.columns:
            df['ds'] = pd.to_datetime(X['DATE_TIME'])
        elif isinstance(X.index, pd.DatetimeIndex):
            df['ds'] = X.index
        else:
            df['ds'] = pd.date_range(start='2020-01-01', periods=len(X), freq='15min')
        if y is not None:
            df['y'] = y.values
        for r in self.regressors:
            if r in X.columns:
                df[r] = X[r].values
        return df

    def predict(self, X: pd.DataFrame, return_std: bool = False):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        future = self._prepare_prophet_data(X)
        fc = self.model.predict(future)
        yhat = np.maximum(fc['yhat'].values, 0)
        if return_std:
            std = (fc['yhat_upper'] - fc['yhat_lower']) / (2 * 1.96)
            return yhat, std.values
        return yhat

    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles: List[float] = [0.025, 0.25, 0.5, 0.75, 0.975]):
        future = self._prepare_prophet_data(X)
        fc = self.model.predict(future)
        y = fc['yhat'].values
        lo = fc['yhat_lower'].values
        hi = fc['yhat_upper'].values
        std = (hi - lo) / (2 * 1.96)
        out = {}
        from scipy.stats import norm
        for q in quantiles:
            if np.isclose(q, 0.5):
                out[q] = np.maximum(y, 0)
            elif np.isclose(q, 0.025):
                out[q] = np.maximum(lo, 0)
            elif np.isclose(q, 0.975):
                out[q] = np.maximum(hi, 0)
            else:
                z = norm.ppf(q)
                out[q] = np.maximum(y + z * std, 0)
        return out

    def __repr__(self) -> str:
        return f"ProphetForecastModel ({'trained' if self.is_trained else 'untrained'}, {len(self.regressors)} regressors)"
