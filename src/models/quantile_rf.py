import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor
from src.models.base_model import BaseForecastModel
import warnings
warnings.filterwarnings('ignore')

def _robust_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(np.float64, errors='ignore')
    # Replace ±inf with NaN, then fill
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    # Clip extreme outliers column-wise to the 0.1–99.9 percentile range
    cleaned = df.copy()
    for c in cleaned.columns:
        if np.issubdtype(cleaned[c].dtype, np.number):
            lo = np.nanpercentile(cleaned[c].values, 0.1)
            hi = np.nanpercentile(cleaned[c].values, 99.9)
            if not np.isfinite(lo): lo = np.nanmin(cleaned[c].values)
            if not np.isfinite(hi): hi = np.nanmax(cleaned[c].values)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                cleaned[c] = np.clip(cleaned[c].values, lo, hi)
            cleaned[c] = np.nan_to_num(cleaned[c].values, nan=0.0, posinf=hi, neginf=lo)
    return cleaned

class QuantileRandomForest:
    def __init__(self, **rf_params):
        self.rf = RandomForestRegressor(**rf_params)
        self.is_fitted = False

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.rf.predict(X)

    def predict_quantiles(self, X, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        tree_preds = np.array([t.predict(X) for t in self.rf.estimators_], dtype=np.float64)
        return {q: np.quantile(tree_preds, q, axis=0) for q in quantiles}

    def feature_importances_(self):
        return self.rf.feature_importances_

class QuantileRFForecastModel(BaseForecastModel):
    def __init__(self, config: Dict):
        super().__init__(config, "QuantileRF")
        self.n_estimators = config.get('n_estimators', 200)
        self.max_depth = config.get('max_depth', 15)
        self.min_samples_split = config.get('min_samples_split', 10)
        self.min_samples_leaf = config.get('min_samples_leaf', 5)
        self.max_features = config.get('max_features', 'sqrt')
        self.n_jobs = config.get('n_jobs', -1)
        self.random_state = config.get('random_state', 42)
        self.feature_importance_ = None

    def _select_features(self, X: pd.DataFrame) -> List[str]:
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if not any(x in c.upper() for x in ['ID', 'SOURCE', 'PLANT'])]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        self.validate_data(X_train, y_train)
        self.feature_names = self._select_features(X_train)

        Xtr = X_train[self.feature_names]
        Xtr = _robust_clean(Xtr)

        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        self.model = QuantileRandomForest(**params)
        self.model.fit(Xtr.values, y_train.values.astype(np.float64))
        self.is_trained = True
        self.feature_importance_ = self.model.feature_importances_()

    def predict(self, X: pd.DataFrame, return_std: bool = False):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        Xs = _robust_clean(X[self.feature_names])
        yhat = np.maximum(self.model.predict(Xs.values), 0)
        if return_std:
            qs = self.model.predict_quantiles(Xs.values, [0.025, 0.975])
            std = (qs[0.975] - qs[0.025]) / (2 * 1.96)
            return yhat, std
        return yhat

    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles: List[float] = [0.025, 0.25, 0.5, 0.75, 0.975]):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        Xs = _robust_clean(X[self.feature_names])
        out = self.model.predict_quantiles(Xs.values, quantiles)
        for q in list(out.keys()):
            out[q] = np.maximum(out[q], 0)
        return out

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        import pandas as pd
        imp = pd.DataFrame({'feature': self.feature_names, 'importance': self.feature_importance_})
        return imp.sort_values('importance', ascending=False).head(top_n)

    def __repr__(self) -> str:
        return f"QuantileRFForecastModel ({'trained' if self.is_trained else 'untrained'}, {self.n_estimators} trees)"
