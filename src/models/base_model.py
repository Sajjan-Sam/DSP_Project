from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

class BaseForecastModel(ABC):
    """
    Base class for forecasting models with a consistent interface.
    """

    def __init__(self, config: Dict, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.scaler = None
        self.training_history: Dict = {}
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "model_name": model_name,
            "config": config
        }

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, return_std: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        pass

    # NOTE: made concrete (no @abstractmethod) to avoid abstract-class errors
    def predict_quantiles(self, X: pd.DataFrame,
                          quantiles: List[float] = [0.025, 0.25, 0.5, 0.75, 0.975]) -> Dict[float, np.ndarray]:
        """
        Default quantile prediction using a normal approximation if the model
        supports predict(..., return_std=True). Subclasses can override.
        """
        preds = self.predict(X, return_std=True)
        if isinstance(preds, tuple) and len(preds) == 2:
            yhat, std = preds
            out: Dict[float, np.ndarray] = {}
            # z-scores via SciPy-inverse CDF is nice, but to keep dependencies
            # minimal we precompute common ones and fall back to numpy percentile.
            from math import sqrt
            # Simple approximate inverse-CDF for common quantiles
            z_lookup = {
                0.025: -1.959964, 0.05: -1.644854, 0.25: -0.674490,
                0.5: 0.0, 0.75: 0.674490, 0.95: 1.644854, 0.975: 1.959964
            }
            for q in quantiles:
                z = z_lookup.get(q, None)
                if z is None:
                    # Fallback using np.percentile on synthetic draws
                    # (small sample to stay lightweight)
                    draws = np.random.normal(loc=yhat, scale=std, size=(128, len(yhat)))
                    out[q] = np.maximum(np.percentile(draws, q*100, axis=0), 0)
                else:
                    out[q] = np.maximum(yhat + z * std, 0)
            return out
        raise NotImplementedError(
            f"{self.model_name} does not implement uncertainty; override predict_quantiles in the subclass."
        )

    def predict_intervals(self, X: pd.DataFrame, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = 1 - confidence
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        qdict = self.predict_quantiles(X, [lower_q, 0.5, upper_q])
        return (qdict[0.5], qdict[lower_q], qdict[upper_q])

    def save(self, filepath: Path) -> None:
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "config": self.config,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "training_history": self.training_history,
            "metadata": self.metadata,
            "is_trained": self.is_trained
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Model saved: {filepath}")

    def load(self, filepath: Path) -> None:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.config = state["config"]
        self.model_name = state["model_name"]
        self.feature_names = state["feature_names"]
        self.scaler = state.get("scaler")
        self.training_history = state.get("training_history", {})
        self.metadata = state.get("metadata", {})
        self.is_trained = state["is_trained"]
        print(f"Model loaded: {filepath}")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return None

    def validate_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series or numpy array.")
        if len(X) == 0:
            raise ValueError("Input data is empty.")
        if X.isnull().any().any():
            n_missing = int(X.isnull().sum().sum())
            print(f"Warning: input features contain {n_missing} missing values.")

    def get_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names) if self.feature_names else None,
            "config": self.config,
            "metadata": self.metadata,
            "training_history": self.training_history
        }

    def __repr__(self) -> str:
        return f"{self.model_name} ({'trained' if self.is_trained else 'untrained'})"

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is not None:
            missing = set(self.feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {missing}")
            return X[self.feature_names].copy()
        return X.copy()

    def _log_training_metric(self, metric_name: str, value: float, epoch: Optional[int] = None):
        if metric_name not in self.training_history:
            self.training_history[metric_name] = []
        if epoch is not None:
            self.training_history[metric_name].append({"epoch": epoch, "value": value})
        else:
            self.training_history[metric_name].append(value)
