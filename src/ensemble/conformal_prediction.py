from typing import Tuple
import numpy as np
import pandas as pd


class ConformalPI:
    """
    Split-conformal intervals for regression.
    Fit on (y_val, yhat_val), then wrap test predictions.
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.q_: float = np.nan

    def fit(self, y_val: np.ndarray, yhat_val: np.ndarray):
        y_val = np.asarray(y_val, dtype=float).ravel()
        yhat_val = np.asarray(yhat_val, dtype=float).ravel()
        resid = np.abs(y_val - yhat_val)
        # quantile with finite-sample correction
        k = int(np.ceil((1 - self.alpha) * (len(resid) + 1))) - 1
        k = min(max(k, 0), len(resid) - 1)
        self.q_ = float(np.partition(resid, k)[k])
        return self

    def predict(self, yhat_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert not np.isnan(self.q_), "Call fit() first."
        yhat_test = np.asarray(yhat_test, dtype=float).ravel()
        lo = yhat_test - self.q_
        hi = yhat_test + self.q_
        lo = np.maximum(lo, 0.0)
        return lo, hi
