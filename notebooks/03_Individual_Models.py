import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root so "src" and "notebooks" imports work when run as a module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import *  # assumes you already have these constants (DIRs & CONFIGs)
from src.utils.logger import get_logger
from src.models.prophet_model import ProphetForecastModel
from src.models.sarima_model import SARIMAForecastModel
from src.models.lstm_model import LSTMForecastModel
from src.models.quantile_rf import QuantileRFForecastModel

logger = get_logger("individual_models")

# -------------------- Load data --------------------
logger.info("Loading prepared data")
train = pd.read_csv(SPLITS_DIR / 'plant1_train.csv')
val   = pd.read_csv(SPLITS_DIR / 'plant1_val.csv')
test  = pd.read_csv(SPLITS_DIR / 'plant1_test.csv')
logger.info(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")

# Prefer AC power if present; otherwise fall back.
priority = [
    'TOTAL_AC_POWER', 'AC_POWER', 'POWER_AC',
    'TOTAL_DC_POWER', 'DC_POWER', 'POWER_DC'
]
target_cols = [c for c in priority if c in train.columns]
if not target_cols:
    fallback = [c for c in train.columns if 'POWER' in c.upper()]
    if not fallback:
        raise RuntimeError("No power column found.")
    target_cols = [fallback[0]]
TARGET = target_cols[0]
logger.info(f"Target column: {TARGET}")

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_val,   y_val   = val.drop(columns=[TARGET]),   val[TARGET]
X_test,  y_test  = test.drop(columns=[TARGET]),  test[TARGET]
logger.info(f"Features: {X_train.shape[1]} | Target range: {y_train.min():.2f}–{y_train.max():.2f}")

# -------------------- Helpers --------------------
def evaluate(y_true, y_pred, name, X_ref=None, power_col=None):
    """
    Computes MAE, RMSE, R2, NRMSE and robust daylight-only MAPE/sMAPE.
    """
    y_true = y_true.reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    # Daylight mask
    if X_ref is not None:
        cand = [c for c in X_ref.columns if c.upper() in ("IRRADIATION", "GHI", "THEORETICAL_IRRADIANCE")]
        if not cand:
            cand = [c for c in X_ref.columns if "IRRADIATION" in c.upper() or "GHI" in c.upper()]
        if cand:
            daylight = (pd.to_numeric(X_ref[cand[0]], errors="coerce") > 1.0).reset_index(drop=True)
        else:
            daylight = (y_true > 50).reset_index(drop=True)
    else:
        daylight = (y_true > 50).reset_index(drop=True)

    base_mask = y_pred.notna()
    mask = base_mask & daylight

    yt = y_true[mask].astype(float).values
    yp = y_pred[mask].astype(float).values
    if len(yt) == 0:
        yt = y_true[base_mask].astype(float).values
        yp = y_pred[base_mask].astype(float).values

    mae = np.mean(np.abs(yp - yt))
    rmse = np.sqrt(np.mean((yp - yt) ** 2))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    mape = 100 * np.mean(np.abs((yp - yt) / np.maximum(yt, 1e-6)))
    smape = 100 * np.mean(2 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-6))
    nrmse = 100 * rmse / (yt.max() - yt.min() + 1e-9)

    return {
        'Model': name, 'MAE': mae, 'RMSE': rmse,
        'MAPE': mape, 'sMAPE': smape,
        'R2': r2, 'NRMSE': nrmse, 'Split': 'Test'
    }


def plot_predictions(y_true, preds_dict, title, save_path=None):
    plt.figure(figsize=(16, 6))
    plt.plot(y_true.values, label='Ground Truth', color='black', linewidth=2, alpha=0.7)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for i, (name, preds) in enumerate(preds_dict.items()):
        plt.plot(preds, label=name, color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
    plt.xlabel('Time step')
    plt.ylabel('Power (W)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")
    plt.close()

results = []
preds_all = {}

# -------------------- Prophet --------------------
try:
    logger.info("Training Prophet")
    m = ProphetForecastModel(PROPHET_CONFIG)
    m.fit(X_train, y_train, X_val, y_val)  # no unsupported kwargs
    yp = m.predict(X_test)
    results.append(evaluate(y_test, yp, 'Prophet', X_ref=X_test))
    preds_all['Prophet'] = yp
    m.save(MODELS_DIR / 'prophet_plant1.pkl')
except Exception as e:
    logger.error(f"Prophet error: {e}")

# -------------------- SARIMA --------------------
try:
    logger.info("Training SARIMA")
    sarima_cfg = SARIMA_CONFIG if 'SARIMA_CONFIG' in globals() else {}
    sar = SARIMAForecastModel(sarima_cfg)
    sar.fit(X_train, y_train, X_val, y_val)
    yp_val = sar.predict(X_val)
    yp = sar.predict(X_test)
    preds_all['SARIMA_val'] = yp_val
    preds_all['SARIMA'] = yp
    results.append(evaluate(y_test, yp, 'SARIMA', X_ref=X_test))
    sar.save(MODELS_DIR / 'sarima_plant1.pkl')
except Exception as e:
    logger.error(f"SARIMA error: {e}")


# -------------------- LSTM --------------------
try:
    logger.info("Training LSTM")
    lc = LSTM_CONFIG.copy()
    lc['epochs'] = max(20, lc.get('epochs', 50))
    lstm = LSTMForecastModel(lc)
    lstm.fit(X_train, y_train, X_val, y_val)
    yp_val = lstm.predict(X_val)
    yp = lstm.predict(X_test)
    preds_all['LSTM_val'] = yp_val
    results.append(evaluate(y_test, yp, 'LSTM', X_ref=X_test))
    preds_all['LSTM'] = yp
    lstm.save(MODELS_DIR / 'lstm_plant1.pkl')
except Exception as e:
    logger.error(f"LSTM error: {e}")

# -------------------- Quantile RF --------------------
try:
    logger.info("Training Quantile RF")
    qrf = QuantileRFForecastModel(QRF_CONFIG)
    qrf.fit(X_train, y_train, X_val, y_val)
    yp_val = qrf.predict(X_val)
    yp = qrf.predict(X_test)
    preds_all['Quantile RF_val'] = yp_val
    results.append(evaluate(y_test, yp, 'Quantile RF', X_ref=X_test))
    preds_all['Quantile RF'] = yp
    qrf.save(MODELS_DIR / 'quantile_rf_plant1.pkl')
except Exception as e:
    logger.error(f"Quantile RF error: {e}")

# -------------------- Results & Plots --------------------









# ----- Simple Stacking (RidgeCV, non-negative, std. preds) -----
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

def daylight_mask(X_ref, y):
    cand = [c for c in X_ref.columns if c.upper() in ("IRRADIATION","GHI","THEORETICAL_IRRADIANCE")]
    if cand: return (np.asarray(X_ref[cand[0]]).astype(float) > 1.0)
    return (np.asarray(y) > 50)

val_mask = daylight_mask(X_val, y_val)
test_mask = daylight_mask(X_test, y_test)

cols_val = [k for k in ["Prophet_val","SARIMA_val","LSTM_val","Quantile RF_val"] if k in preds_all]
if len(cols_val) >= 2:
    Xv_cols = [np.asarray(preds_all[k]) for k in cols_val]
    Xv = np.vstack([c[val_mask] for c in Xv_cols]).T
    yv = np.asarray(y_val)[val_mask]

    # Impute + standardize per feature
    for j in range(Xv.shape[1]):
        med = np.nanmedian(Xv[:, j])
        if np.isnan(med): med = 0.0
        Xv[np.isnan(Xv[:, j]), j] = med
    scaler = StandardScaler()
    Xv = scaler.fit_transform(Xv)

    # Non-negative weights via projected gradient on top of RidgeCV fit
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    base = RidgeCV(alphas=alphas, fit_intercept=True)
    base.fit(Xv, yv)
    w = base.coef_.clip(min=0.0)
    if w.sum() == 0: w[:] = 1.0
    w /= w.sum()
    b = base.intercept_

    # TEST path: same preprocessing
    Xt_cols = [np.asarray(preds_all[c.replace("_val","")]) for c in cols_val]
    Xt = np.vstack([c[test_mask] for c in Xt_cols]).T
    for j in range(Xt.shape[1]):
        med = np.nanmedian(Xt[:, j])
        if np.isnan(med): med = 0.0
        Xt[np.isnan(Xt[:, j]), j] = med
    Xt = scaler.transform(Xt)

    yhat_stack = np.full_like(np.asarray(y_test), np.nan, dtype=float)
    yhat_stack[test_mask] = Xt.dot(w) + b

    stack_metrics = evaluate(y_test, yhat_stack, "Stacked (RidgeCV+NN)", X_ref=X_test)
    results.append(stack_metrics)
    preds_all['Stacked (RidgeCV+NN)'] = yhat_stack

res_df = pd.DataFrame(results)
print("\nModel comparison (Test):\n", res_df.to_string(index=False))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
res_df.to_csv(RESULTS_DIR / 'individual_models_results.csv', index=False)
logger.info(f"Results saved: {RESULTS_DIR / 'individual_models_results.csv'}")

plot_predictions(
    y_test[:500],
    {k: v[:500] for k, v in preds_all.items()},
    "Individual Models - Test (first 500 points)",
    save_path=FIGURES_DIR / 'individual_models_predictions.png'
)

logger.info("Done.")
