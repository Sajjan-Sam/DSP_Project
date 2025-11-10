import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import FIGURES_DIR, LSTM_CONFIG, MODELS_DIR, PROPHET_CONFIG, QRF_CONFIG, RESULTS_DIR, SARIMA_CONFIG, SPLITS_DIR
# --- Fallback configs if not defined in src.utils.config ---
try: PROPHET_CONFIG
except NameError: PROPHET_CONFIG = {}
try: LSTM_CONFIG
except NameError: LSTM_CONFIG = {}
try: QRF_CONFIG
except NameError: QRF_CONFIG = {}
try: SARIMA_CONFIG
except NameError: SARIMA_CONFIG = {}
from src.utils.logger import get_logger
from src.models.quantile_rf import QuantileRFForecastModel
from src.models.prophet_model import ProphetForecastModel
from src.models.lstm_model import LSTMForecastModel, LSTMNet
from src.models.sarima_model import SARIMAForecastModel
from src.ensemble.attention_fusion import AttentionEnsemble
from src.ensemble.conformal_prediction import ConformalPI

logger = get_logger("ensemble_training")

# ---------------- data ----------------
logger.info("Loading data")
train = pd.read_csv(SPLITS_DIR / "plant1_train.csv")
val   = pd.read_csv(SPLITS_DIR / "plant1_val.csv")
test  = pd.read_csv(SPLITS_DIR / "plant1_test.csv")

priority = ['TOTAL_AC_POWER','AC_POWER','POWER_AC','TOTAL_DC_POWER','DC_POWER','POWER_DC']
target_cols = [c for c in priority if c in train.columns]
if not target_cols:
    target_cols = [c for c in train.columns if 'POWER' in c.upper()]
TARGET = target_cols[0]

X_tr, y_tr = train.drop(columns=[TARGET]), train[TARGET]
X_va, y_va = val.drop(columns=[TARGET]),   val[TARGET]
X_te, y_te = test.drop(columns=[TARGET]),  test[TARGET]

logger.info(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")
logger.info(f"Target: {TARGET}")
context_cols = [c for c in ["IRRADIATION","AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","solar_elevation","hour"] if c in X_tr.columns or c=="hour"]

# ------------- helpers -------------
def evaluate(y_true, y_pred, name, X_ref=None):
    y_true = pd.Series(y_true).reset_index(drop=True).astype(float)
    y_pred = pd.Series(y_pred).reset_index(drop=True).astype(float)
    # daylight mask
    if X_ref is not None:
        cand = [c for c in X_ref.columns if c.upper() in ("IRRADIATION","GHI","THEORETICAL_IRRADIANCE")]
        daylight = (pd.to_numeric(X_ref[cand[0]], errors="coerce") > 1.0) if cand else (y_true > 50)
        daylight = daylight.reset_index(drop=True)
    else:
        daylight = (y_true > 50)
    mask = daylight & y_pred.notna()
    yt = y_true[mask].values
    yp = y_pred[mask].values
    if len(yt) == 0:
        yt = y_true.values; yp = y_pred.values

    mae  = np.mean(np.abs(yp-yt))
    rmse = float(np.sqrt(np.mean((yp-yt)**2)))
    ss_res = np.sum((yt-yp)**2); ss_tot = np.sum((yt-yt.mean())**2)
    r2 = 1 - ss_res/(ss_tot + 1e-9)
    mape  = 100*np.mean(np.abs((yp-yt)/np.maximum(yt,1e-6)))
    smape = 100*np.mean(2*np.abs(yp-yt)/(np.abs(yt)+np.abs(yp)+1e-6))
    nrmse = 100*rmse/(yt.max()-yt.min()+1e-9)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "R2": r2, "NRMSE": nrmse, "Split": "Test"}



def smart_predict(model, X, name):
    """
    Robust prediction wrapper:
    - Uses wrapper .predict when available
    - Rebuilds LSTM network on-the-fly if it's None (after load)
    - For Prophet wrappers, fills any missing regressors with 0.0 and retries
    - Falls back to raw model.predict(X)
    """
    import numpy as _np
    import pandas as _pd

    # 1) Our wrappers first
    try:
        cname = type(model).__name__
        if cname.endswith("ForecastModel") and hasattr(model, "predict"):
            # LSTM: rebuild network if missing
            try:
                from src.models.lstm_model import LSTMForecastModel as _LFM, LSTMNet as _LSTMNet
                if isinstance(model, _LFM) and getattr(model, "network", None) is None:
                    input_dim = len(getattr(model, "feature_names", [])) or X.select_dtypes(include=_np.number).shape[1]
                    try:
                        model.network = _LSTMNet(
                            input_dim=input_dim,
                            hidden_dim=model.hidden_dim,
                            num_layers=model.num_layers,
                            dropout=model.dropout,
                        ).to(model.device)
                        if getattr(model, "best_state", None) is not None:
                            try: model.network.load_state_dict(model.best_state, strict=False)
                            except Exception: pass
                        model.is_trained = True
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                return _np.asarray(model.predict(X), dtype=float)
            except Exception:
                # Prophet-style: regressors missing; zero-fill and retry
                needed = []
                for attr in ("regressor_names", "extra_regressors", "_regressors", "regressors_"):
                    try:
                        obj = getattr(model, attr)
                        if isinstance(obj, dict): needed += list(obj.keys())
                        elif isinstance(obj, (list, tuple, set)): needed += list(obj)
                    except Exception:
                        pass
                needed = list(dict.fromkeys(needed))
                if needed:
                    X_aug = X.copy()
                    for c in needed:
                        if c not in X_aug.columns:
                            X_aug[c] = 0.0
                    try:
                        return _np.asarray(model.predict(X_aug), dtype=float)
                    except Exception:
                        pass
                raise
    except Exception:
        pass

    # 2) Raw Prophet support (rare path)
    try:
        from prophet import Prophet as _P
    except Exception:
        try:
            from fbprophet import Prophet as _P
        except Exception:
            _P = None
    if _P is not None and isinstance(model, _P):
        df = _pd.DataFrame(index=X.index)
        ds = None
        for cand in ["DATE_TIME","ds","date","timestamp","time"]:
            if cand in X.columns:
                ds = _pd.to_datetime(X[cand], errors="coerce"); break
        if ds is None and _pd.api.types.is_datetime64_any_dtype(X.index):
            ds = _pd.to_datetime(X.index)
        if ds is None:
            raise ValueError("Prophet requires a datetime column or DatetimeIndex.")
        df["ds"] = ds
        reg_names = list(getattr(model, "extra_regressors", {}).keys()) if hasattr(model, "extra_regressors") else []
        for r in reg_names:
            df[r] = _pd.to_numeric(X[r], errors="coerce").fillna(0.0).values if r in X.columns else 0.0
        return model.predict(df)["yhat"].values.astype(float)

    # 3) Generic
    return _np.asarray(model.predict(X), dtype=float)
        except Exception as e:
            # Prophet-like error handling
            if "Regressor" in str(e) or "Prophet" in str(type(model)):
                df = pd.DataFrame(index=X.index)
                ds = None
                for cand in ["DATE_TIME", "ds", "date", "timestamp", "time"]:
                    if cand in X.columns:
                        ds = pd.to_datetime(X[cand], errors="coerce")
                        break
                if ds is None and pd.api.types.is_datetime64_any_dtype(X.index):
                    ds = pd.to_datetime(X.index)
                if ds is None:
                    ds = pd.date_range("2000-01-01", periods=len(X), freq="H")
                df["ds"] = ds

                reg_names = []
                if hasattr(model, "model") and hasattr(model.model, "extra_regressors"):
                    reg_names = list(model.model.extra_regressors.keys())
                elif hasattr(model, "extra_regressors"):
                    reg_names = list(model.extra_regressors.keys())

                for r in reg_names:
                    df[r] = pd.to_numeric(X[r], errors="coerce").fillna(0.0).values if r in X.columns else 0.0

                try:
                    if hasattr(model, "model"):
                        return model.model.predict(df)["yhat"].values.astype(float)
                except Exception:
                    pass
            raise

    # Case 2: Raw Prophet instances
    try:
        from prophet import Prophet as _P
    except Exception:
        try:
            from fbprophet import Prophet as _P
        except Exception:
            _P = None
    if _P is not None and isinstance(model, _P):
        df = pd.DataFrame(index=X.index)
        ds = None
        for cand in ["DATE_TIME", "ds", "date", "timestamp", "time"]:
            if cand in X.columns:
                ds = pd.to_datetime(X[cand], errors="coerce")
                break
        if ds is None and pd.api.types.is_datetime64_any_dtype(X.index):
            ds = pd.to_datetime(X.index)
        if ds is None:
            ds = pd.date_range("2000-01-01", periods=len(X), freq="H")
        df["ds"] = ds

        reg_names = list(getattr(model, "extra_regressors", {}).keys()) if hasattr(model, "extra_regressors") else []
        for r in reg_names:
            df[r] = pd.to_numeric(X[r], errors="coerce").fillna(0.0).values if r in X.columns else 0.0

        return model.predict(df)["yhat"].values.astype(float)

    # Case 3: Generic sklearn-like model
    return np.asarray(model.predict(X), dtype=float)


# ------------- load models -------------


# ------------- load models -------------
    # Use the models you saved in 03_Individual_Models.py
    prophet_path = MODELS_DIR / "prophet_plant1.pkl"
    lstm_path    = MODELS_DIR / "lstm_plant1.pkl"
    qrf_path     = MODELS_DIR / "quantile_rf_plant1.pkl"
    sarima_path  = MODELS_DIR / "sarima_plant1.pkl"  # if available

def _load_or_none_instance(path, cls, cfg):
    """
    Try to load a saved model instance at 'path'.
    1) Instantiate the class with 'cfg' and call .load(path) (which may return None)
    2) Fallback to joblib.load(path) if class .load fails
    """
    if path.exists():
        try:
            obj = cls(cfg)
            # IMPORTANT: many implementations of .load() mutate self and return None
            _ret = obj.load(path)
            # if .load returned a new instance, prefer it; else keep obj
            if _ret is not None:
                obj = _ret
            from inspect import isclass
            name = getattr(path, "name", str(path))
            print(f"Model loaded: {path}")
            import logging
            try:
                logger.info(f"✓ Loaded {name} via class.load()")
            except Exception:
                pass
            return obj
        except Exception as e:
            try:
                import logging
                logger.warning(f"class.load failed for {path.name}: {e}; trying joblib.load")
            except Exception:
                pass
            try:
                from joblib import load as joblib_load
                obj = joblib_load(path)
                try:
                    logger.info(f"✓ Loaded {path.name} via joblib.load")
                except Exception:
                    pass
                return obj
            except Exception as e2:
                try:
                    logger.warning(f"joblib.load failed for {path.name}: {e2}")
                except Exception:
                    pass
    else:
        try:
            logger.warning(f"Missing model file: {path}")
        except Exception:
            pass
    return None

# --- [FORCE-DEFINE MODEL PATHS BEFORE LOADING] ---
# These definitions are unconditional to avoid NameError at load time.
prophet_path = MODELS_DIR / "prophet_plant1.pkl"
lstm_path    = MODELS_DIR / "lstm_plant1.pkl"
qrf_path     = MODELS_DIR / "quantile_rf_plant1.pkl"
sarima_path  = MODELS_DIR / "sarima_plant1.pkl"
# -------------------------------------------------

prophet = _load_or_none_instance(prophet_path, ProphetForecastModel, PROPHET_CONFIG)
lstm    = _load_or_none_instance(lstm_path,    LSTMForecastModel,   LSTM_CONFIG)
qrf     = _load_or_none_instance(qrf_path,     QuantileRFForecastModel, QRF_CONFIG)
sarima  = _load_or_none_instance(sarima_path,  SARIMAForecastModel,  SARIMA_CONFIG)

# ---- Diagnostics for loaded models ----
for _name, _obj in [("Prophet", prophet), ("LSTM", lstm), ("Quantile RF", qrf), ("SARIMA", sarima)]:
    try:
        logger.info(f"{_name} loaded? is None={_obj is None}; type={type(_obj)}")
    except Exception as _e:
        logger.warning(f"{_name} diagnostics failed: {_e}")

# ---- Build base_models deterministically ----
base_models = [
    ("Prophet", prophet),
    ("LSTM", lstm),
    ("Quantile RF", qrf),
    ("SARIMA", sarima),
]
base_models = [(n, m) for (n, m) in base_models if m is not None]
logger.info(f"Using {len(base_models)} models: {[n for n,_ in base_models]}")
if not base_models:
    raise SystemExit("No base models loaded. Re-run 03_Individual_Models first so .pkl files exist.")

# ------------- predictions -------------
# ------------- predictions -------------


def collect_preds(models, X_tr, y_tr, X_va, y_va, X_te):
    preds_val, preds_test, ok = {}, {}, []
    for name, model in models:
        logger.info(f"Predicting with {name}...")
        try:
            yp_v = smart_predict(model, X_va, name)
            yp_t = smart_predict(model, X_te, name)
            preds_val[name]  = yp_v
            preds_test[name] = yp_t
            ok.append(name)
        except Exception as e:
            logger.error(f"{name} prediction failed: {e}")
    return preds_val, preds_test, ok

preds_val, preds_test, ok_models = collect_preds(base_models, X_tr, y_tr, X_va, y_va, X_te)

# ------------- attention ensemble -------------
att_models = list(preds_val.keys())  # only models that produced VAL preds
if len(att_models) < 1:
    raise SystemExit('No model produced validation predictions; fix base models first.')
att = AttentionEnsemble(model_names=att_models, context_cols=context_cols, epochs=150, lr=1e-3, batch_size=128)
att.fit(preds_val, y_va, X_va, verbose=True)
yhat_val, w_val = att.predict({k: preds_val[k] for k in att_models}, X_va)
yhat_test, w_te = att.predict({k: preds_test[k] for k in att_models}, X_te)
att_stats = att.get_attention_stats(w_te)
logger.info("Attention weight stats (test):\n" + att_stats.to_string(index=False))

# ------------- metrics + conformal -------------
def add_result(rows, y_true, y_pred, name):
    rows.append(evaluate(y_true, y_pred, name, X_ref=X_te))

rows = []
for name, _ in base_models:
    if name in preds_test:
        add_result(rows, y_te, preds_test[name], name)

add_result(rows, y_te, yhat_test, "Attention Ensemble")

# conformal PI (split using VAL residuals)
cp = ConformalPI(alpha=0.1).fit(y_va, yhat_val)
lo, hi = cp.predict(yhat_test)
rows.append({"Model": "Attention Ensemble (Conformal 90%)",
             "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan,
             "R2": np.nan, "NRMSE": np.nan, "Split": "Test"})

res_df = pd.DataFrame(rows)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out = RESULTS_DIR / "ensemble_results.csv"
res_df.to_csv(out, index=False)
logger.info(f"Saved: {out}")
print("\nTest results:\n", res_df.sort_values("RMSE", na_position='last').to_string(index=False))

# store predictions
pred_dir = ROOT / "predictions"
pred_dir.mkdir(exist_ok=True, parents=True)
pd.DataFrame({"y_test": y_te, **{f"pred_{k}": v for k,v in preds_test.items()},
              "pred_attention": yhat_test, "pi_lo": lo, "pi_hi": hi}).to_csv(pred_dir / "test_preds.csv", index=False)