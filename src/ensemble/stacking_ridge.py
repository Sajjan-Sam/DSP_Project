import argparse, json, os
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred, eps=1e-9):
    a = np.abs(y_true)
    b = np.abs(y_pred)
    return 100.0 * np.mean(2.0*np.abs(y_pred - y_true) / (a + b + eps))

def per_column_standardize_fit(X: pd.DataFrame):
    means = X.mean(axis=0)
    stds  = X.std(axis=0).replace(0, 1.0)
    Xs = (X - means) / stds
    return Xs, {"mean": means.to_dict(), "std": stds.to_dict()}

def per_column_standardize_apply(X: pd.DataFrame, stats: dict):
    means = pd.Series(stats["mean"])
    stds  = pd.Series(stats["std"]).replace(0, 1.0)
    means = means.reindex(X.columns)
    stds  = stds.reindex(X.columns).replace(0, 1.0)
    return (X - means) / stds

def tscv_ridge_alpha_search(X, y, alphas, n_splits=5):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for a in alphas:
        fold_mae = []
        for tr_idx, va_idx in splitter.split(X):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
            model = Ridge(alpha=a, fit_intercept=True, random_state=42)
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            fold_mae.append(mean_absolute_error(yva, pred))
        results.append({"alpha": float(a), "cv_mae": float(np.mean(fold_mae))})
    best = min(results, key=lambda r: r["cv_mae"])
    return best["alpha"], results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_path", default="data/gate/val.parquet")
    p.add_argument("--test_path", default="data/gate/test.parquet")
    p.add_argument("--out_preds", default="predictions/ensemble_stacking.npy")
    p.add_argument("--out_metrics", default="results/ensemble_results.csv")
    p.add_argument("--out_model", default="models/stacking_ridge.json")
    p.add_argument("--alphas", default="0.01,0.05,0.1,0.5,1.0,2.0,5.0,10.0")
    p.add_argument("--splits", type=int, default=5)
    args = p.parse_args()

    os.makedirs("predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1) Load datasets
    v = pd.read_parquet(args.val_path)
    t = pd.read_parquet(args.test_path)

    # 2) Experts
    expert_cols = ["pred_prophet", "pred_lstm", "pred_qrf", "pred_sarima"]
    for c in expert_cols:
        if c not in v.columns or c not in t.columns:
            raise ValueError(f"Missing expert column: {c}")

    yv = v["y"].astype(float)
    yt = t["y"].astype(float)
    Xv_raw = v[expert_cols].astype(float)
    Xt_raw = t[expert_cols].astype(float)

    # 3) Per-expert normalization (fit on VAL, apply to TEST)
    Xv, stats = per_column_standardize_fit(Xv_raw)
    Xt = per_column_standardize_apply(Xt_raw, stats)

    # 4) Time-series CV for alpha
    alphas = [float(a) for a in args.alphas.split(",")]
    best_alpha, cv_table = tscv_ridge_alpha_search(Xv, yv, alphas, n_splits=args.splits)

    # 5) Fit final on full VAL
    final = Ridge(alpha=best_alpha, fit_intercept=True, random_state=42)
    final.fit(Xv, yv)
    yv_hat = final.predict(Xv)
    yt_hat = final.predict(Xt)

    # 6) Metrics (compute RMSE manually for compatibility)
    mae_v  = float(mean_absolute_error(yv, yv_hat))
    mse_v  = float(mean_squared_error(yv, yv_hat))
    rmse_v = float(np.sqrt(mse_v))
    smape_v= float(smape(yv.values, yv_hat))

    mae_t  = float(mean_absolute_error(yt, yt_hat))
    mse_t  = float(mean_squared_error(yt, yt_hat))
    rmse_t = float(np.sqrt(mse_t))
    smape_t= float(smape(yt.values, yt_hat))

    # 7) Save predictions
    np.save(args.out_preds, yt_hat)

    # 8) Append metrics
    row_v = ["Stacking Ridge (norm experts)", mae_v, rmse_v, "", smape_v, "", "", "Val"]
    row_t = ["Stacking Ridge (norm experts)", mae_t, rmse_t, "", smape_t, "", "", "Test"]
    header = ["Model","MAE","RMSE","MAPE","sMAPE","R2","NRMSE","Split"]

    if os.path.exists(args.out_metrics):
        df = pd.read_csv(args.out_metrics)
        df.loc[len(df)] = row_v
        df.loc[len(df)] = row_t
    else:
        df = pd.DataFrame([row_v, row_t], columns=header)
    df.to_csv(args.out_metrics, index=False)

    # 9) Save model card
    model_card = {
        "type": "ridge_stacking",
        "expert_columns": expert_cols,
        "alpha": best_alpha,
        "cv_table": cv_table,
        "standardize_stats": stats,
        "val_metrics": {"MAE": mae_v, "RMSE": rmse_v, "sMAPE": smape_v},
        "test_metrics": {"MAE": mae_t, "RMSE": rmse_t, "sMAPE": smape_t},
    }
    with open(args.out_model, "w") as f:
        json.dump(model_card, f, indent=2)

    print(f"[OK] Stacking Ridge alpha={best_alpha}")
    print(f"VAL -> MAE {mae_v:.3f} | RMSE {rmse_v:.3f} | sMAPE {smape_v:.3f}%")
    print(f"TEST-> MAE {mae_t:.3f} | RMSE {rmse_t:.3f} | sMAPE {smape_t:.3f}%")
    print(f"[SAVED] {args.out_preds}, {args.out_metrics}, {args.out_model}")

if __name__ == "__main__":
    main()
