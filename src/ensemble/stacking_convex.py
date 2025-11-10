import argparse, json, os, numpy as np, pandas as pd
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred, eps=1e-9):
    a = np.abs(y_true); b = np.abs(y_pred)
    return 100.0 * np.mean(2.0*np.abs(y_pred - y_true) / (a + b + eps))

def simplex_grid(n, step=0.05):
    """All length-n nonnegative vectors summing to 1 on a step grid."""
    # generate (n-1) variables, last = 1 - sum; reject negatives
    vals = np.arange(0.0, 1.0+1e-9, step)
    best = []
    if n == 1:
        return [np.array([1.0])]
    def rec(prefix, depth, remaining):
        if depth == n-1:
            last = 1.0 - sum(prefix)
            if last >= -1e-9:
                v = np.array(prefix+[max(0.0,last)])
                # snap to grid to avoid tiny negative due to fp errors
                v = np.clip(v, 0.0, 1.0)
                if abs(v.sum()-1.0) < 1e-6:
                    best.append(v)
            return
        for x in vals:
            if sum(prefix)+x <= 1.0+1e-12:
                rec(prefix+[float(x)], depth+1, remaining-x)
    rec([], 0, 1.0)
    return best

def score_weights(y, X, w, metric="mae"):
    yhat = X @ w
    if metric == "mae":
        return mean_absolute_error(y, yhat), yhat
    elif metric == "rmse":
        return np.sqrt(mean_squared_error(y, yhat)), yhat
    else:
        raise ValueError("metric must be mae or rmse")

def fit_convex_global(y_val, X_val, step=0.05, metric="mae"):
    grid = simplex_grid(X_val.shape[1], step=step)
    best = None
    for w in grid:
        m, _ = score_weights(y_val, X_val.values, w, metric)
        if (best is None) or (m < best[0]):
            best = (m, w)
    return best[1]  # weights

def fit_convex_by_regime(y_val, X_val, regimes, step=0.05, metric="mae"):
    """Return dict: regime -> weights (each sums to 1)."""
    out = {}
    for r in sorted(pd.unique(regimes)):
        ix = (regimes == r)
        w = fit_convex_global(y_val[ix], X_val[ix], step=step, metric=metric)
        out[int(r)] = w.tolist()
    return out

def apply_weights(w, X):
    return X.values @ w

def apply_weights_by_regime(w_map, X, regimes):
    yhat = np.zeros(len(X), dtype=float)
    for r in sorted(pd.unique(regimes)):
        ix = (regimes == r)
        w = np.array(w_map[int(r)])
        yhat[ix] = X[ix].values @ w
    return yhat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_path", default="data/gate/val.parquet")
    ap.add_argument("--test_path", default="data/gate/test.parquet")
    ap.add_argument("--by_regime", action="store_true",
                    help="learn separate convex weights for each 'regime' value")
    ap.add_argument("--step", type=float, default=0.05,
                    help="grid step on the simplex (smaller=more precise, slower)")
    ap.add_argument("--metric", default="mae", choices=["mae","rmse"])
    ap.add_argument("--out_preds", default="predictions/ensemble_convex.npy")
    ap.add_argument("--out_metrics", default="results/ensemble_results.csv")
    ap.add_argument("--out_model", default="models/stacking_convex.json")
    args = ap.parse_args()

    os.makedirs("predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    v = pd.read_parquet(args.val_path)
    t = pd.read_parquet(args.test_path)

    experts = ["pred_prophet", "pred_lstm", "pred_qrf", "pred_sarima"]
    for c in experts:
        if c not in v.columns or c not in t.columns:
            raise ValueError(f"Missing expert column {c}")

    yv = v["y"].astype(float)
    yt = t["y"].astype(float)
    Xv = v[experts].astype(float)
    Xt = t[experts].astype(float)

    if args.by_regime:
        if "regime" not in v.columns or "regime" not in t.columns:
            raise ValueError("regime column not found in gate parquet for --by_regime")
        w_map = fit_convex_by_regime(yv, Xv, v["regime"].astype(int), step=args.step, metric=args.metric)
        yv_hat = apply_weights_by_regime(w_map, Xv, v["regime"].astype(int))
        yt_hat = apply_weights_by_regime(w_map, Xt, t["regime"].astype(int))
        model_obj = {"mode": "by_regime", "experts": experts, "step": args.step, "metric": args.metric, "weights_by_regime": w_map}
    else:
        w = fit_convex_global(yv, Xv, step=args.step, metric=args.metric)
        yv_hat = apply_weights(w, Xv)
        yt_hat = apply_weights(w, Xt)
        model_obj = {"mode": "global", "experts": experts, "step": args.step, "metric": args.metric, "weights": w.tolist()}

    # metrics
    mae_v  = float(mean_absolute_error(yv, yv_hat))
    rmse_v = float(np.sqrt(mean_squared_error(yv, yv_hat)))
    smape_v= float(smape(yv.values, yv_hat))
    mae_t  = float(mean_absolute_error(yt, yt_hat))
    rmse_t = float(np.sqrt(mean_squared_error(yt, yt_hat)))
    smape_t= float(smape(yt.values, yt_hat))

    # save
    np.save(args.out_preds, yt_hat)
    with open(args.out_model, "w") as f:
        json.dump(model_obj, f, indent=2)

    # append metrics
    header = ["Model","MAE","RMSE","MAPE","sMAPE","R2","NRMSE","Split"]
    row_v = ["Convex Stacking (global)" if model_obj["mode"]=="global" else "Convex Stacking (by regime)",
             mae_v, rmse_v, "", smape_v, "", "", "Val"]
    row_t = ["Convex Stacking (global)" if model_obj["mode"]=="global" else "Convex Stacking (by regime)",
             mae_t, rmse_t, "", smape_t, "", "", "Test"]
    if os.path.exists(args.out_metrics):
        df = pd.read_csv(args.out_metrics)
        df.loc[len(df)] = row_v
        df.loc[len(df)] = row_t
    else:
        df = pd.DataFrame([row_v, row_t], columns=header)
    df.to_csv(args.out_metrics, index=False)

    print("[OK]", " | ".join([f"{k}={v}" for k,v in model_obj.items() if k!='weights_by_regime']))
    print(f"VAL -> MAE {mae_v:.3f} | RMSE {rmse_v:.3f} | sMAPE {smape_v:.3f}%")
    print(f"TEST-> MAE {mae_t:.3f} | RMSE {rmse_t:.3f} | sMAPE {smape_t:.3f}%")
    print(f"[SAVED] {args.out_preds}, {args.out_metrics}, {args.out_model}")
