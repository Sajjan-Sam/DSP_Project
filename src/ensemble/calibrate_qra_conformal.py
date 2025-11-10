import os, numpy as np, pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split

def evaluate_interval(y, lower, upper, alpha=0.1):
    y = np.asarray(y)
    coverage = np.mean((y >= lower) & (y <= upper))
    width = np.mean(upper - lower)
    return coverage, width

def main():
    os.makedirs("results", exist_ok=True)

    # load MoE predictions and actuals
    moe = np.load("predictions/ensemble_moe.npz")
    val_pred, test_pred = moe["val"], moe["test"]
    y_val = pd.read_csv("data/y_val.csv")["y"].values
    y_test = pd.read_csv("data/y_test.csv")["y"].values

    # Split val into train/cal for conformal
    y_train, y_cal, yh_train, yh_cal = train_test_split(
        y_val, val_pred, test_size=0.25, random_state=42
    )

    quantiles = [0.1, 0.5, 0.9]
    qmodels = {}
    for q in quantiles:
        qr = QuantileRegressor(quantile=q, alpha=0, solver="highs")
        qr.fit(yh_train.reshape(-1,1), y_train)
        qmodels[q] = qr
        print(f"Fitted QRA quantile={q}")

    # Predict quantiles on cal and test
    cal_preds = {q: qmodels[q].predict(yh_cal.reshape(-1,1)) for q in quantiles}
    test_preds = {q: qmodels[q].predict(test_pred.reshape(-1,1)) for q in quantiles}

    # Split-Conformal adjustment
    residuals_lower = y_cal - cal_preds[0.1]
    residuals_upper = cal_preds[0.9] - y_cal
    q_lower = np.quantile(residuals_lower, 0.9)
    q_upper = np.quantile(residuals_upper, 0.9)

    lower_adj = test_preds[0.1] - q_lower
    upper_adj = test_preds[0.9] + q_upper
    median_adj = test_preds[0.5]

    coverage, width = evaluate_interval(y_test, lower_adj, upper_adj)
    mae = np.mean(np.abs(median_adj - y_test))

    print(f"Conformal coverage={coverage*100:.2f}% width={width:.3f} MAE={mae:.3f}")

    out = pd.DataFrame({
        "timestamp": pd.read_csv("data/y_test.csv")["timestamp"],
        "y_true": y_test,
        "p10": lower_adj,
        "p50": median_adj,
        "p90": upper_adj
    })
    out.to_csv("results/qra_conformal_forecasts.csv", index=False)
    print("[OK] saved results/qra_conformal_forecasts.csv")

if __name__ == "__main__":
    main()
