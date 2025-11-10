import os, json, math
import numpy as np
import pandas as pd

def hour_sin_cos(dt):
    h = dt.hour + dt.minute/60
    return np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)

def main():
    # ---- paths (keep in sync with configs/moe.yaml) ----
    base = "predictions"
    y_val_csv = "data/y_val.csv"
    y_test_csv = "data/y_test.csv"
    meta_csv = "data/meta_features.csv"
    caps_csv = "data/plant_caps.csv"
    os.makedirs("data", exist_ok=True)

    # ----- load target splits -----
    y_val = pd.read_csv(y_val_csv, parse_dates=["timestamp"])
    y_test = pd.read_csv(y_test_csv, parse_dates=["timestamp"])

    # ----- load / create meta features -----
    if os.path.exists(meta_csv):
        meta = pd.read_csv(meta_csv, parse_dates=["timestamp"])
    else:
        # minimal meta derivation — replace with your own rich set if available
        meta = pd.concat([y_val[["timestamp"]], y_test[["timestamp"]]])
        meta = meta.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
        # placeholders (fill from your EDA tables if you have them)
        meta["irradiance"] = 0.0
        meta["ambient_temp"] = 25.0
        # regimes: if you already saved them, merge here; else default 0
        meta["regime"] = 0
        # hour features
        s, c = zip(*[hour_sin_cos(ts) for ts in meta["timestamp"]])
        meta["hour_sin"], meta["hour_cos"] = s, c
        meta.to_csv(meta_csv, index=False)

    caps = pd.read_csv(caps_csv, parse_dates=["timestamp"]) if os.path.exists(caps_csv) else None

    # ----- load base model predictions -----
    def load_preds(split):
        # expected npy/npz names; adjust to your filenames if needed
        names = ["prophet", "lstm", "qrf", "sarima"]
        out = {}
        for n in names:
            # support both .npy or .npz with key=preds
            p1 = os.path.join(base, f"{n}_{split}.npy")
            p2 = os.path.join(base, f"{n}_{split}.npz")
            if os.path.exists(p1):
                out[n] = np.load(p1)
            elif os.path.exists(p2):
                out[n] = np.load(p2)["preds"]
            else:
                raise FileNotFoundError(f"Missing {n}_{split} predictions in {base}")
        X = np.vstack([out[n] for n in names]).T  # [T, 4]
        return X, names

    Xv, names = load_preds("val")
    Xt, _ = load_preds("test")

    # ----- align by timestamp, join meta & caps -----
    def assemble(df_y, pred_mat):
        df = df_y.copy()
        df["idx"] = range(len(df))
        # sanity
        if pred_mat.shape[0] != len(df):
            raise ValueError(f"Pred length {pred_mat.shape[0]} != {len(df)}")
        for i, n in enumerate(names):
            df[f"pred_{n}"] = pred_mat[:, i]
        df = df.merge(meta, on="timestamp", how="left")
        if caps is not None:
            df = df.merge(caps, on="timestamp", how="left")
        # fallbacks
        for col in ["theoretical_irradiance", "plant_capacity"]:
            if col not in df:
                df[col] = np.nan
        return df

    val_df = assemble(y_val, Xv)
    test_df = assemble(y_test, Xt)

    os.makedirs("data/gate", exist_ok=True)
    val_df.to_parquet("data/gate/val.parquet", index=False)
    test_df.to_parquet("data/gate/test.parquet", index=False)

    # feature list for the gate (keep small and robust)
    features = (["pred_prophet","pred_lstm","pred_qrf","pred_sarima",
                "hour_sin","hour_cos","irradiance","ambient_temp","regime"])
    with open("data/gate/features.json","w") as f:
        json.dump({"features":features,"experts":["prophet","lstm","qrf","sarima"]}, f)

    print("saved: data/gate/val.parquet, test.parquet, features.json")

if __name__ == "__main__":
    main()
