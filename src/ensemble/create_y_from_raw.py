import argparse, os, sys, glob
import numpy as np
import pandas as pd

PRED_DIR = "predictions"
OUT_DIR = "data"

CANDIDATE_TARGETS = [
    "y", "target", "power", "Power", "TOTAL_AC_POWER", "AC_POWER",
    "PLANT_POWER", "GEN_POWER", "P_AC"
]

def discover_raw_file():
    # Prefer Parquet, then CSV
    patterns = [
        "data/**/*.parquet",
        "data/**/*.pq",
        "data/**/*.feather",
        "data/**/*.csv",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    # filter out tiny helper files
    files = [f for f in files if os.path.getsize(f) > 10_000]  # >10KB
    # prefer names that look like merged/processed plant files
    ranked = sorted(files, key=lambda p: (
        0 if any(k in os.path.basename(p).lower() for k in ["merge","merged","processed","plant","final"]) else 1,
        0 if p.endswith((".parquet",".pq",".feather")) else 1,
        os.path.getsize(p) * -1
    ))
    return ranked[0] if ranked else None

def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    elif ext == ".feather":
        return pd.read_feather(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def choose_target(df, user_target=None):
    if user_target and user_target in df.columns:
        return user_target
    for c in CANDIDATE_TARGETS:
        if c in df.columns:
            return c
    # last resort: numeric column with highest correlation to a power-ish name
    numeric_cols = df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found to use as target.")
    return numeric_cols[0]

def read_len(path):
    if os.path.exists(path):
        arr = np.load(path)
        return int(arr.shape[0])
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="Path to raw/processed dataset (parquet/csv).")
    ap.add_argument("--target", default=None, help="Target column name if known (e.g., TOTAL_AC_POWER).")
    ap.add_argument("--timestamp", default=None, help="Timestamp column name if not 'timestamp'/'date'/'datetime'.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- discover VAL/TEST lengths from predictions ----
    val_len = read_len(os.path.join(PRED_DIR, "prophet_val.npy"))
    test_len = read_len(os.path.join(PRED_DIR, "prophet_test.npy"))
    if not val_len or not test_len:
        # try any base model
        for m in ["lstm", "qrf", "sarima"]:
            val_len = val_len or read_len(os.path.join(PRED_DIR, f"{m}_val.npy"))
            test_len = test_len or read_len(os.path.join(PRED_DIR, f"{m}_test.npy"))
    if not val_len or not test_len:
        print("❌ Could not infer val/test lengths from predictions/*.npy")
        print("   Expected files like prophet_val.npy and prophet_test.npy (or lstm/qrf/sarima).")
        sys.exit(1)

    # ---- locate raw data ----
    raw_path = args.raw or discover_raw_file()
    if not raw_path:
        print("❌ Could not find a raw dataset under data/**")
        print("   Please re-run with:  --raw path/to/your_merged.parquet  --target YOUR_TARGET_COL")
        sys.exit(1)
    print(f"✓ Using raw data: {raw_path}")

    df = load_table(raw_path)
    # standardize timestamp column
    ts_col = args.timestamp
    if ts_col is None:
        for c in ["timestamp", "Timestamp", "date", "datetime", "Date", "Datetime"]:
            if c in df.columns:
                ts_col = c; break
    if ts_col is None:
        raise ValueError("No timestamp column found. Use --timestamp to specify it.")

    # choose target
    tgt = choose_target(df, args.target)
    print(f"✓ Using target column: {tgt}")
    print(f"Columns available: {list(df.columns)[:10]} ...")

    # sort & trim to last VAL+TEST
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    needed = val_len + test_len
    if len(df) < needed:
        raise ValueError(f"Dataset too short: need {needed}, have {len(df)}")

    df_tail = df.iloc[-needed:].reset_index(drop=True)

    # split
    val_df = df_tail.iloc[:val_len][[ts_col, tgt]].rename(columns={ts_col:"timestamp", tgt:"y"})
    test_df = df_tail.iloc[val_len:][[ts_col, tgt]].rename(columns={ts_col:"timestamp", tgt:"y"})

    # save
    val_out = os.path.join(OUT_DIR, "y_val.csv")
    test_out = os.path.join(OUT_DIR, "y_test.csv")
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"✅ Saved {val_out} -> {val_df.shape}")
    print(f"✅ Saved {test_out} -> {test_df.shape}")

if __name__ == "__main__":
    main()
