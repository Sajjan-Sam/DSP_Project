import argparse, os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to raw dataset (parquet/csv).")
    ap.add_argument("--target", required=True, help="Target column name, e.g., TOTAL_AC_POWER")
    ap.add_argument("--timestamp", required=True, help="Timestamp column name, e.g., DATE_TIME or timestamp")
    ap.add_argument("--val_days", type=int, default=7)
    ap.add_argument("--test_days", type=int, default=7)
    ap.add_argument("--tz_localize", default=None, help="Optional timezone to localize (e.g., Asia/Kolkata)")
    args = ap.parse_args()

    raw = args.raw
    ext = os.path.splitext(raw)[1].lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(raw)
    else:
        df = pd.read_csv(raw)

    # basic checks
    if args.timestamp not in df.columns:
        raise ValueError(f"Timestamp column '{args.timestamp}' not found.")
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found.")

    # sort by time
    df = df[[args.timestamp, args.target]].copy()
    df[args.timestamp] = pd.to_datetime(df[args.timestamp])
    if args.tz_localize:
        # only localize if naive
        if df[args.timestamp].dt.tz is None:
            df[args.timestamp] = df[args.timestamp].dt.tz_localize(args.tz_localize)
    df = df.sort_values(args.timestamp).reset_index(drop=True)

    # derive split by days from the end
    end_time = df[args.timestamp].iloc[-1]
    test_start = end_time - pd.Timedelta(days=args.test_days) + pd.Timedelta(seconds=1)
    val_start  = test_start - pd.Timedelta(days=args.val_days)

    val_df  = df[(df[args.timestamp] > val_start) & (df[args.timestamp] <= test_start)]
    test_df = df[(df[args.timestamp] > test_start)]

    # save
    os.makedirs("data", exist_ok=True)
    val_df.rename(columns={args.timestamp:"timestamp", args.target:"y"}) \
          .to_csv("data/y_val.csv", index=False)
    test_df.rename(columns={args.timestamp:"timestamp", args.target:"y"}) \
           .to_csv("data/y_test.csv", index=False)

    print("saved data/y_val.csv", val_df.shape)
    print("saved data/y_test.csv", test_df.shape)

if __name__ == "__main__":
    main()
