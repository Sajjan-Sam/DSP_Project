import os
import numpy as np
import pandas as pd

# Paths where your base model predictions are saved
PRED_DIR = "predictions"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# Load the actual y arrays used for validation/test (true targets)
# Replace the file names if you saved them differently
y_val_path = os.path.join(PRED_DIR, "y_val.npy")
y_test_path = os.path.join(PRED_DIR, "y_test.npy")

# If you have timestamps stored somewhere, load them — else create dummy indices
if os.path.exists(os.path.join(PRED_DIR, "timestamps_val.npy")):
    timestamps_val = np.load(os.path.join(PRED_DIR, "timestamps_val.npy"))
else:
    timestamps_val = pd.date_range("2020-06-01", periods=len(np.load(y_val_path)), freq="10min")

if os.path.exists(os.path.join(PRED_DIR, "timestamps_test.npy")):
    timestamps_test = np.load(os.path.join(PRED_DIR, "timestamps_test.npy"))
else:
    timestamps_test = pd.date_range("2020-07-01", periods=len(np.load(y_test_path)), freq="10min")

# Build DataFrames
y_val = pd.DataFrame({
    "timestamp": pd.to_datetime(timestamps_val),
    "y": np.load(y_val_path)
})
y_test = pd.DataFrame({
    "timestamp": pd.to_datetime(timestamps_test),
    "y": np.load(y_test_path)
})

# Save them
y_val.to_csv(os.path.join(OUT_DIR, "y_val.csv"), index=False)
y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

print(" Saved:")
print("  data/y_val.csv  ->", y_val.shape)
print("  data/y_test.csv ->", y_test.shape)
