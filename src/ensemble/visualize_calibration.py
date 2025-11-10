import os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    os.makedirs("figures", exist_ok=True)
    df = pd.read_csv("results/qra_conformal_forecasts.csv")
    y, p10, p50, p90 = df["y_true"], df["p10"], df["p50"], df["p90"]

    # === 1. Calibration: nominal vs empirical ===
    qs = np.linspace(0.05, 0.95, 19)
    cover = [(y.between(np.quantile(p10,q), np.quantile(p90,q)).mean()) for q in qs]
    plt.figure(figsize=(4,4))
    plt.plot(qs, cover, "o-", label="Empirical")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("Nominal quantile"); plt.ylabel("Empirical coverage")
    plt.title("Calibration curve (QRA + Conformal)")
    plt.legend(); plt.tight_layout()
    plt.savefig("figures/calibration_curve.png", dpi=300)

    # === 2. PIT histogram ===
    pit = (y - p10) / (p90 - p10 + 1e-9)
    pit = np.clip(pit, 0, 1)
    plt.figure(figsize=(4,3))
    plt.hist(pit, bins=20, range=(0,1), color="skyblue", edgecolor="k")
    plt.axhline(len(pit)/20, color="r", linestyle="--")
    plt.xlabel("PIT value"); plt.ylabel("Count")
    plt.title("PIT Histogram")
    plt.tight_layout()
    plt.savefig("figures/pit_hist.png", dpi=300)

    # === 3. Uncertainty vs irradiance ===
    if "theoretical_irradiance" in df.columns:
        plt.figure(figsize=(5,4))
        plt.scatter(df["theoretical_irradiance"], p90-p10, s=8, alpha=0.5)
        plt.xlabel("Theoretical irradiance"); plt.ylabel("Interval width")
        plt.title("Uncertainty vs Irradiance")
        plt.tight_layout()
        plt.savefig("figures/uncertainty_vs_irradiance.png", dpi=300)

    # === 4. Summary table ===
    coverage = np.mean((y>=p10)&(y<=p90))
    width = np.mean(p90-p10)
    mae = np.mean(np.abs(p50-y))
    tab = pd.DataFrame([{
        "Model":"QRA + Conformal","Coverage":coverage,"Width":width,"MAE":mae
    }])
    tab.to_csv("results/model_summary.csv", index=False)
    with open("results/model_summary.tex","w") as f:
        f.write(tab.to_latex(index=False,float_format="%.3f"))
    print("Saved figures/ and results/model_summary.{csv,tex}")

if __name__ == "__main__":
    main()
