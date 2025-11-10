import os, json, argparse, yaml, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLPGate(nn.Module):
    def __init__(self, d_in, n_experts, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_experts)
        )
        self.sm = nn.Softmax(dim=-1)
    def forward(self, x):
        return self.sm(self.net(x))

def load_conf(path):
    with open(path) as f: 
        return yaml.safe_load(f)

def _to_tensor_safe(arr):
    a = np.asarray(arr, dtype=np.float32)
    a[~np.isfinite(a)] = 0.0
    return torch.tensor(a, dtype=torch.float32)

def cap_rowwise(yhat, irr, cap, eta=0.9):
    if irr is None or cap is None:
        return yhat
    irr = irr.clone()
    cap = cap.clone()
    mask_bad = ~torch.isfinite(irr) | ~torch.isfinite(cap)
    maxp = eta * irr * cap
    maxp[mask_bad] = float("inf")
    return torch.minimum(yhat, maxp)

def make_xy(df, feat_cols, expert_cols, y_col="y"):
    cols_needed = set(feat_cols + expert_cols + [y_col, "theoretical_irradiance", "plant_capacity"])
    present = [c for c in cols_needed if c in df.columns]
    df = df.copy()
    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[present] = df[present].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = _to_tensor_safe(df[feat_cols].values)
    E = _to_tensor_safe(df[expert_cols].values)
    y = _to_tensor_safe(df[y_col].values).view(-1,1)
    irr = _to_tensor_safe(df["theoretical_irradiance"].values) if "theoretical_irradiance" in df else None
    cap = _to_tensor_safe(df["plant_capacity"].values) if "plant_capacity" in df else None
    return X, E, y, irr, cap

def metrics(y, yhat):
    y = y.numpy().ravel(); yhat = yhat.numpy().ravel()
    mae = float(np.mean(np.abs(yhat-y)))
    rmse = float(np.sqrt(np.mean((yhat-y)**2)))
    smape = float(100*np.mean(2*np.abs(yhat-y)/(np.abs(y)+np.abs(yhat)+1e-9)))
    return mae, rmse, smape

def train_gate(cfg):
    print("[INFO] loading features and data …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["random_seed"])

    meta = json.load(open("data/gate/features.json"))
    feat_cols = meta["features"]
    experts   = meta["experts"]
    expert_cols = [f"pred_{e}" for e in experts]
    n_experts = len(experts)

    val_df  = pd.read_parquet("data/gate/val.parquet")
    test_df = pd.read_parquet("data/gate/test.parquet")

    Xv, Ev, yv, irr_v, cap_v = make_xy(val_df, feat_cols, expert_cols)
    Xt, Et, yt, irr_t, cap_t = make_xy(test_df, feat_cols, expert_cols)
    print(f"[INFO] VAL size={len(yv)} TEST size={len(yt)} | experts={experts}")

    dl = DataLoader(TensorDataset(Xv, Ev, yv), batch_size=cfg["training"]["batch_size"], shuffle=True)
    gate = MLPGate(d_in=len(feat_cols), n_experts=n_experts,
                   hidden=cfg["model"]["hidden"], dropout=cfg["model"]["dropout"]).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    mse = nn.MSELoss()
    best_loss, best_state, best_epoch = 1e18, None, 0
    patience = cfg["training"]["early_stop_patience"]

    print("[INFO] training …")
    for epoch in range(cfg["training"]["epochs"]):
        gate.train(); run_loss = 0.0
        for xb, eb, yb in dl:
            xb, eb, yb = xb.to(device), eb.to(device), yb.to(device)
            w = gate(xb)
            y_hat = (w * eb).sum(-1, keepdim=True)
            loss = mse(y_hat, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * len(xb)
        run_loss /= len(dl.dataset)

        gate.eval()
        with torch.no_grad():
            wv = gate(Xv.to(device))
            yv_hat = (wv * Ev.to(device)).sum(-1, keepdim=True).cpu().squeeze(1)
            yv_hat = cap_rowwise(yv_hat, irr_v, cap_v)
            val_loss = mse(yv_hat.view(-1,1), yv).item()

        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, gate.state_dict(), epoch
        elif epoch - best_epoch >= patience:
            print(f"[INFO] early stop @ epoch {epoch} (best epoch {best_epoch})")
            break

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"[epoch {epoch+1}] train_mse={run_loss:.6f} val_mse={val_loss:.6f}")

    gate.load_state_dict(best_state)

    print("[INFO] inference …")
    gate.eval()
    with torch.no_grad():
        wv = gate(Xv.to(device)); yv_hat = (wv * Ev.to(device)).sum(-1, keepdim=True).cpu().squeeze(1)
        wt = gate(Xt.to(device)); yt_hat = (wt * Et.to(device)).sum(-1, keepdim=True).cpu().squeeze(1)
        yv_hat = cap_rowwise(yv_hat, irr_v, cap_v)
        yt_hat = cap_rowwise(yt_hat, irr_t, cap_t)

    m_val = metrics(yv, yv_hat)
    m_tst = metrics(yt, yt_hat)

    os.makedirs(os.path.dirname(cfg["save"]["model_path"]), exist_ok=True)
    torch.save({"state_dict": gate.state_dict(),
                "features": feat_cols, "experts": experts}, cfg["save"]["model_path"])

    np.savez(cfg["save"]["pred_out_npz"], val=yv_hat.numpy(), test=yt_hat.numpy())
    os.makedirs(os.path.dirname(cfg["save"]["metrics_csv"]), exist_ok=True)
    out_row = pd.DataFrame([{
        "Model":"MoE-Gate (regime-aware)","Split":"VAL","MAE":m_val[0],"RMSE":m_val[1],"SMAPE":m_val[2]
    },{
        "Model":"MoE-Gate (regime-aware)","Split":"TEST","MAE":m_tst[0],"RMSE":m_tst[1],"SMAPE":m_tst[2]
    }])
    if os.path.exists(cfg["save"]["metrics_csv"]):
        out_row.to_csv(cfg["save"]["metrics_csv"], mode="a", index=False, header=False)
    else:
        out_row.to_csv(cfg["save"]["metrics_csv"], index=False)

    print("VAL -> MAE {:.6f} | RMSE {:.6f} | sMAPE {:.3f}%".format(*m_val))
    print("TEST-> MAE {:.6f} | RMSE {:.6f} | sMAPE {:.3f}%".format(*m_tst))
    print(f"[OK] saved: {cfg['save']['model_path']}, {cfg['save']['pred_out_npz']}, {cfg['save']['metrics_csv']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/moe.yaml")
    args = p.parse_args()
    cfg = load_conf(args.config)
    train_gate(cfg)
