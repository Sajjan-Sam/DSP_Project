from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _AttNet(nn.Module):
    """Maps [base_preds, context] -> attention weights over models."""
    def __init__(self, in_dim: int, n_models: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_models)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        return self.softmax(logits)


class AttentionEnsemble:
    """
    Train on VAL data:
      inputs  = concat([base_preds_val, context_val])
      target  = y_val
      output  = attention weights -> weighted sum of base preds
    Predict on TEST with the same pipeline.
    """
    def __init__(
        self,
        model_names: List[str],
        context_cols: Optional[List[str]] = None,
        lr: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.model_names = model_names
        self.context_cols = context_cols or []
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.net: Optional[_AttNet] = None
        self.trained = False
        self.history: Dict[str, List[float]] = {"train_loss": []}

    @staticmethod
    def _tensor(x, device):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _build_inputs(
        self,
        preds: Dict[str, np.ndarray],
        context_df: pd.DataFrame
    ) -> np.ndarray:
        blocks: List[np.ndarray] = []
        for name in self.model_names:
            if name not in preds:
                raise KeyError(f"Missing predictions for model '{name}'")
            blocks.append(np.asarray(preds[name], dtype=float).reshape(-1, 1))
        for c in self.context_cols:
            if c in context_df.columns:
                col = pd.to_numeric(context_df[c], errors="coerce").fillna(0.0).to_numpy().reshape(-1, 1)
            elif c.lower() == "hour" and "DATE_TIME" in context_df.columns:
                col = pd.to_datetime(context_df["DATE_TIME"], errors="coerce").dt.hour.fillna(0).to_numpy().reshape(-1,1)
            else:
                col = np.zeros((len(context_df), 1), dtype=float)
            blocks.append(col)
        X = np.hstack(blocks).astype(np.float32)
        return X

    def fit(
        self,
        preds_val: Dict[str, np.ndarray],
        y_val: np.ndarray,
        context_val: pd.DataFrame,
        verbose: bool = True,
    ):
        n_models = len(self.model_names)
        X_val = self._build_inputs(preds_val, context_val)
        y = np.asarray(y_val, dtype=float).reshape(-1, 1)

        in_dim = X_val.shape[1]
        self.net = _AttNet(in_dim, n_models).to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        X_t = self._tensor(X_val, self.device)
        y_t = self._tensor(y, self.device)

        n = len(X_val)
        idx = np.arange(n)

        for ep in range(1, self.epochs + 1):
            np.random.shuffle(idx)
            ep_loss = 0.0
            self.net.train()
            for s in range(0, n, self.batch_size):
                batch = idx[s:s+self.batch_size]
                xb = X_t[batch]
                yb = y_t[batch]

                opt.zero_grad()
                w = self.net(xb)                       # [B, n_models]
                base_preds = xb[:, :n_models]          # [B, n_models]
                yhat = (w * base_preds).sum(dim=1, keepdim=True)  # [B,1]
                loss = loss_fn(yhat, yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(batch)
            ep_loss /= n
            self.history["train_loss"].append(ep_loss)
            if verbose and (ep % 25 == 0 or ep == 1):
                print(f"AttentionEnsemble Epoch {ep:>3d}: loss={ep_loss:.6f}")

        self.trained = True
        return self

    def predict(
        self,
        preds: Dict[str, np.ndarray],
        context: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.trained and self.net is not None, "Call fit() first."
        n_models = len(self.model_names)
        X = self._build_inputs(preds, context)
        X_t = self._tensor(X, self.device)

        self.net.eval()
        with torch.no_grad():
            w = self.net(X_t)                               # [N, n_models]
            base_preds = X_t[:, :n_models]                  # [N, n_models]
            yhat = (w * base_preds).sum(dim=1).cpu().numpy()
        return yhat, w.cpu().numpy()

    def get_attention_stats(self, weights: np.ndarray) -> pd.DataFrame:
        rows = []
        for i, m in enumerate(self.model_names):
            wi = weights[:, i]
            rows.append({
                "Model": m,
                "Weight_mean": float(np.mean(wi)),
                "Weight_std": float(np.std(wi)),
                "Weight_min": float(np.min(wi)),
                "Weight_max": float(np.max(wi)),
            })
        return pd.DataFrame(rows)
