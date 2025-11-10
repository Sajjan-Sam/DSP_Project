import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseForecastModel

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 96):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.X) - self.sequence_length)

    def __getitem__(self, idx):
        X_seq = self.X[idx: idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_target])


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, lstm_output):
        scores = self.attention(lstm_output)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_output, 1)
        return context, weights.squeeze(-1)


class LSTMAttentionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = AttentionLayer(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ctx, w = self.attn(lstm_out)
        y = self.fc(ctx)
        return y, w


class LSTMForecastModel(BaseForecastModel):
    def __init__(self, config: Dict):
        super().__init__(config, "LSTM")
        self.epochs: int = config.get("epochs", 120)
        self.batch_size: int = config.get("batch_size", 128)
        self.learning_rate: float = config.get("learning_rate", 1.5e-4)
        self.hidden_dim: int = config.get("hidden_dim", 128)
        self.num_layers: int = config.get("num_layers", 2)
        self.dropout: float = config.get("dropout", 0.15)
        self.sequence_length: int = config.get("sequence_length", 48)
        self.early_stopping_patience: int = config.get("early_stopping_patience", 10)
        self.use_residual: bool = config.get("use_residual", False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.network: Optional[nn.Module] = None
        self.best_state = None
        self.is_trained: bool = False
        self._input_dim: Optional[int] = None
        self._log_target: bool = True

    def _to_model_space(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(y) if self._log_target else y

    def _from_model_space(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y) if self._log_target else y

    def _build_baseline(self, X_df: pd.DataFrame, y_arr: Optional[np.ndarray] = None) -> np.ndarray:
        for cand in ["TOTAL_AC_POWER_ewm_4", "TOTAL_AC_POWER_ewm_8", "TOTAL_AC_POWER_vs_daily_mean"]:
            if cand in X_df.columns:
                b = X_df[cand].values.astype(np.float32)
                return np.maximum(b, 0.0)
        if y_arr is not None and len(y_arr) > 0:
            b = np.roll(y_arr, 1)
            b[0] = b[1]
            return np.maximum(b.astype(np.float32), 0.0)
        return np.full(len(X_df), 1.0, dtype=np.float32)

    def _select_features(self, X: pd.DataFrame, max_features: int = 24) -> List[str]:
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric = [c for c in numeric if not any(x in c.upper() for x in ["ID", "SOURCE", "PLANT"])]
        patterns = ["IRRADIATION", "TEMP", "SOLAR", "HOUR", "DAY", "LAG", "ROLL", "CLEARNESS", "RATIO", "EWM", "PCA"]
        prioritized: List[str] = []
        for p in patterns:
            prioritized += [c for c in numeric if p in c.upper()]
        remaining = [c for c in numeric if c not in prioritized]
        return (prioritized + remaining)[:max_features]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        self.feature_names = self._select_features(X_train, max_features=24)
        Xtr = X_train[self.feature_names].ffill().fillna(0.0).values
        Xv = X_val[self.feature_names].ffill().fillna(0.0).values
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xv_s = self.scaler.transform(Xv)
        input_dim = Xtr_s.shape[1]
        if (self.network is None) or (self._input_dim != input_dim):
            self._input_dim = input_dim
            self.network = LSTMAttentionNetwork(input_dim=input_dim, hidden_dim=self.hidden_dim,
                                                num_layers=self.num_layers, dropout=self.dropout).to(self.device)
        ytr_raw = y_train.values.astype(np.float32)
        yv_raw = y_val.values.astype(np.float32)
        if self.use_residual:
            base_tr = self._build_baseline(X_train, ytr_raw)
            base_v = self._build_baseline(X_val, yv_raw)
            ytr = (self._to_model_space(ytr_raw) - self._to_model_space(base_tr)).astype(np.float32)
            yv = (self._to_model_space(yv_raw) - self._to_model_space(base_v)).astype(np.float32)
        else:
            ytr = self._to_model_space(ytr_raw).astype(np.float32)
            yv = self._to_model_space(yv_raw).astype(np.float32)
        train_ds = TimeSeriesDataset(Xtr_s, ytr, self.sequence_length)
        val_ds = TimeSeriesDataset(Xv_s, yv, self.sequence_length)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
        best_val = float("inf")
        patience_ct = 0
        self.network.train()
        for epoch in range(1, self.epochs + 1):
            train_loss_sum = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                yhat, _ = self.network(xb)
                loss = criterion(yhat.squeeze(-1), yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            val_loss_sum = 0.0
            self.network.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    yhat, _ = self.network(xb)
                    vloss = criterion(yhat.squeeze(-1), yb)
                    val_loss_sum += vloss.item()
            self.network.train()
            train_loss = train_loss_sum / max(1, len(train_loader))
            val_loss = val_loss_sum / max(1, len(val_loader))
            scheduler.step(val_loss)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_ct = 0
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                patience_ct += 1
                if patience_ct >= self.early_stopping_patience:
                    break
        if self.best_state is not None:
            self.network.load_state_dict(self.best_state)
        self.is_trained = True
        return self

    def predict(self, X: pd.DataFrame, return_std: bool = False):
        if not self.is_trained or self.network is None:
            raise ValueError("Model must be trained before prediction.")
        X_full = X.ffill().fillna(0.0)
        X_raw = X_full[self.feature_names].values
        Xs = self.scaler.transform(X_raw)
        T = len(Xs)
        L = self.sequence_length
        if self.use_residual:
            baseline_full = self._build_baseline(X_full)
            baseline_in_model_space = self._to_model_space(baseline_full.astype(np.float32))
        else:
            baseline_in_model_space = None
        if return_std:
            self.network.train()
            S = 30
            samples = []
            with torch.no_grad():
                for _ in range(S):
                    preds_s = []
                    for i in range(L, T):
                        seq = Xs[i - L: i]
                        xt = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                        res, _ = self.network(xt)
                        res_v = float(res.cpu().numpy()[0, 0])
                        if self.use_residual:
                            pred = self._from_model_space(res_v + float(baseline_in_model_space[i]))
                        else:
                            pred = self._from_model_space(res_v)
                        preds_s.append(max(pred, 0.0))
                    samples.append([float("nan")] * L + preds_s)
            samples = np.array(samples, dtype=float)
            yhat = np.nanmedian(samples, axis=0)
            std = np.nanstd(samples, axis=0)
            self.network.eval()
            return yhat, std
        self.network.eval()
        preds = []
        with torch.no_grad():
            for i in range(L, T):
                seq = Xs[i - L: i]
                xt = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                res, _ = self.network(xt)
                res_v = float(res.cpu().numpy()[0, 0])
                if self.use_residual and baseline_in_model_space is not None:
                    pred = self._from_model_space(res_v + float(baseline_in_model_space[i]))
                else:
                    pred = self._from_model_space(res_v)
                preds.append(max(pred, 0.0))
        yhat = [float("nan")] * L + preds
        return np.array(yhat, dtype=float)

    def save(self, filepath):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        from pathlib import Path
        import pickle
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "network_state": self.network.state_dict() if self.network else None,
            "config": self.config, "model_name": self.model_name, "feature_names": self.feature_names,
            "scaler": self.scaler, "sequence_length": self.sequence_length, "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers, "dropout": self.dropout, "input_dim": self._input_dim,
            "training_history": self.training_history, "is_trained": self.is_trained,
            "use_residual": self.use_residual, "log_target": self._log_target
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Model saved: {filepath}")

    def load(self, filepath):
        from pathlib import Path
        import pickle
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)
        self.config = save_dict["config"]
        self.model_name = save_dict["model_name"]
        self.feature_names = save_dict["feature_names"]
        self.scaler = save_dict["scaler"]
        self.sequence_length = save_dict["sequence_length"]
        self.hidden_dim = save_dict["hidden_dim"]
        self.num_layers = save_dict["num_layers"]
        self.dropout = save_dict["dropout"]
        self._input_dim = save_dict.get("input_dim")
        self.training_history = save_dict.get("training_history", {})
        self.is_trained = save_dict["is_trained"]
        self.use_residual = save_dict.get("use_residual", False)
        self._log_target = save_dict.get("log_target", True)
        if save_dict.get("network_state") and self._input_dim:
            self.network = LSTMAttentionNetwork(input_dim=self._input_dim, hidden_dim=self.hidden_dim,
                                                num_layers=self.num_layers, dropout=self.dropout).to(self.device)
            self.network.load_state_dict(save_dict["network_state"])
            self.network.eval()
        print(f"Model loaded: {filepath}")

    def __repr__(self) -> str:
        return f"LSTMAttentionForecastModel ({'trained' if self.is_trained else 'untrained'}, seq_len={self.sequence_length})"
