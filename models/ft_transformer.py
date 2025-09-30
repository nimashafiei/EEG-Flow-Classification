# models/ft_transformer.py
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class FeatureTokenizer(nn.Module):
    """
    Linear token per numeric feature + learned bias per feature.
    Turns an [B, F] numeric matrix into [B, F, d_token] tokens.
    """
    def __init__(self, n_num_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_num_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_num_features, d_token))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        # token_i = x[:, i] * W[i] + b[i]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)  # [B, F, d_token]


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with pre-LN, MHA, and FFN.
    """
    def __init__(self, d_token: int, n_heads: int, attn_dropout: float, ff_dropout: float, d_ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        d_ff = d_token * d_ff_mult
        self.ff = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_ff, d_token),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class FTTransformerTorch2(BaseEstimator, ClassifierMixin):
    """
    Torch 2.x compatible FT-Transformer for binary classification on numeric/tabular features.

    - Works as an sklearn estimator (fit / predict / predict_proba).
    - Adds a learnable [CLS] token; the head uses the [CLS] representation for classification.
    - Optionally standardizes inputs with sklearn's StandardScaler.
    """

    def __init__(
        self,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 4,
        attn_dropout: float = 0.10,
        ff_dropout: float = 0.10,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        batch_size: int = 512,
        n_epochs: int = 50,
        patience: int = 8,
        standardize: bool = True,
        device: str = "auto",     # "auto" | "cpu" | "cuda"
        seed: int = 42,
        num_workers: int = 0,
        verbose: bool = False
    ):
        self.d_token=d_token; self.n_blocks=n_blocks; self.n_heads=n_heads
        self.attn_dropout=attn_dropout; self.ff_dropout=ff_dropout
        self.lr=lr; self.weight_decay=weight_decay
        self.batch_size=batch_size; self.n_epochs=n_epochs; self.patience=patience
        self.standardize=standardize; self.device=device; self.seed=seed
        self.num_workers=num_workers; self.verbose=verbose

        # will be set during fit
        self._scaler = None
        self._model = None
        self._device = None
        self._n_features = None

    # --------- internal model builder ---------
    def _build(self, n_features: int):
        # a learnable CLS token (shared across batch)
        cls = nn.Parameter(torch.zeros(1, 1, self.d_token))
        tok = FeatureTokenizer(n_features, self.d_token)
        blocks = nn.ModuleList([
            TransformerBlock(self.d_token, self.n_heads, self.attn_dropout, self.ff_dropout)
            for _ in range(self.n_blocks)
        ])
        head = nn.Sequential(nn.LayerNorm(self.d_token), nn.Linear(self.d_token, 1))
        return cls, tok, blocks, head

    # --------- sklearn API: fit ---------
    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)

        # to numpy
        X = X.values if hasattr(X, "values") else X
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # optional standardization
        if self.standardize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X).astype(np.float32)

        self._n_features = X.shape[1]
        # choose device
        if self.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device)

        # build the model
        cls, tok, blocks, head = self._build(self._n_features)

        class Model(nn.Module):
            def __init__(self, cls, tok, blocks, head):
                super().__init__()
                self.cls = cls
                self.tok = tok
                self.blocks = blocks
                self.head = head
            def forward(self, x):
                # x: [B, F]
                B = x.size(0)
                tokens = self.tok(x)            # [B, F, d]
                cls = self.cls.expand(B, -1, -1) # [B, 1, d]
                h = torch.cat([cls, tokens], dim=1)  # [B, 1+F, d]
                for blk in self.blocks:
                    h = blk(h)
                cls_out = h[:, 0]               # [B, d]
                return self.head(cls_out).squeeze(1)  # [B]

        self._model = Model(cls, tok, blocks, head).to(self._device)

        # build loaders (10% validation for early stopping)
        ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
        n = len(ds)
        val_n = max(1, int(0.1*n)) if self.patience else 0

        if val_n > 0:
            idx = np.random.permutation(n)
            va_idx, tr_idx = idx[:val_n], idx[val_n:]
            tr_ds = TensorDataset(ds.tensors[0][tr_idx], ds.tensors[1][tr_idx])
            va_ds = TensorDataset(ds.tensors[0][va_idx], ds.tensors[1][va_idx])
            tr_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            va_dl = DataLoader(va_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            tr_dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            va_dl = None

        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        def run_epoch(loader, train=True):
            self._model.train(train)
            total = 0.0; nobs = 0
            for xb, yb in loader:
                xb = xb.to(self._device, dtype=torch.float32)
                yb = yb.to(self._device, dtype=torch.float32)
                if train:
                    opt.zero_grad()
                logits = self._model(xb)
                loss = loss_fn(logits, yb)
                if train:
                    loss.backward()
                    opt.step()
                total += loss.item() * xb.size(0); nobs += xb.size(0)
            return total / nobs

        best = math.inf; best_state = None
        patience = self.patience if val_n > 0 else 0

        for epoch in range(self.n_epochs):
            tr_loss = run_epoch(tr_dl, train=True)
            if va_dl:
                val_loss = run_epoch(va_dl, train=False)
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.n_epochs} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")
                if val_loss < best - 1e-6:
                    best = val_loss
                    patience = self.patience
                    best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    patience -= 1
                    if patience <= 0:
                        break
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.n_epochs} | train_loss={tr_loss:.4f}")

        # restore best
        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    # --------- internal: probability helper ---------
    def _proba(self, X):
        X = X.values if hasattr(X, "values") else X
        X = np.asarray(X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X).astype(np.float32)

        xb = torch.tensor(X, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(xb)
            p1 = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1.0 - p1, p1]).T  # [B, 2]

    # --------- sklearn API: predict_proba / predict ---------
    def predict_proba(self, X):
        return self._proba(X)

    def predict(self, X):
        return (self._proba(X)[:, 1] >= 0.5).astype(int)
