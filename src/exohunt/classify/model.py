"""AstroNet-style two-tower CNN (Shallue & Vanderburg 2018), sized for a few
thousand labelled TESS signals instead of Kepler's 15,000."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _block(cin, cout, k=5, pool=True):
    layers = [nn.Conv1d(cin, cout, k, padding=k // 2), nn.ReLU(), nn.Conv1d(cout, cout, k, padding=k // 2), nn.ReLU()]
    if pool:
        layers.append(nn.MaxPool1d(5, stride=2, padding=2))
    return layers


class TransitCNN(nn.Module):
    def __init__(self, n_local_ch: int = 5, n_scalar: int = 0, width: int = 16, dropout: float = 0.3):
        super().__init__()
        w = width
        self.glob = nn.Sequential(*_block(1, w), *_block(w, 2 * w), *_block(2 * w, 4 * w), *_block(4 * w, 4 * w),
                                  nn.AdaptiveMaxPool1d(4), nn.Flatten())
        self.loc = nn.Sequential(*_block(n_local_ch, w), *_block(w, 2 * w), nn.AdaptiveMaxPool1d(4), nn.Flatten())
        n_in = 4 * w * 4 + 2 * w * 4 + n_scalar
        self.head = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))
        self.n_scalar = n_scalar

    def forward(self, g, l, s=None):
        z = [self.glob(g), self.loc(l)]
        if self.n_scalar:
            z.append(s)
        return self.head(torch.cat(z, 1)).squeeze(1)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def train_cnn(G, L, S, y, idx_tr, idx_va, n_scalar=0, epochs=60, lr=1e-3, batch=64, seed=0, device="cpu",
              weight_decay=1e-4, patience=12, verbose=False):
    """Train one network; early-stop on validation AUC. Returns (model, scaler(mean, std))."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu = S[idx_tr].mean(0) if n_scalar else None
    sd = S[idx_tr].std(0) + 1e-6 if n_scalar else None
    Sn = (S - mu) / sd if n_scalar else np.zeros((len(y), 0), np.float32)
    tG = torch.tensor(G[:, None, :], dtype=torch.float32)
    tL = torch.tensor(L, dtype=torch.float32)
    tS = torch.tensor(Sn, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    model = TransitCNN(L.shape[1], n_scalar).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(idx_tr)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            g, l = tG[b], tL[b]
            flip = torch.rand(len(b)) < 0.5  # transits are time-symmetric
            g = torch.where(flip[:, None, None], g.flip(-1), g)
            l = torch.where(flip[:, None, None], l.flip(-1), l)
            g = g + 0.02 * torch.randn_like(g)
            l = l + 0.02 * torch.randn_like(l)
            opt.zero_grad()
            loss = lossf(model(g.to(device), l.to(device), tS[b].to(device)), ty[b].to(device))
            loss.backward()
            opt.step()
        p = predict(model, G[idx_va], L[idx_va], Sn[idx_va], device)
        auc = _auc(y[idx_va], p)
        if verbose:
            print(ep, round(float(loss), 3), round(auc, 4))
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, (mu, sd), best


@torch.no_grad()
def predict(model, G, L, Sn, device="cpu"):
    model.eval()
    out = []
    for i in range(0, len(G), 512):
        g = torch.tensor(G[i:i + 512, None, :], dtype=torch.float32, device=device)
        l = torch.tensor(L[i:i + 512], dtype=torch.float32, device=device)
        s = torch.tensor(Sn[i:i + 512], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(g, l, s)).cpu().numpy())
    return np.concatenate(out)


class Ensemble:
    """Average of several seeds; carries its own scalar scaling."""

    def __init__(self, models, scalers, n_scalar, feature_cols):
        self.models, self.scalers, self.n_scalar, self.feature_cols = models, scalers, n_scalar, feature_cols

    def predict(self, G, L, S=None):
        ps = []
        for m, (mu, sd) in zip(self.models, self.scalers):
            Sn = (S - mu) / sd if self.n_scalar else np.zeros((len(G), 0), np.float32)
            ps.append(predict(m, G, L, Sn))
        return np.mean(ps, 0)

    def save(self, path):
        torch.save({"states": [m.state_dict() for m in self.models], "scalers": self.scalers,
                    "n_scalar": self.n_scalar, "feature_cols": self.feature_cols,
                    "n_local_ch": self.models[0].loc[0].in_channels}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        models = []
        for st in d["states"]:
            m = TransitCNN(d["n_local_ch"], d["n_scalar"])
            m.load_state_dict(st)
            m.eval()
            models.append(m)
        return cls(models, d["scalers"], d["n_scalar"], d["feature_cols"])
