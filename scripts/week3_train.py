"""Train and compare classifiers on ExoFOP dispositions (CP/KP vs FP/FA), split by host star.

Models: logistic regression and gradient boosting on plain BLS statistics, the
rule-based vetter, gradient boosting on BLS + vetting metrics, and the CNN on
folded views with and without the scalar features. 5-seed ensembles for CNNs.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from exohunt import config
from exohunt.classify import BLS_FEATURES, VET_FEATURES, scalar_matrix
from exohunt.classify.model import Ensemble, train_cnn

torch.set_num_threads(4)
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
out = Path("results/classifier"); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(config.DATA_DIR / "training" / "tois_features.csv")
V = np.load(config.DATA_DIR / "training" / "tois_views.npz")
G, L = V["global_view"], V["local_view"]
y = df["label"].to_numpy(int)
tr, va, te = (np.where(df["split"] == s)[0] for s in ("train", "val", "test"))
trva = np.concatenate([tr, va])
print({k: (len(i), int(y[i].sum())) for k, i in (("train", tr), ("val", va), ("test", te))})
Sb, Sv = scalar_matrix(df, BLS_FEATURES), scalar_matrix(df, VET_FEATURES)

preds = {}
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Sb[trva], y[trva])
preds["logreg_bls"] = lr.predict_proba(Sb)[:, 1]
for name, S in (("gbm_bls", Sb), ("gbm_vet", Sv)):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                       l2_regularization=1.0, random_state=0).fit(S[trva], y[trva])
    preds[name] = m.predict_proba(S)[:, 1]
preds["vet_rules"] = df["vet_ok"].astype(float).to_numpy()

ensembles = {}
for name, S, ns in (("cnn_views", None, 0), ("cnn_views_vet", Sv, Sv.shape[1])):
    models, scalers, vals = [], [], []
    for seed in range(N_SEEDS):
        m, sc, best = train_cnn(G, L, S if S is not None else np.zeros((len(y), 0), np.float32), y, tr, va,
                                n_scalar=ns, seed=seed)
        models.append(m); scalers.append(sc); vals.append(best)
        print(name, seed, round(best, 4), flush=True)
    ens = Ensemble(models, scalers, ns, VET_FEATURES if ns else [])
    preds[name] = ens.predict(G, L, S)
    ensembles[name] = ens
ensembles["cnn_views_vet"].save(out / "cnn_views_vet.pt")
ensembles["cnn_views"].save(out / "cnn_views.pt")

preds["ensemble"] = 0.5 * (preds["gbm_vet"] + preds["cnn_views_vet"])  # what the hunt uses


def metrics(p, idx):
    yt, pt = y[idx], p[idx]
    pr, rc, th = precision_recall_curve(yt, pt)
    at = lambda P: float(rc[np.where(pr[:-1] >= P)[0]].max()) if (pr[:-1] >= P).any() else 0.0
    hard = pt >= 0.5
    return dict(auc=roc_auc_score(yt, pt) if len(np.unique(pt)) > 2 else np.nan, avg_precision=average_precision_score(yt, pt),
                precision_at_0p5=float(yt[hard].mean()) if hard.any() else np.nan, recall_at_0p5=float(hard[yt == 1].mean()),
                recall_at_p90=at(0.90), recall_at_p95=at(0.95), n=len(idx), n_planet=int(yt.sum()))

rows = {k: metrics(p, te) for k, p in preds.items()}
mt = pd.DataFrame(rows).T
print(mt.round(3).to_string())
mt.to_csv(out / "metrics_test.csv")
pd.DataFrame({"toi": df.toi, "tic": df.tic, "label": y, "split": df.split, "disposition": df.disposition,
              **{k: np.round(v, 4) for k, v in preds.items()}}).to_csv(out / "predictions.csv", index=False)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 5))
names = {"logreg_bls": "Logistic reg., BLS stats", "gbm_bls": "Gradient boosting, BLS stats",
         "gbm_vet": "Gradient boosting, BLS + vetting", "cnn_views": "CNN, folded views",
         "cnn_views_vet": "CNN, views + vetting", "ensemble": "Mean of CNN + GB (used)"}
for k, lab in names.items():
    pr, rc, _ = precision_recall_curve(y[te], preds[k][te])
    ax.plot(rc, pr, label=f"{lab} (AP {rows[k]['avg_precision']:.2f})")
r = rows["vet_rules"]
ax.plot(r["recall_at_0p5"], r["precision_at_0p5"], "ko", label="Vetting rules (pass/fail)")
ax.axhline(y[te].mean(), color="0.7", ls=":", lw=1)
ax.set_xlabel("recall (confirmed/known planets)"); ax.set_ylabel("precision"); ax.set_ylim(0.4, 1.01)
ax.set_title(f"Held-out stars: {len(te)} TOIs ({int(y[te].sum())} planets)")
ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig(out / "pr_curves.png", dpi=150)
