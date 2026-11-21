"""Score sector signals: systematics filter, CNN + GBM ensemble, cross-match with TOI / CTOI / SPOC TCE lists."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from exohunt import catalog, config
from exohunt.classify import VET_FEATURES, scalar_matrix
from exohunt.classify.model import Ensemble

sector = int(sys.argv[1]); out = Path(f"results/s{sector:04d}"); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(config.DATA_DIR / f"s{sector:04d}_signals_raw.csv")
V = np.load(config.DATA_DIR / f"s{sector:04d}_views.npz")
G, L = V["global_view"], V["local_view"]

# --- classifier: mean of CNN (views + vetting) and gradient boosting (vetting) -----------------
tr = pd.read_csv(config.DATA_DIR / "training" / "tois_features.csv")
noise_csv = config.DATA_DIR / "training" / "noise_features.csv"
if noise_csv.exists():  # same training data as the CNN (scripts/week3_train.py)
    tr = pd.concat([tr, pd.read_csv(noise_csv)], ignore_index=True)
gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                     random_state=0).fit(scalar_matrix(tr[tr.split != "test"], VET_FEATURES),
                                                         tr.label[tr.split != "test"])
S = scalar_matrix(df, VET_FEATURES)
cnn = Ensemble.load("results/classifier/cnn_views_vet.pt")
df["p_cnn"] = cnn.predict(G, L, S)
df["p_gbm"] = gbm.predict_proba(S)[:, 1]
df["score"] = 0.5 * (df.p_cnn + df.p_gbm)

# --- sector-wide systematics: many stars dipping at the same time / same period ---------------
single = df.method == "single"
tb = np.floor(df.t0 / 0.25)
cnt = tb[single].value_counts()
lam = np.median(cnt.reindex(np.arange(tb[single].min(), tb[single].max() + 1), fill_value=0))
bad_t = set(cnt[cnt > lam + 5 * np.sqrt(max(lam, 1)) + 3].index)
df["common_time"] = single & tb.isin(bad_t)
pb = np.round(np.log(df.period.where(~single)) / 0.005)
pc = pb.value_counts()
bad_p = set(pc[pc >= 8].index)
df["common_period"] = (~single) & pb.isin(bad_p)
df["systematic"] = df.common_time | df.common_period

# --- cross-matches ------------------------------------------------------------------------------
toi = catalog.fetch_toi()
ctoi = pd.read_csv(config.CATALOG_DIR / "ctoi.csv")
tces = [pd.read_csv(p, comment="#") for p in sorted(config.CATALOG_DIR.glob("*dvr-tcestats.csv")) + [config.CATALOG_DIR / "tce_s0001-s0096.csv"] if p.exists()]
tce = pd.concat(tces)[["ticid", "tce_period"]].drop_duplicates()

def pmatch(P, others, tol=0.01):
    if not np.isfinite(P):
        return False
    for q in others:
        if np.isfinite(q) and q > 0:
            for h in (1, 0.5, 2, 1 / 3, 3):
                if abs(P / q - h) / h < tol:
                    return True
    return False

tp = toi.groupby("tic")["period"].apply(list).to_dict()
cp = ctoi.groupby("TIC ID")["Period (days)"].apply(list).to_dict()
ep = tce.groupby("ticid")["tce_period"].apply(list).to_dict()
df["toi_host"] = df.tic.isin(tp)
df["toi_match"] = [pmatch(P, tp.get(t, [])) for t, P in zip(df.tic, df.period)]
df["ctoi_host"] = df.tic.isin(cp)
df["ctoi_match"] = [pmatch(P, cp.get(t, [])) for t, P in zip(df.tic, df.period)]
df["tce_host"] = df.tic.isin(ep)
df["tce_match"] = [pmatch(P, ep.get(t, [])) for t, P in zip(df.tic, df.period)]
df["known"] = df.toi_match | df.ctoi_match | df.tce_match
# signals vetted before the local-noise check existed: add it for the ones still in the running
from exohunt import ingest
from exohunt.vetting import Ephemeris, masked_detrend, local_noise_ratio
pre = df.vet_ok.astype(bool) & ~df.systematic
if "local_noise_ratio" not in df:
    df["local_noise_ratio"] = np.nan
from exohunt.vetting import edge_fraction
if "edge_frac" not in df:
    df["edge_frac"] = np.nan
for i in df.index[pre & (df.local_noise_ratio.isna() | df.edge_frac.isna())]:
    r = df.loc[i]
    eph = Ephemeris(r.period if np.isfinite(r.period) else 1e4, r.t0, r.duration)
    raw = ingest.read_qlp(ingest.qlp_path(r.tic, sector))
    df.loc[i, "local_noise_ratio"] = local_noise_ratio(masked_detrend(raw, eph), eph)
    df.loc[i, "edge_frac"] = edge_fraction(raw.time, eph)
nev = df.n_events.fillna(1)
edge = (df.edge_frac > 0.5) | ((nev <= 1) & (df.edge_frac > 0))
df.loc[edge, "flags"] = df.loc[edge, "flags"].fillna("").map(lambda s: ",".join([x for x in s.split(",") if x] + ["edge"]))
df.loc[edge, "vet_ok"] = False
noisy = df.local_noise_ratio > 1.5
df.loc[noisy, "flags"] = df.loc[noisy, "flags"].fillna("").map(lambda s: ",".join([x for x in s.split(",") if x] + ["noisy"]))
df.loc[noisy, "vet_ok"] = False
# second pass: lone dips that pass vetting but still bunch up in time across stars
ok1 = df.vet_ok.astype(bool) & ~df.systematic & single
tt = df.t0.where(ok1)
crowd = np.array([ok1[i] and ((np.abs(tt - tt[i]) < 0.5) & (df.tic != df.tic[i])).sum() >= 3 for i in df.index])
df.loc[crowd, "common_time"] = True
df["systematic"] = df.common_time | df.common_period
df["rank_ok"] = df.vet_ok.astype(bool) & ~df.systematic
cols = [c for c in df.columns if not c.startswith("x_") or c in ("x_ses_ratio_up", "x_p_circ_est")]
df = df.sort_values("score", ascending=False)
df[cols].to_csv(out / "signals.csv", index=False)

sample = pd.read_csv(config.CATALOG_DIR / f"s{sector:04d}_sample.csv")
summ = {
    "stars_searched": int(df.attrs.get("n", len(sample))),
    "signals": len(df), "periodic": int((~single).sum()), "single": int(single.sum()),
    "systematic": int(df.systematic.sum()), "fail_vetting": int((~df.vet_ok.astype(bool)).sum()),
    "pass_all": int(df.rank_ok.sum()), "pass_all_score_ge_0.5": int((df.rank_ok & (df.score >= 0.5)).sum()),
    "pass_new_score_ge_0.5": int((df.rank_ok & (df.score >= 0.5) & ~df.known & ~df.toi_host).sum()),
}
print(summ)
pd.Series(summ).to_csv(out / "funnel.csv")
# known TOIs in the sample: recovered?
t = toi[toi.tic.isin(sample.tic) & toi.disposition.isin(["CP", "KP", "PC", "APC"])]
rec = []
for r in t.itertuples():
    d = df[(df.tic == r.tic)]
    rec.append(dict(toi=r.toi, tic=r.tic, disposition=r.disposition, period=r.period, depth_ppm=r.depth_ppm,
                    recovered=bool(d.toi_match.any()), single_event_on_host=bool((d.method == "single").any()),
                    best_score=float(d[d.toi_match].score.max()) if d.toi_match.any() else np.nan))
rec = pd.DataFrame(rec); rec.to_csv(out / "toi_recovery.csv", index=False)
print("TOIs on sample stars:", len(rec), "recovered by period:", int(rec.recovered.sum()),
      "median score of recovered:", rec.best_score.median())
