"""Noise / systematics examples for the classifier: run the full hunt on stars
that host no TOI and keep every signal it reports (label 0, kind "noise").
Sources: the 500 week-2 sector-75 field stars, plus 1,000 random sector-103
QLP targets (a recent sector, not the one we search)."""
import os; os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from exohunt import catalog, config, ingest, labels
from exohunt.hunt import hunt_star
from exohunt.injection import field_dir

def job(p):
    try:
        return hunt_star(p)
    except Exception:
        return []

if __name__ == "__main__":
    toi_tics = set(catalog.fetch_toi().tic.astype(int)) | set(pd.read_csv(config.CATALOG_DIR / "ctoi.csv")["TIC ID"].astype(int))
    paths = sorted(field_dir(75).glob("*.fits"))
    tl = pd.read_csv(config.CATALOG_DIR / "qlp_targets_s0103.csv"); tl.columns = ["tic", "ra", "dec"]
    tl = tl[~tl.tic.isin(toi_tics)].sample(1300, random_state=103)
    got = ingest.download_many(tl.tic.tolist(), 103, workers=16)
    paths += [p for p in got.values() if p][:1000]
    print("stars", len(paths), flush=True)
    recs = []
    with ProcessPoolExecutor(4) as ex:
        for i, rs in enumerate(ex.map(job, paths, chunksize=8)):
            recs.extend(rs)
            if i % 300 == 0: print(i, len(recs), flush=True)
    G = np.stack([r.pop("global_view") for r in recs]); L = np.stack([r.pop("local_view") for r in recs])
    df = pd.DataFrame(recs)
    df["label"], df["disposition"], df["kind"] = 0, "noise", "noise"
    df["split"] = df.tic.map(labels.split_of)
    df["toi"] = np.nan
    np.savez_compressed(config.DATA_DIR / "training" / "noise_views.npz", global_view=G, local_view=L)
    df.to_csv(config.DATA_DIR / "training" / "noise_features.csv", index=False)
    print(len(df), df.method.value_counts().to_dict(), df.split.value_counts().to_dict())
