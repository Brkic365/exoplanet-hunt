"""More noise examples from the population we search: M and K dwarfs in sector 104
(same selection as the sector-105 sample, different sector), no TOI or CTOI hosts."""
import os; os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np, pandas as pd
from astroquery.mast import Catalogs
from exohunt import catalog, config, ingest, labels
from exohunt.hunt import hunt_star

def job(p):
    try:
        return hunt_star(p)
    except Exception:
        return []

def tic(chunk):
    c = Catalogs.query_criteria(catalog="Tic", ID=chunk)
    return pd.DataFrame({"tic": np.array(c["ID"]).astype(int), "Teff": np.array(c["Teff"], float),
                         "Tmag": np.array(c["Tmag"], float), "lumclass": np.array(c["lumclass"]).astype(str)})

if __name__ == "__main__":
    known = set(catalog.fetch_toi().tic.astype(int)) | set(pd.read_csv(config.CATALOG_DIR / "ctoi.csv")["TIC ID"].astype(int))
    tl = pd.read_csv(config.CATALOG_DIR / "qlp_targets_s0104.csv"); tl.columns = ["tic", "ra", "dec"]
    s105 = set(pd.read_csv(config.CATALOG_DIR / "s0105_sample.csv").tic)
    tl = tl[~tl.tic.isin(known | s105)].sample(80000, random_state=104)
    ids = [str(t) for t in tl.tic]
    with ThreadPoolExecutor(4) as ex:
        p = pd.concat(ex.map(tic, [ids[i:i + 2000] for i in range(0, len(ids), 2000)]))
    dw = p[p.lumclass == "DWARF"]
    sel = pd.concat([dw[(dw.Teff < 3900) & (dw.Tmag <= 13.5)].head(1000),
                     dw[(dw.Teff >= 3900) & (dw.Teff < 5300) & (dw.Tmag <= 12)].head(1000)])
    print("selected", len(sel), flush=True)
    got = ingest.download_many(sel.tic.tolist(), 104, workers=16)
    paths = [q for q in got.values() if q]
    recs = []
    with ProcessPoolExecutor(4) as ex:
        for i, rs in enumerate(ex.map(job, paths, chunksize=8)):
            recs.extend(rs)
            if i % 500 == 0: print(i, len(recs), flush=True)
    G = np.stack([r.pop("global_view") for r in recs]); L = np.stack([r.pop("local_view") for r in recs])
    df = pd.DataFrame(recs).assign(label=0, disposition="noise", kind="noise", toi=np.nan)
    df["split"] = df.tic.map(labels.split_of)
    old = pd.read_csv(config.DATA_DIR / "training" / "noise_features.csv")
    Vo = np.load(config.DATA_DIR / "training" / "noise_views.npz")
    np.savez_compressed(config.DATA_DIR / "training" / "noise_views.npz",
                        global_view=np.concatenate([Vo["global_view"], G]), local_view=np.concatenate([Vo["local_view"], L]))
    pd.concat([old, df], ignore_index=True).to_csv(config.DATA_DIR / "training" / "noise_features.csv", index=False)
    print("added", len(df), df.method.value_counts().to_dict(), flush=True)
