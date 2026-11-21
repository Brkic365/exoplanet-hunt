"""Views + vetting features for every labelled TOI (ExoFOP CP/KP vs FP/FA) -> data/training/tois.npz/.parquet."""
import sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from exohunt import catalog, config, ingest, labels
from exohunt.classify import signal_record
from exohunt.vetting import Ephemeris
from exohunt.views import refine_ephemeris

warnings.filterwarnings("ignore")
toi = catalog.fetch_toi()
files = pd.read_csv(config.CATALOG_DIR / "training_files.csv")
lt = labels.labelled_tois(toi)
lt = lt[lt.tic.isin(files.tic)]
all_toi = toi[np.isfinite(toi.period) & np.isfinite(toi.epoch_btjd)]
by_star = files.groupby("tic")["path"].apply(list).to_dict()

def one(tic):
    warnings.filterwarnings("ignore")
    try:
        lcs = [ingest.read_qlp(p) for p in by_star[tic]]
        lcs = [lc for lc in lcs if len(lc) > 200]
        if not lcs:
            return []
        raw = ingest.stitch(lcs)
    except Exception as e:
        return []
    out = []
    mine = all_toi[all_toi.tic == tic]
    for r in lt[lt.tic == tic].itertuples():
        D = r.duration_hr / 24
        eph = Ephemeris(r.period, r.epoch_btjd, D, r.depth_ppm * 1e-6)
        try:
            eph = refine_ephemeris(raw, eph, r.epoch_err, r.period_err)
            others = [Ephemeris(o.period, o.epoch_btjd, o.duration_hr / 24) for o in mine.itertuples()
                      if o.toi != r.toi and np.isfinite(o.duration_hr) and o.duration_hr > 0]
            rec = signal_record(raw, eph, others)
        except Exception as e:
            continue
        if not rec:
            continue
        rec.update(toi=r.toi, tic=tic, label=r.label, disposition=r.disposition, split=r.split,
                   sectors=",".join(map(str, raw.sectors)), toi_depth_ppm=r.depth_ppm)
        out.append(rec)
    return out

if __name__ == "__main__":
    tics = sorted(lt.tic.unique())
    recs = []
    with ProcessPoolExecutor(4) as ex:
        for i, rs in enumerate(ex.map(one, tics, chunksize=8)):
            recs.extend(rs)
            if i % 200 == 0:
                print(i, len(recs), flush=True)
    G = np.stack([r.pop("global_view") for r in recs])
    L = np.stack([r.pop("local_view") for r in recs])
    df = pd.DataFrame(recs)
    out = config.DATA_DIR / "training"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "tois_views.npz", global_view=G, local_view=L)
    df.to_csv(out / "tois_features.csv", index=False)
    print(len(df), df.label.value_counts().to_dict())
