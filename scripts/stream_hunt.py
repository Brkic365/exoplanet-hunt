"""Download + hunt a sector's sample in one stream, so a whole sector never sits on disk.
Light curves whose signals all fail vetting are deleted right after the hunt; the rest stay
for week3_score.py (local-noise re-check) and the vetting sheets.
Writes the same files as week3_hunt.py. Usage: python scripts/stream_hunt.py SECTOR"""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
from exohunt import config, ingest
from exohunt.hunt import hunt_star


def job(p):
    try:
        rs = hunt_star(p)
    except Exception as e:
        rs = [{"tic": int(p.stem.split("_")[-1]), "error": repr(e)}]
    if not any(r.get("vet_ok") for r in rs):
        p.unlink(missing_ok=True)
    return rs


if __name__ == "__main__":
    sector = int(sys.argv[1])
    tics = pd.read_csv(config.CATALOG_DIR / f"s{sector:04d}_sample.csv").tic.tolist()
    if len(sys.argv) > 2:  # quick test on the first N stars
        tics = tics[: int(sys.argv[2])]
    recs, n_lc = [], 0
    with ThreadPoolExecutor(24) as dl, ProcessPoolExecutor(10) as ex:
        hunts = []
        for i, f in enumerate(as_completed([dl.submit(ingest._safe_download, t, sector) for t in tics])):
            p = f.result()
            if p:
                n_lc += 1
                hunts.append(ex.submit(job, p))
            if i % 2000 == 0:
                print("downloaded", i, flush=True)
        for h in hunts:
            recs.extend(h.result())
    ok = [r for r in recs if "global_view" in r]
    G = np.stack([r.pop("global_view") for r in ok]); L = np.stack([r.pop("local_view") for r in ok])
    np.savez_compressed(config.DATA_DIR / f"s{sector:04d}_views.npz", global_view=G, local_view=L)
    pd.DataFrame(ok).to_csv(config.DATA_DIR / f"s{sector:04d}_signals_raw.csv", index=False)
    print("sector", sector, "sample", len(tics), "light curves", n_lc, "signals", len(ok),
          "errors", sum("error" in r for r in recs), flush=True)
