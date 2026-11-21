"""Run the full pipeline over a sector's sample, then score every signal with the CNN."""
import os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from exohunt import config
from exohunt.hunt import hunt_star

sector = int(sys.argv[1]); out = Path(f"results/s{sector:04d}"); out.mkdir(parents=True, exist_ok=True)
lcdir = config.LC_DIR / f"s{sector:04d}"

def job(p):
    try:
        return hunt_star(p)
    except Exception as e:
        return [{"tic": int(p.stem.split("_")[-1]), "error": repr(e)}]

if __name__ == "__main__":
    paths = sorted(lcdir.glob("*.fits"))
    if len(sys.argv) > 2:
        paths = paths[: int(sys.argv[2])]
    recs = []
    with ProcessPoolExecutor(4) as ex:
        for i, rs in enumerate(ex.map(job, paths, chunksize=16)):
            recs.extend(rs)
            if i % 1000 == 0:
                print(i, len(recs), flush=True)
    ok = [r for r in recs if "global_view" in r]
    G = np.stack([r.pop("global_view") for r in ok]); L = np.stack([r.pop("local_view") for r in ok])
    df = pd.DataFrame(ok)
    np.savez_compressed(config.DATA_DIR / f"s{sector:04d}_views.npz", global_view=G, local_view=L)
    df.to_csv(config.DATA_DIR / f"s{sector:04d}_signals_raw.csv", index=False)
    print("stars", len(paths), "signals", len(df), "errors", sum("error" in r for r in recs), flush=True)
