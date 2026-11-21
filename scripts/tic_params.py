"""Attach TIC stellar parameters to a QLP sector target list (batched MAST queries)."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from astroquery.mast import Catalogs

sector = int(sys.argv[1])
src = f"data/catalogs/qlp_targets_s{sector:04d}.csv"
out = f"data/catalogs/qlp_targets_s{sector:04d}_tic.csv"
tl = pd.read_csv(src); tl.columns = ["tic", "ra", "dec"]
ids = [str(int(x)) for x in tl.tic]
cols = ["ID", "Tmag", "Teff", "rad", "mass", "logg", "rho", "lumclass", "d", "contratio", "objType", "disposition"]

def q(chunk):
    for attempt in range(4):
        try:
            c = Catalogs.query_criteria(catalog="Tic", ID=chunk)
            return pd.DataFrame({k: np.array(c[k]) for k in cols if k in c.colnames})
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()

chunks = [ids[i:i + 2000] for i in range(0, len(ids), 2000)]
with ThreadPoolExecutor(4) as ex:
    parts = list(ex.map(q, chunks))
df = pd.concat(parts).rename(columns={"ID": "tic"})
df["tic"] = df["tic"].astype(int)
df = tl.merge(df, on="tic", how="left")
df.to_csv(out, index=False)
print(len(df), df.Teff.notna().sum())
