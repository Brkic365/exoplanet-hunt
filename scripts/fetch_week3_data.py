"""Download week-3 inputs: sector-75 TOI hosts + field stars, labelled-TOI training light curves."""
import pandas as pd
from exohunt import catalog, ingest, labels, config

toi = catalog.fetch_toi()
s75 = catalog.tois_in_sector(toi, 75, None, max_tmag=99)
ingest.download_many(sorted(set(s75.tic.astype(int))), 75, workers=12)
hosts = pd.read_csv("results/s0075/injections/hosts.csv")
import exohunt.injection as inj
print("s75 TOI hosts done", flush=True)
# field stars live in s0075_field (see README); reuse download into the same layout
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
d = config.LC_DIR / "s0075_field"; d.mkdir(parents=True, exist_ok=True)
def get(tic):
    p = d / f"qlp_s0075_{int(tic)}.fits"
    if p.exists(): return
    for v in ingest.QLP_VERSIONS:
        r = requests.get(ingest.qlp_url(tic, 75, v), timeout=120)
        if r.ok:
            p.write_bytes(r.content); return
with ThreadPoolExecutor(12) as ex: list(ex.map(get, hosts.tic))
print("field done", flush=True)
lt = labels.labelled_tois(toi)
print(len(lt), lt.label.value_counts().to_dict(), lt.split.value_counts().to_dict(), flush=True)
files = labels.download_for(lt, max_sectors=2)
files.to_csv(config.CATALOG_DIR / "training_files.csv", index=False)
print("training files", len(files), files.tic.nunique(), flush=True)
