"""Single-event detector on sector-75 TOIs that have exactly one transit in the data."""
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from exohunt import catalog, ingest
from exohunt.search.single import single_event_search

toi = catalog.fetch_toi()
t = catalog.tois_in_sector(toi, 75, {"CP", "KP", "PC", "APC", "FP"}, max_tmag=99)
t = t[t.tic.map(lambda x: ingest.qlp_path(x, 75).exists())]

def one(r):
    warnings.filterwarnings("ignore")
    lc = ingest.read_qlp(ingest.qlp_path(r["tic"], 75))
    D = r["duration_hr"] / 24
    n = catalog.expected_transits(lc.time, r["period"], r["epoch_btjd"], D)
    if n != 1:
        return None
    k = np.ceil((lc.time[0] - r["epoch_btjd"]) / r["period"])
    tcs = r["epoch_btjd"] + np.arange(k, k + 3) * r["period"]
    tc = [x for x in tcs if lc.time[0] <= x <= lc.time[-1]][0]
    dets = single_event_search(lc, min_ses=0)
    hit = [d for d in dets if abs(d.t0 - tc) < max(0.5 * D, 0.05)]
    return dict(toi=r["toi"], tic=r["tic"], disposition=r["disposition"], period=r["period"], depth_ppm=r["depth_ppm"],
                duration_hr=r["duration_hr"], found_ses=hit[0].sde if hit else np.nan, found_rank=hit[0].rank if hit else np.nan)

with ProcessPoolExecutor(4) as ex:
    rows = [x for x in ex.map(one, [r for _, r in t.iterrows()]) if x]
df = pd.DataFrame(rows); df.to_csv("results/s0075/single_event_tois.csv", index=False)
for th in (7, 9):
    print(th, df.groupby(df.disposition.isin(["FP"])).apply(lambda g: f"{(g.found_ses>=th).sum()}/{len(g)}").to_dict())
print(df.sort_values("found_ses", ascending=False).to_string())
