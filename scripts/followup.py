"""Look for more transits of the lead candidates in every other QLP sector of the same star.
Box scan at the candidate's duration over each sector; lists every dip with SES >= 6 and draws
one page per star (each sector, binned, with the dips marked, plus zooms on the strongest).
Usage: python scripts/followup.py"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from astroquery.mast import Observations
from exohunt import ingest
from exohunt.detrend import detrend
from exohunt.search.single import box_scan, robust_std

LEADS = [(99412936, 92), (278708716, 95), (402710280, 102), (219413978, 98)]
out = Path("results/followup"); out.mkdir(parents=True, exist_ok=True)
cand = pd.read_csv("results/multi/candidates.csv")


def qlp_sectors(tic):
    t = Observations.query_criteria(target_name=str(tic), provenance_name="QLP", dataproduct_type="timeseries")
    return sorted(set(int(s) for s in t["sequence_number"]))


rows = []
for tic, sec0 in LEADS:
    c = cand[(cand.tic == tic) & (cand.sector == sec0)].iloc[0]
    dur, depth = float(c.duration), float(c.depth)
    secs = qlp_sectors(tic)
    fig, axes = plt.subplots(len(secs) + 1, 1, figsize=(12, 1.9 * (len(secs) + 1)))
    events = []
    for ax, s in zip(axes, secs):
        p = ingest.download_qlp(tic, s)
        raw = ingest.read_qlp(p)
        flat, _ = detrend(raw, window_length=max(0.75, 3 * dur))
        t, f = flat.time, flat.flux
        ctr, dep, n_in, ok = box_scan(t, f, dur)
        sig = robust_std(f - 1)
        ses = dep / (sig / np.sqrt(np.maximum(n_in, 1)))
        ses[~ok] = np.nan
        # strongest separate dips
        order = np.argsort(-np.nan_to_num(ses, nan=-1))
        taken = []
        for i in order:
            if not np.isfinite(ses[i]) or ses[i] < 6 or len(taken) >= 3:
                break
            if all(abs(ctr[i] - ctr[j]) > 2 * dur for j in taken):
                taken.append(i)
                events.append(dict(tic=tic, sector=s, t=round(float(ctr[i]), 3), ses=round(float(ses[i]), 1),
                                   depth_ppm=round(float(dep[i]) * 1e6), cand_depth_ppm=round(depth * 1e6),
                                   cadence_min=round(float(np.median(np.diff(raw.time))) * 1440, 1),
                                   point_std_ppm=round(sig * 1e6)))
        nb = max(1, int(round(dur / 4 / np.median(np.diff(t)))))
        m = len(t) // nb * nb
        ax.plot(t, f, ",", color="0.75")
        ax.plot(t[:m].reshape(-1, nb).mean(1), f[:m].reshape(-1, nb).mean(1), ".", ms=2, color="k")
        for i in taken:
            ax.axvline(ctr[i], color="r", alpha=0.4, lw=3)
        ax.set_ylim(1 - 2.5 * max(depth, 4 * sig / np.sqrt(nb)), 1 + 1.5 * max(depth, 4 * sig / np.sqrt(nb)))
        ax.text(0.005, 0.85, f"S{s}", transform=ax.transAxes, fontsize=9)
    # zoom: the candidate event and the strongest other one
    ax = axes[-1]
    ev = pd.DataFrame(events)
    other = ev[ev.sector != sec0].sort_values("ses", ascending=False).head(1) if len(ev) else ev
    for e, col in [(dict(sector=sec0, t=float(c.t0)), "C0")] + [(r._asdict(), "C3") for r in other.itertuples()]:
        raw = ingest.read_qlp(ingest.qlp_path(tic, e["sector"]))
        flat, _ = detrend(raw, window_length=max(0.75, 3 * dur))
        x = (flat.time - e["t"]) * 24
        k = np.abs(x) < 4 * dur * 24
        ax.plot(x[k], flat.flux[k], ".", ms=2, color=col, label=f"S{e['sector']} {e['t']:.2f}")
    ax.legend(fontsize=8); ax.set_xlabel("hours from event")
    fig.suptitle(f"TIC {tic}: candidate S{sec0} t0={c.t0:.2f}, {depth*1e6:.0f} ppm, {dur*24:.1f} h; QLP sectors {secs}")
    fig.tight_layout(); fig.savefig(out / f"tic{tic}_all_sectors.png", dpi=110); plt.close(fig)
    rows += events
    print(tic, secs, flush=True)
pd.DataFrame(rows).to_csv(out / "events.csv", index=False)
print(pd.DataFrame(rows).to_string())
