"""Single-transit injection-recovery and false alarms on the 500 sector-75 field stars."""
import sys, math, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from exohunt import ingest
from exohunt.injection import Injection, transit_model, field_dir
from exohunt.search.single import single_event_search

warnings.filterwarnings("ignore")
hosts = pd.read_csv("results/s0075/injections/hosts.csv")
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 6
rng = np.random.default_rng(105)

def run_star(args):
    warnings.filterwarnings("ignore")
    h, injs = args
    p = field_dir(75) / f"qlp_s0075_{int(h['tic'])}.fits"
    if not p.exists():
        return []
    lc = ingest.read_qlp(p)
    rows = []
    dets = single_event_search(lc, min_ses=0)
    top = dets[0] if dets else None
    rows.append(dict(kind="noinj", tic=h["tic"], top_ses=top.sde if top else np.nan,
                     top_t0=top.t0 if top else np.nan, top_dur=top.duration if top else np.nan,
                     ses_ratio_up=top.extra["ses_ratio_up"] if top else np.nan))
    for inj in injs:
        model, dur = transit_model(lc.time, inj)
        raw = lc.subset(np.arange(len(lc)), flux=lc.flux * model)
        dets = single_event_search(raw, min_ses=0)
        intr = np.abs(lc.time - inj.t0) < 0.5 * dur
        sigma = 1.4826 * np.median(np.abs(np.diff(lc.flux))) / np.sqrt(2)
        depth = float(np.mean(1 - model[intr])) if intr.any() else 0.0
        snr = depth * math.sqrt(intr.sum()) / sigma if intr.sum() else 0.0
        hit = [d for d in dets if abs(d.t0 - inj.t0) < max(0.5 * dur, 0.05)]
        rows.append(dict(kind="inj", tic=h["tic"], period=inj.period, t0=inj.t0, rp_re=inj.rp_re, b=inj.b,
                         duration_d=dur, depth_inj=float(1 - model.min()), snr_expected=snr, n_in=int(intr.sum()),
                         found_ses=hit[0].sde if hit else np.nan, found_rank=hit[0].rank if hit else np.nan,
                         found_depth=hit[0].depth if hit else np.nan, top_ses=dets[0].sde if dets else np.nan))
    return rows

if __name__ == "__main__":
    jobs = []
    k = 0
    for h in hosts.to_dict("records"):
        p = field_dir(75) / f"qlp_s0075_{int(h['tic'])}.fits"
        if not p.exists():
            continue
        t = ingest.read_qlp(p).time
        r, m = float(h["rad"]), float(h["mass"]) if np.isfinite(h["mass"]) else float(np.clip(h["rad"] ** 1.25, .1, 2))
        injs = []
        for _ in range(N_PER):
            P = float(np.exp(rng.uniform(np.log(30), np.log(400))))
            injs.append(Injection(k, int(h["tic"]), P, float(rng.choice(t)), float(np.exp(rng.uniform(np.log(1.5), np.log(16)))),
                                  float(rng.uniform(0, 0.9)), r, m)); k += 1
        jobs.append((h, injs))
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for i, rs in enumerate(ex.map(run_star, jobs, chunksize=4)):
            rows.extend(rs)
            if i % 100 == 0: print(i, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/s0075/injections/single_events.csv", index=False)
    print(df.groupby("kind").size())
