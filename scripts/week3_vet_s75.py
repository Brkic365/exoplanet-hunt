"""Vetting checks on sector-75 detections that match known TOIs: planets vs known false positives."""
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from exohunt import ingest
from exohunt.vetting import Ephemeris, vet_signal

det = pd.read_csv("results/s0075/detections.csv")
rec = pd.concat([pd.read_csv("results/s0075/recovery_sde9_all_tmag99.csv"),
                 pd.read_csv("results/s0075/recovery_sde9_fp.csv")])
rec = rec[rec.status.isin(["match", "alias"])]
rows = rec.merge(det, left_on=["tic", "det_rank"], right_on=["tic", "rank"])

def one(r):
    lc = ingest.read_qlp(ingest.qlp_path(r["tic"], 75))
    o = det[(det.tic == r["tic"]) & (det["rank"] != r["rank"]) & (det.sde >= 7)]
    others = [Ephemeris(x.period, x.t0, x.duration) for x in o.itertuples()]
    m, _ = vet_signal(lc, Ephemeris(r["period"], r["t0"], r["duration"], r["depth"]), others)
    return {"toi": r["toi"], "tic": r["tic"], "disposition": r["disposition"], "period": r["period"],
            "sde": r["sde"], **m}

with ProcessPoolExecutor(4) as ex:
    out = pd.DataFrame(list(ex.map(one, [r for _, r in rows.iterrows()])))
out.to_csv("results/s0075/vetting_known.csv", index=False)
out["group"] = out.disposition.map(lambda d: "planet (CP/KP)" if d in ("CP", "KP") else ("FP" if d in ("FP", "FA") else "candidate (PC/APC)"))
flags = ["oddeven", "secondary", "centroid", "dump", "oneevent", "toobig", "toolong", "edge", "noisy"]
for f in flags:
    out[f] = out["flags"].fillna("").str.split(",").map(lambda s: f in s)
s = out.groupby("group").agg(n=("toi", "size"), flagged=("vet_ok", lambda v: (~v).mean()), **{f: (f, "mean") for f in flags})
print(s.round(2).to_string())
s.to_csv("results/s0075/vetting_known_summary.csv")
print(out[out.group=="planet (CP/KP)"][["toi","flags","oddeven_sigma","sec_snr","centroid_sigma","centroid_offset_px","rp_re","dur_ratio"]].loc[lambda x: x["flags"].fillna("")!=""].to_string())
