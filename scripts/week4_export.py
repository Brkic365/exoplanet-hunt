"""Export the data the portfolio page needs into site/data/ (compact JSON).

    python scripts/week4_export.py            # everything
    python scripts/week4_export.py stars      # only the per-star walkthrough files

Writes
  backdrop.bin    faint backdrop: other QLP targets of the same sectors (not searched)
  sky.json        every star searched (RA, Dec, T mag, best classifier score, class)
  stars/<tic>.json one file per walkthrough star: raw flux, trend, detrended flux,
                  periodogram (or single-event statistic), folded dip, verdict
  heatmap.json    injection-recovery grid + one found and one missed example per cell
  metrics.json    the validation numbers (recall, classifier, speed, sector-105 funnel)
  stars.json      index of walkthrough stars, incl. the 21 hand-vetted sector-105 signals
Needs network for QLP light curves (MAST) and, for sky.json, the TIC parameters
of the sector-105 target list (scripts/tic_params.py 105).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exohunt import config, ingest
from exohunt.detrend import detrend
from exohunt.injection import Injection, transit_model
from exohunt.search import single
from exohunt.search.bls import bls_periodogram_cpu, make_grid, sde
from exohunt.vetting import Ephemeris, masked_detrend, phase_days

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "site" / "data"

# Walkthrough stars from sector 75: known planets the pipeline recovered on its own.
S75_FEATURED = [
    ("1194.01", 147950620, "A hot Saturn: the easy case"),
    ("1430.01", 293954617, "A 2 Earth-radius planet, 700 ppm deep"),
    ("4481.01", 239332587, "An ultra-short period planet, 22 hours"),
    ("1184.01", 233087860, "A sub-Neptune the classifier never saw in training"),
]


def r(x, nd=0):
    """Round for JSON; ints when nd == 0. NaN/inf -> None."""
    a = np.asarray(x, float)
    out = np.where(np.isfinite(a), np.round(a, nd), np.nan)
    if nd == 0:
        return [None if not np.isfinite(v) else int(v) for v in out]
    return [None if not np.isfinite(v) else float(v) for v in out]


def num(v, nd=4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def bin_series(t, ys, width):
    """Mean of each y in time bins of `width` days; empty bins dropped."""
    k = np.floor((t - t[0]) / width).astype(int)
    uk, inv = np.unique(k, return_inverse=True)
    cnt = np.bincount(inv)
    tb = np.bincount(inv, t) / cnt
    return tb, [np.bincount(inv, y) / cnt for y in ys]


def envelope(x, y, n):
    """Downsample a periodogram to n points, keeping each chunk's maximum."""
    if len(x) <= n:
        return x, y
    idx = np.array_split(np.arange(len(x)), n)
    j = np.array([i[np.argmax(y[i])] for i in idx])
    return x[j], y[j]


def fold_view(lc, eph, half_width_d, nbins=60, max_points=900, rng=None):
    ph = phase_days(lc.time, eph) if np.isfinite(eph.period) else lc.time - eph.t0
    m = np.abs(ph) < half_width_d
    x, y = ph[m] * 24, (lc.flux[m] - 1) * 1e6
    edges = np.linspace(-half_width_d * 24, half_width_d * 24, nbins + 1)
    k = np.digitize(x, edges) - 1
    bx, by = [], []
    for i in range(nbins):
        s = k == i
        if s.sum() >= 3:
            bx.append(0.5 * (edges[i] + edges[i + 1]))
            by.append(float(np.median(y[s])))
    if len(x) > max_points:
        rng = rng or np.random.default_rng(0)
        j = np.sort(rng.choice(len(x), max_points, replace=False))
        x, y = x[j], y[j]
    return {"x": r(x, 2), "y": r(y), "bx": r(bx, 2), "by": r(by)}


def star_payload(path: Path, *, method: str, eph: Ephemeris | None, info: dict) -> dict:
    raw = ingest.read_qlp(path)
    flat, trend = detrend(raw, window_length=0.75)
    t0 = float(raw.time[0])
    # stage 1-2: raw, trend, flat on a 10-min grid (QLP cadence is 200 s)
    tr = np.interp(raw.time, raw.time[np.isfinite(trend)], trend[np.isfinite(trend)])
    fl_full = np.interp(raw.time, flat.time, flat.flux)
    keep = np.isin(raw.time, flat.time)
    tb, (rawb, trb, flb) = bin_series(raw.time, [raw.flux, tr, np.where(keep, fl_full, np.nan)], 10 / 1440)
    lcj = {"t0": round(t0, 4), "t": r(tb - t0, 4), "raw": r((rawb - 1) * 1e6), "trend": r((trb - 1) * 1e6),
           "flat": r((flb - 1) * 1e6)}

    # stage 3: search statistic
    if method == "single":
        flat_long, _ = detrend(raw, window_length=2.0)
        best = None
        for d, c, ses, _up, _dep, sides in single.scan(flat, flat_long):
            if best is None or abs(d - eph.duration) < abs(best[0] - eph.duration):
                best = (d, c, np.where(sides, ses, np.nan))
        d, c, ses = best
        cx, cy = envelope(c, np.nan_to_num(ses, nan=0.0), 700)
        search = {"kind": "ses", "x": r(cx - t0, 3), "y": r(cy, 2), "dur_h": round(d * 24, 2),
                  "peak": {"t": round(eph.t0 - t0, 3), "value": num(info.get("search_sde"), 2)}}
    else:
        grid = make_grid(float(flat.time.max() - flat.time.min()))
        power, jb, kb, tref = bls_periodogram_cpu(flat, grid)
        s, detr = sde(power)
        snorm = (detr - detr.mean()) / detr.std()
        px, py = envelope(grid.periods, snorm, 700)
        i = int(np.argmax(detr))
        if eph is None:  # use our own BLS peak
            from exohunt.search.bls import to_detection
            det = to_detection(flat, grid, power, jb, kb, tref)
            eph = Ephemeris(det.period, det.t0, det.duration, det.depth)
            info.setdefault("snr", det.snr)
        search = {"kind": "periodogram", "x": r(px, 5), "y": r(py, 2),
                  "peak": {"period": round(float(grid.periods[i]), 5), "sde": round(s, 2)},
                  "signal_period": round(eph.period, 5), "n_periods": int(len(grid.periods))}

    # stage 4: fold on a light curve re-detrended with the transit masked
    mflat = masked_detrend(raw, eph)
    hw = min(max(3 * eph.duration, 0.25), 0.5 * eph.period if np.isfinite(eph.period) else 9)
    fold = fold_view(mflat, eph, hw)
    fold.update(period=num(eph.period, 5), t0=round(eph.t0, 4), dur_h=round(eph.duration * 24, 2))
    meta = raw.meta
    return {"tic": int(raw.tic), "sector": int(raw.sectors[0]), "method": method,
            "star": {"tmag": num(meta.get("TESSMAG"), 2), "teff": num(meta.get("TEFF"), 0),
                     "rstar": num(meta.get("RADIUS"), 3), "ra": num(meta.get("RA_OBJ"), 4),
                     "dec": num(meta.get("DEC_OBJ"), 4)},
            "lc": lcj, "search": search, "fold": fold, **info}


def write(name, obj):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} kB", flush=True)


# ------------------------------------------------------------------ stars

def hand_vetted() -> pd.DataFrame:
    s = pd.read_csv(RES / "s0105/signals.csv")
    h = pd.read_csv(RES / "s0105/hand_vetting.csv")
    m = h.merge(s[s.rank_ok], on=["tic", "method"], how="left")
    assert len(m) == len(h), "one passing signal per hand-vetted star expected"
    return m


def verdict_group(v: str) -> str:
    v = v.lower()
    if "ctoi" in v:
        return "ctoi"
    if "false positive" in v:
        return "fp"
    if "variab" in v:
        return "variability"
    if "inconclusive" in v:
        return "inconclusive"
    return "noise"


def export_stars():
    index = []
    pred = pd.read_csv(RES / "classifier/predictions.csv")
    vk = pd.read_csv(RES / "s0075/vetting_known.csv")
    rec = pd.read_csv(RES / "s0075/recovery_sde9_all_tmag13.5.csv")
    for toi, tic, blurb in S75_FEATURED:
        path = ingest.download_qlp(tic, 75)
        p = pred[(pred.toi.astype(str) == toi) & (pred.kind == "toi")].iloc[0]
        v = vk[vk.toi.astype(str) == toi].iloc[0]
        rr = rec[rec.toi.astype(str) == toi].iloc[0]
        info = {"id": f"tic{tic}", "label": f"TOI-{toi}", "group": "known", "blurb": blurb,
                "known": {"disposition": rr.disposition, "toi_period": num(rr.toi_period, 5),
                          "toi_depth_ppm": num(rr.toi_depth_ppm, 0), "toi_radius_re": num(rr.toi_radius_re, 2)},
                "verdict": {"score": num(p.ensemble, 3), "p_cnn": num(p.cnn_views_vet, 3), "p_gbm": num(p.gbm_vet, 3),
                            "baseline": num(p.gbm_bls, 3), "split": p.split, "flags": v.flags if isinstance(v.flags, str) else "",
                            "snr": num(v.snr, 1), "oddeven_sigma": num(v.oddeven_sigma, 1), "sec_snr": num(v.sec_snr, 1),
                            "centroid_sigma": num(v.centroid_sigma, 1), "rp_re": num(v.rp_re, 2),
                            "depth_ppm": num(v.depth * 1e6, 0)}}
        d = star_payload(path, method="bls", eph=None, info=info)
        write(f"stars/tic{tic}.json", d)
        index.append({k: d[k] for k in ("id", "label", "group", "blurb", "sector", "method")})

    for _, h in hand_vetted().iterrows():
        path = ingest.download_qlp(int(h.tic), 105)
        eph = Ephemeris(h.period if h.method == "bls" else float("nan"), h.t0, h.duration, h.depth)
        g = verdict_group(h.verdict)
        info = {"id": f"tic{int(h.tic)}", "label": f"TIC {int(h.tic)}", "group": g, "search_sde": h.search_sde,
                "hand": {"verdict": h.verdict, "reason": h.reason},
                "verdict": {"score": num(h.score, 3), "p_cnn": num(h.p_cnn, 3), "p_gbm": num(h.p_gbm, 3),
                            "flags": h.flags if isinstance(h.flags, str) else "", "snr": num(h.snr, 1),
                            "oddeven_sigma": num(h.oddeven_sigma, 1), "sec_snr": num(h.sec_snr, 1),
                            "centroid_sigma": num(h.centroid_sigma, 1), "rp_re": num(h.rp_re, 2),
                            "depth_ppm": num(h.depth * 1e6, 0)}}
        d = star_payload(path, method=h.method, eph=eph, info=info)
        write(f"stars/tic{int(h.tic)}.json", d)
        index.append({k: d[k] for k in ("id", "label", "group", "sector", "method")} |
                     {"verdict": h.verdict, "reason": h.reason, "period": num(h.period, 4),
                      "t0": num(h.t0, 3), "dur_h": num(h.duration * 24, 1), "depth_ppm": num(h.depth * 1e6, 0),
                      "snr": num(h.snr, 1), "score": num(h.score, 3), "tmag": d["star"]["tmag"],
                      "teff": d["star"]["teff"]})
    write("stars.json", index)
    # vetting sheets for the candidate cards
    sheets = OUT / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    for f in (RES / "s0105/sheets").glob("*.png"):
        shutil.copy(f, sheets / f"{f.name.split('_')[0]}.png")


# ------------------------------------------------------------------ sky

def export_sky():
    """Sector-105 M/K dwarf sample + sector-75 TOI hosts and field stars."""
    tic = pd.read_csv(config.CATALOG_DIR / "qlp_targets_s0105_tic.csv")
    dw = tic[tic.lumclass == "DWARF"]
    m = dw[(dw.Teff < 3900) & (dw.Tmag <= 13.5)]
    k = dw[(dw.Teff >= 3900) & (dw.Teff < 5300) & (dw.Tmag <= 12)]
    s105 = pd.concat([m, k]).drop_duplicates("tic")[["tic", "ra", "dec", "Tmag", "Teff"]]
    s105.columns = ["tic", "ra", "dec", "tmag", "teff"]
    s105["sector"] = 105
    sig = pd.read_csv(RES / "s0105/signals.csv")
    live = sig[~sig.systematic & sig.vet_ok]
    best = live.groupby("tic").score.max()
    s105["score"] = s105.tic.map(best)
    s105["cls"] = np.where(s105.score.notna(), 1, 0)
    kn = pd.read_csv(RES / "s0105/toi_recovery.csv")
    s105.loc[s105.tic.isin(kn.tic), "cls"] = 4
    hv = hand_vetted()
    s105.loc[s105.tic.isin(hv.tic), "cls"] = 2
    s105.loc[s105.tic.isin(hv[hv.verdict.str.contains("CTOI")].tic), "cls"] = 3
    print("sector 105 sample", len(s105), flush=True)

    t75 = pd.read_csv(config.CATALOG_DIR / "qlp_targets_s0075.csv")
    t75.columns = ["tic", "ra", "dec"]
    rec = pd.read_csv(RES / "s0075/recovery_sde9_all_tmag13.5.csv")
    hosts = rec.groupby("tic").agg(tmag=("tmag", "first")).reset_index().merge(t75, on="tic")
    pred = pd.read_csv(RES / "classifier/predictions.csv")
    hosts["score"] = hosts.tic.map(pred[pred.kind == "toi"].groupby("tic").ensemble.max())
    hosts["cls"] = 4
    hosts.loc[hosts.tic.isin([t for _, t, _ in S75_FEATURED]), "cls"] = 5
    field = pd.read_csv(RES / "s0075/injections/hosts.csv")[["tic", "ra", "dec", "tmag", "teff"]]
    field["cls"], field["score"] = 0, np.nan
    s75 = pd.concat([hosts, field]).drop_duplicates("tic")
    s75["sector"] = 75
    allst = pd.concat([s105, s75], ignore_index=True)
    allst = allst[np.isfinite(allst.ra) & np.isfinite(allst.dec)]
    write("sky.json", {
        "n": int(len(allst)),
        "classes": {"0": "searched, nothing survived", "1": "signal passed vetting (scored)",
                    "2": "hand-vetted, rejected", "3": "known single-transit CTOI, new transit",
                    "4": "known TOI host", "5": "walkthrough star"},
        "tic": allst.tic.astype(int).tolist(), "ra": r(allst.ra, 3), "dec": r(allst.dec, 3),
        "tmag": r(allst.tmag, 2), "score": r(allst.score, 3), "cls": allst.cls.astype(int).tolist(),
        "sector": allst.sector.astype(int).tolist()})


def export_backdrop(n_per_sector=30000, seed=1):
    """Faint backdrop: a random subset of the other QLP targets in sectors 75 and 105
    (not searched), packed as uint16 RA (0..360) + int16 Dec (x 300)."""
    rng = np.random.default_rng(seed)
    parts = []
    for sec in (75, 105):
        t = pd.read_csv(config.CATALOG_DIR / f"qlp_targets_s{sec:04d}.csv")
        t.columns = ["tic", "ra", "dec"]
        parts.append(t.iloc[rng.choice(len(t), min(n_per_sector, len(t)), replace=False)])
    t = pd.concat(parts)
    ra = np.round(t.ra.to_numpy() / 360 * 65535).astype("<u2")
    dec = np.round(t.dec.to_numpy() * 300).astype("<i2")
    p = OUT / "backdrop.bin"
    p.write_bytes(ra.tobytes() + dec.tobytes())
    print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} kB ({len(t)} stars)")


# ------------------------------------------------------------------ heatmap

def export_heatmap(seed=4):
    rng = np.random.default_rng(seed)
    cells = pd.read_csv(RES / "s0075/injections/heatmap_tls_w0.75.csv")
    inj = pd.read_csv(RES / "s0075/injections/tls_w0.75.csv")
    out = []
    for _, c in cells.iterrows():
        sel = inj[(inj.period >= c.x_lo) & (inj.period < c.x_hi) & (inj.rp_re >= c.y_lo) & (inj.rp_re < c.y_hi)]
        ex = {}
        for key, rows in (("found", sel[sel.recovered]), ("missed", sel[~sel.recovered])):
            if len(rows) == 0:
                continue
            row = rows.iloc[rng.integers(len(rows))]
            path = ingest.download_qlp(int(row.tic), 75)
            if path is None:
                continue
            raw = ingest.read_qlp(path)
            iv = Injection(int(row.id), int(row.tic), row.period, row.t0, row.rp_re, row.b, row.r_star, row.m_star)
            model, _ = transit_model(raw.time, iv)
            lc = raw.subset(np.ones(len(raw), bool), flux=raw.flux * model)
            flat, _ = detrend(lc, window_length=0.75)
            eph = Ephemeris(row.period, row.t0, row.duration_d)
            fv = fold_view(flat, eph, min(max(3 * row.duration_d, 0.2), 0.5 * row.period), nbins=40, max_points=500,
                           rng=rng)
            fv.update(period=num(row.period, 3), rp_re=num(row.rp_re, 2), depth_ppm=num(row.depth_inj * 1e6, 0),
                      snr=num(row.snr_expected, 1), sde=num(row.det_sde, 1), tic=int(row.tic),
                      tmag=num(row.tmag, 2), n_transits=int(row.n_transits), dur_h=num(row.duration_d * 24, 2))
            ex[key] = fv
        out.append({"p": [c.x_lo, c.x_hi], "r": [c.y_lo, c.y_hi], "n": int(c.n), "found": int(c.n_rec),
                    "frac": num(c.frac, 3), "examples": ex})
        print("cell", c.x_lo, c.y_lo, len(ex), flush=True)
    write("heatmap.json", {"searcher": "TLS, SDE >= 9, 0.75 d biweight", "n_injections": int(len(inj)),
                           "n_hosts": int(inj.tic.nunique()), "cells": out})


# ------------------------------------------------------------------ metrics

def export_metrics():
    rv = pd.read_csv(RES / "s0075/recall_vs_snr.csv", header=[0, 1], index_col=0).iloc[1:]
    cols = rv.columns
    snr = {"bins": list(rv.index), "inj_tls": r(rv[cols[1]].astype(float), 3), "inj_n": r(rv[cols[0]].astype(float)),
           "inj_bls": r(rv[cols[3]].astype(float), 3), "toi_tls": r(rv[cols[5]].astype(float), 3),
           "toi_n": r(rv[cols[4]].astype(float))}
    se = pd.read_csv(RES / "s0075/injections/single_events_by_snr.csv")
    cl = pd.read_csv(RES / "classifier/metrics_test_planet_vs_fp.csv", index_col=0)
    cn = pd.read_csv(RES / "classifier/metrics_test_planet_vs_noise.csv", index_col=0)
    bench = pd.concat([pd.read_csv(RES / "s0075/benchmark_cpu4.csv").assign(hw="4-core cloud VM"),
                       pd.read_csv(RES / "s0075/benchmark_gpu.csv").assign(hw="laptop, RTX 4050")])
    funnel = pd.read_csv(RES / "s0105/funnel.csv", index_col=0).iloc[:, 0].to_dict()
    rec = pd.read_csv(RES / "s0075/recovery_sde9_all_tmag13.5.csv")
    write("metrics.json", {
        "recall_vs_snr": snr,
        "single_vs_snr": {"bins": se.b.tolist(), "n": r(se.n), "recall": r(se.r9, 3)},
        "classifier": {k: {m: num(v, 3) for m, v in row.items()} for k, row in cl.iterrows()},
        "classifier_noise": {k: {m: num(v, 3) for m, v in row.items()} for k, row in cn.iterrows()},
        "bench": [{"backend": b.backend, "hw": b.hw, "stars_per_s": num(b.stars_per_s, 1)} for _, b in bench.iterrows()],
        "funnel_s105": {k: int(v) for k, v in funnel.items()},
        "s75": {"tois": int(len(rec)), "hosts": int(rec.tic.nunique())},
    })


if __name__ == "__main__":
    what = set(sys.argv[1:]) or {"stars", "sky", "heatmap", "metrics"}
    config.ensure_dirs()
    if "metrics" in what:
        export_metrics()
    if "stars" in what:
        export_stars()
    if "heatmap" in what:
        export_heatmap()
    if "sky" in what:
        export_sky()
        export_backdrop()
