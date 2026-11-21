"""Injection-recovery: plant synthetic transits in real light curves, run the
full pipeline (detrend + search + cross-match), and record what came back.

Transits are injected into the *raw* QLP flux, before detrending, so the test
measures everything the real search does to a signal, including what the
biweight filter eats. Each injection also records diagnostics that separate
"the pipeline lost it" from "the noise hides it":

* `snr_expected`: injected depth x sqrt(in-transit points) / point scatter of
  the detrended, transit-free light curve. Roughly what a perfect matched
  filter would see.
* `depth_retained`: mean depth measured at the injected transits *after*
  detrending, divided by the injected depth. Below 1 means detrending ate it.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, ingest
from .crossmatch import match_one
from .detrend import detrend
from .ingest import LightCurve
from .search import SEARCHERS, iterative_search

R_EARTH_PER_R_SUN = 109.1
G_DAY = 2.959122e-4  # G in AU^3 / (M_sun day^2)
AU_PER_R_SUN = 215.03


@dataclass
class Injection:
    id: int
    tic: int
    period: float  # d
    t0: float  # BTJD
    rp_re: float  # planet radius, Earth radii
    b: float  # impact parameter
    r_star: float  # R_sun
    m_star: float  # M_sun


def field_dir(sector: int) -> Path:
    return config.LC_DIR / f"s{sector:04d}_field"


def a_over_rstar(period, m_star, r_star):
    a_au = (G_DAY * m_star * period**2 / (4 * math.pi**2)) ** (1 / 3)
    return a_au * AU_PER_R_SUN / r_star


def transit_duration(period, rp_rs, a_rs, b):
    """Total duration T14 [d] for a circular orbit (Seager & Mallen-Ornelas 2003)."""
    arg = np.sqrt(max((1 + rp_rs) ** 2 - b**2, 0.0)) / (a_rs * math.sin(math.acos(min(b / a_rs, 1.0))))
    return period / math.pi * math.asin(min(arg, 1.0))


def transit_model(time, inj: Injection, u=(0.4, 0.25), exp_time_d: float = 200 / 86400):
    """Quadratic limb-darkened transit (batman), integrated over the 200 s FFI exposure."""
    import batman

    rp_rs = inj.rp_re / R_EARTH_PER_R_SUN / inj.r_star
    a_rs = a_over_rstar(inj.period, inj.m_star, inj.r_star)
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a = inj.t0, inj.period, rp_rs, a_rs
    p.inc = math.degrees(math.acos(min(inj.b / a_rs, 1.0)))
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, list(u), "quadratic"
    m = batman.TransitModel(p, np.asarray(time, float), supersample_factor=5, exp_time=exp_time_d)
    return m.light_curve(p), transit_duration(inj.period, rp_rs, a_rs, inj.b)


def draw_injections(hosts: pd.DataFrame, n_per_star: int, t_start: float, rng: np.random.Generator,
                    p_range=(0.6, 13.5), r_range=(0.5, 16.0)) -> list[Injection]:
    """Period and radius log-uniform, impact parameter uniform in [0, 0.9]."""
    out = []
    i = 0
    for h in hosts.itertuples():
        r = float(h.rad)
        m = float(h.mass) if np.isfinite(h.mass) and 0.08 < h.mass < 3 else float(np.clip(r**1.25, 0.1, 2.0))
        for _ in range(n_per_star):
            P = float(np.exp(rng.uniform(*np.log(p_range))))
            out.append(Injection(i, int(h.tic), P, float(t_start + rng.uniform(0, P)),
                                 float(np.exp(rng.uniform(*np.log(r_range)))), float(rng.uniform(0, 0.9)), r, m))
            i += 1
    return out


def _robust_std(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def run_injection(inj: Injection, path: Path, searcher: str = "tls", window_length: float = 0.75,
                  flux_column: str = "SAP_FLUX", detrend_method: str = "biweight", max_signals: int = 2,
                  min_sde: float = 9.0, **search_kw) -> dict:
    t_run = time.time()
    lc = ingest.read_qlp(path, flux_column=flux_column)
    model, dur = transit_model(lc.time, inj)
    depth_inj = float(1 - model.min())
    raw = lc.subset(np.arange(len(lc)), flux=lc.flux * model)
    if detrend_method == "none":
        flat = raw
    else:
        flat, _ = detrend(raw, window_length=window_length, method=detrend_method)

    # diagnostics on the detrended flux at the injected transits
    ph = ((flat.time - inj.t0 + 0.5 * inj.period) % inj.period) - 0.5 * inj.period
    core = np.abs(ph) < 0.25 * dur  # central half, where the model is ~ full depth
    intr = np.abs(ph) < 0.5 * dur
    oot = np.abs(ph) > dur
    sigma = _robust_std(flat.flux[oot]) if oot.sum() > 10 else float("nan")
    base = float(np.median(flat.flux[oot])) if oot.any() else 1.0
    m_core = float(np.mean(1 - model[np.isin(lc.time, flat.time[core])])) if core.any() else float("nan")
    depth_meas = float(base - np.mean(flat.flux[core])) if core.any() else float("nan")
    n_tr = len(np.unique(np.round((flat.time[intr] - inj.t0) / inj.period))) if intr.any() else 0
    mean_depth = float(np.mean(1 - model[np.isin(lc.time, flat.time[intr])])) if intr.any() else 0.0
    snr_exp = mean_depth * math.sqrt(intr.sum()) / sigma if sigma and sigma > 0 else float("nan")

    dets = iterative_search(flat, SEARCHERS[searcher], max_signals=max_signals, min_sde=min_sde, **search_kw)
    status, best = "missed", None
    for d in dets:
        if d.sde < min_sde:
            continue
        kind, _ = match_one(d.period, d.t0, inj.period, inj.t0, dur)
        if kind in ("match", "alias"):
            status, best = kind, d
            break
    top = dets[0] if dets else None
    row = asdict(inj)
    row.update(
        duration_d=dur, depth_inj=depth_inj, n_transits=n_tr, sigma=sigma, snr_expected=snr_exp,
        depth_retained=depth_meas / m_core if m_core and m_core > 0 else float("nan"),
        status=status, recovered=status in ("match", "alias"),
        det_period=best.period if best else (top.period if top else np.nan),
        det_sde=best.sde if best else (top.sde if top else np.nan),
        det_rank=best.rank if best else np.nan,
        tmag=lc.meta.get("TESSMAG"), searcher=searcher, window=window_length, flux_column=flux_column,
        detrend_method=detrend_method, runtime_s=time.time() - t_run,
    )
    return row


def _run(args):
    inj, path, kw = args
    try:
        return run_injection(inj, path, **kw)
    except Exception as e:  # record and keep going
        r = asdict(inj)
        r.update(status=f"error: {e!r}", recovered=False)
        return r


def run_campaign(injections: list[Injection], sector: int, out_csv: Path, workers: int | None = None,
                 **kw) -> pd.DataFrame:
    """Run every injection in parallel; resumable via `out_csv` (by injection id)."""
    workers = workers or os.cpu_count() or 1
    done: set[int] = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv)["id"].astype(int))
    todo = [i for i in injections if i.id not in done]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = not out_csv.exists()
    d = field_dir(sector)
    t_start = time.time()
    buf = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_run, (inj, d / ingest.qlp_path(inj.tic, sector).name, kw)) for inj in todo]
        for n, f in enumerate(as_completed(futs), 1):
            buf.append(f.result())
            if len(buf) >= 50 or n == len(todo):
                pd.DataFrame(buf).to_csv(out_csv, mode="a", header=header, index=False)
                header = False
                buf = []
                el = time.time() - t_start
                print(f"[{n}/{len(todo)}] {el:.0f}s elapsed, ~{el / n * (len(todo) - n):.0f}s left", flush=True)
    return pd.read_csv(out_csv)


# ----------------------------------------------------------------------------- summaries

P_EDGES = np.array([0.6, 1, 2, 3, 5, 7, 10, 13.5])
R_EDGES = np.array([0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 11, 16])


def heatmap(df: pd.DataFrame, x="period", y="rp_re", x_edges=P_EDGES, y_edges=R_EDGES) -> pd.DataFrame:
    """Recovery fraction per (period, radius) cell, long format with counts."""
    d = df[~df["status"].astype(str).str.startswith("error")].copy()
    d["xb"] = pd.cut(d[x], x_edges)
    d["yb"] = pd.cut(d[y], y_edges)
    g = d.groupby(["yb", "xb"], observed=False)["recovered"]
    out = pd.DataFrame({"n": g.size(), "n_rec": g.sum()}).reset_index()
    out["frac"] = out["n_rec"] / out["n"].where(out["n"] > 0)
    out["x_lo"] = [iv.left for iv in out["xb"]]
    out["x_hi"] = [iv.right for iv in out["xb"]]
    out["y_lo"] = [iv.left for iv in out["yb"]]
    out["y_hi"] = [iv.right for iv in out["yb"]]
    return out.drop(columns=["xb", "yb"])


def recovery_vs(df: pd.DataFrame, col: str, edges) -> pd.DataFrame:
    d = df[~df["status"].astype(str).str.startswith("error")].copy()
    d["bin"] = pd.cut(d[col], edges)
    g = d.groupby("bin", observed=False)
    return pd.DataFrame({"n": g.size(), "recall": g["recovered"].mean(),
                         "depth_retained_median": g["depth_retained"].median()})


SNR_EDGES = np.array([0, 3, 5, 7, 9, 12, 16, 25, 50, 1e5])
DUR_EDGES_HR = np.array([0, 1.5, 2.5, 3.5, 5, 7, 12])


def report(df: pd.DataFrame, outdir: Path, tag: str) -> dict:
    """Write heatmap CSV/PNG + summary tables for one campaign; return headline numbers."""
    from .plots import heatmap_figure

    outdir.mkdir(parents=True, exist_ok=True)
    df = df[~df["status"].astype(str).str.startswith("error")].copy()
    df["duration_hr"] = df["duration_d"] * 24
    p_edges = P_EDGES if df["period"].max() <= P_EDGES[-1] + 1e-9 else np.r_[P_EDGES[:-1], 13.5, 20, 27]
    cells = heatmap(df, x_edges=p_edges)
    cells.to_csv(outdir / f"heatmap_{tag}.csv", index=False)
    heatmap_figure(cells, outdir / f"heatmap_{tag}.png", title=f"Injection-recovery, sector 75 ({tag})")
    by_snr = recovery_vs(df, "snr_expected", SNR_EDGES)
    by_dur = recovery_vs(df, "duration_hr", DUR_EDGES_HR)
    by_snr.to_csv(outdir / f"by_snr_{tag}.csv")
    by_dur.to_csv(outdir / f"by_duration_{tag}.csv")
    hi = df[df["snr_expected"] >= 12]
    return dict(tag=tag, n=len(df), recall=df["recovered"].mean(), recall_snr12=hi["recovered"].mean(),
                n_snr12=len(hi), depth_retained_median=df["depth_retained"].median(),
                runtime_s_median=df["runtime_s"].median())


def toi_expected_snr(cm: pd.DataFrame, toi: pd.DataFrame, sector: int, window_length: float = 0.75) -> pd.DataFrame:
    """Add `snr_expected` (same definition as for injections) and
    `depth_retained` to a TOI cross-match table, from each TOI's catalogue
    depth, duration and ephemeris on our detrended light curve."""
    toi = toi.drop_duplicates("toi").set_index("toi")
    flat_cache: dict[int, LightCurve] = {}
    rows = []
    for r in cm.itertuples():
        t = toi.loc[r.toi]
        P, T0, dur = float(t["period"]), float(t["epoch_btjd"]), float(t["duration_hr"]) / 24
        if not (np.isfinite(P) and P > 0 and np.isfinite(dur) and dur > 0):
            rows.append(dict(toi=r.toi, snr_expected=np.nan, depth_retained=np.nan))
            continue
        if r.tic not in flat_cache:
            flat_cache[r.tic] = detrend(ingest.read_qlp(ingest.qlp_path(r.tic, sector)), window_length=window_length)[0]
        f = flat_cache[r.tic]
        ph = ((f.time - T0 + 0.5 * P) % P) - 0.5 * P
        intr, core, oot = np.abs(ph) < 0.5 * dur, np.abs(ph) < 0.25 * dur, np.abs(ph) > dur
        sig = _robust_std(f.flux[oot])
        depth = float(t["depth_ppm"]) * 1e-6
        ret = (np.median(f.flux[oot]) - np.mean(f.flux[core])) / depth if core.any() and depth > 0 else np.nan
        rows.append(dict(toi=r.toi, snr_expected=depth * math.sqrt(intr.sum()) / sig, depth_retained=ret))
    return cm.merge(pd.DataFrame(rows), on="toi", how="left")
