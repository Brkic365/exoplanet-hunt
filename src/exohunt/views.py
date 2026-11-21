"""Folded "views" of a signal for the classifier (after Shallue & Vanderburg 2018).

* global view: the whole orbit, 201 phase bins
* local view: +-2 transit durations around the transit, 61 bins, in channels
    0 all transits, 1 odd transits, 2 even transits,
    3 the same window around the strongest secondary dip,
    4 centroid displacement (in units of its out-of-transit scatter)

Flux channels are shifted to median 0 and scaled so the primary transit's
minimum is -1, so the network sees shape, not depth; depth and SNR go in as
scalar features instead.
"""

from __future__ import annotations

import numpy as np
from wotan import flatten

from .ingest import LightCurve
from .vetting import Ephemeris, phase_days, epoch_index, _robust_std

N_GLOBAL = 201
N_LOCAL = 61
LOCAL_HALF_WIDTH = 2.0  # durations


def binned(x: np.ndarray, y: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    """Median of y in n equal bins of x over [lo, hi); empty bins interpolated (NaN if all empty)."""
    out = np.full(n, np.nan)
    if len(x) == 0:
        return out
    idx = np.floor((x - lo) / (hi - lo) * n).astype(int)
    ok = (idx >= 0) & (idx < n) & np.isfinite(y)
    if not ok.any():
        return out
    idx, yy = idx[ok], y[ok]
    order = np.argsort(idx, kind="stable")
    idx, yy = idx[order], yy[order]
    edges = np.searchsorted(idx, np.arange(n + 1))
    for b in range(n):
        if edges[b + 1] > edges[b]:
            out[b] = np.median(yy[edges[b]:edges[b + 1]])
    good = np.isfinite(out)
    if 0 < good.sum() < n:
        xs = np.arange(n)
        out[~good] = np.interp(xs[~good], xs[good], out[good])
    return out


def make_views(flat: LightCurve, eph: Ephemeris, sec_phase: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (global (201,), local (5, 61)) float32 arrays."""
    t, f = flat.time, flat.flux - 1.0
    P, D = eph.period, eph.duration
    ph = phase_days(t, eph)
    oot = np.abs(ph) > D
    f = f - (np.median(f[oot]) if oot.sum() > 10 else 0.0)

    g = binned(ph / P, f, -0.5, 0.5, N_GLOBAL)
    w = LOCAL_HALF_WIDTH * D
    w = min(w, 0.5 * P)
    loc = binned(ph, f, -w, w, N_LOCAL)
    ep = epoch_index(t, eph)
    odd = binned(ph[ep % 2 == 1], f[ep % 2 == 1], -w, w, N_LOCAL)
    even = binned(ph[ep % 2 == 0], f[ep % 2 == 0], -w, w, N_LOCAL)
    if sec_phase is not None and np.isfinite(sec_phase):
        ps = ((ph - sec_phase * P + 0.5 * P) % P) - 0.5 * P
        sec = binned(ps, f, -w, w, N_LOCAL)
    else:
        sec = np.zeros(N_LOCAL)

    cen = np.zeros(N_LOCAL)
    if flat.cx is not None and np.isfinite(flat.cx).sum() > 100:
        r2 = np.zeros_like(t)
        okc = np.isfinite(flat.cx) & np.isfinite(flat.cy)
        for c in (flat.cx, flat.cy):
            cn = c - np.nanmedian(c[oot & okc])
            try:
                _, tr = flatten(t[oot & okc], cn[oot & okc] + 10.0, method="biweight",
                                window_length=max(0.75, 3 * D), return_trend=True)
                gd = np.isfinite(tr)
                cn = cn + 10.0 - np.interp(t, t[oot & okc][gd], tr[gd])
            except Exception:
                pass
            s = _robust_std(cn[oot & okc])
            r2 = r2 + (cn / s) ** 2 if s > 0 else r2
        cen = binned(ph[okc], np.sqrt(r2[okc]), -w, w, N_LOCAL) - np.sqrt(2) * 0.886  # mean of chi(2) ~ 1.25
        cen = np.clip(np.nan_to_num(cen), -5, 20) / 5.0

    scale = -np.nanmin(loc) if np.isfinite(loc).any() and np.nanmin(loc) < 0 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    gscale = -np.nanmin(g) if np.isfinite(g).any() and np.nanmin(g) < 0 else scale
    chans = [loc / scale, odd / scale, even / scale, sec / scale]
    local = np.stack([np.clip(np.nan_to_num(c), -3, 3) for c in chans] + [cen]).astype(np.float32)
    glob = np.clip(np.nan_to_num(g / gscale), -3, 3).astype(np.float32)
    return glob, local


def refine_ephemeris(raw: LightCurve, eph: Ephemeris, sig_t0: float, sig_p: float,
                     n_p: int = 31, max_shift_frac: float = 0.25) -> Ephemeris:
    """Catalogue ephemerides drift when propagated years ahead. Re-centre with a
    grid over period and epoch that maximises the box depth SNR."""
    w = max(0.75, 3.0 * eph.duration)
    flat = flatten(raw.time, raw.flux, method="biweight", window_length=w, edge_cutoff=0.0,
                   break_tolerance=0.5)
    ok = np.isfinite(flat)
    t, f = raw.time[ok], flat[ok] - 1.0
    if len(t) < 100:
        return eph
    tmid = 0.5 * (t.min() + t.max())
    n_mid = np.round((tmid - eph.t0) / eph.period)
    t_ref = eph.t0 + n_mid * eph.period  # epoch near the data
    span = np.sqrt(sig_t0**2 + (n_mid * sig_p) ** 2) if np.isfinite(sig_t0 + sig_p) else eph.duration
    span = float(np.clip(3 * span, 0.5 * eph.duration, max_shift_frac * eph.period))
    base_n = (t.max() - t.min()) / eph.period
    dp = float(np.clip(3 * sig_p, 0, 0.1 * eph.duration / max(base_n, 1))) if np.isfinite(sig_p) else 0.0
    periods = eph.period + np.linspace(-dp, dp, n_p) if dp > 0.02 * eph.duration / max(base_n, 1) else np.array([eph.period])
    shifts = np.arange(-span, span + 1e-9, max(eph.duration / 8, 2 * span / 160))
    sig = 1.4826 * np.median(np.abs(f - np.median(f)))
    best = (-np.inf, eph.period, t_ref)
    for P in periods:
        ph0 = ((t - t_ref + 0.5 * P) % P) - 0.5 * P
        o = np.argsort(ph0)
        ps, cs = ph0[o], np.concatenate([[0.0], np.cumsum(f[o])])
        lo = np.searchsorted(ps, shifts - 0.5 * eph.duration)
        hi = np.searchsorted(ps, shifts + 0.5 * eph.duration)
        n = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = -(cs[hi] - cs[lo]) / np.sqrt(n) / sig
        snr[n < 3] = -np.inf
        j = int(np.argmax(snr))
        if snr[j] > best[0]:
            best = (snr[j], P, t_ref + shifts[j])
    return Ephemeris(best[1], best[2], eph.duration, eph.depth)
