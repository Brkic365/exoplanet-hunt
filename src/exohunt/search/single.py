"""Single-event detector: lone transit-like dips that a period-folding search
cannot find (planets with P longer than the data, or with one transit landing
in good data).

For each trial duration (1 to 16 h) a box slides along the detrended light
curve in steps of a quarter duration. The statistic for a box is its depth
divided by the robust scatter of *all* box depths of that duration in the same
light curve, so correlated (red) noise on that timescale is priced in
automatically: a star whose variability makes 6 h dips everywhere gets a high
denominator.

Long transits are partly eaten by a short biweight window, so durations above
`long_split_hr` are searched on a light curve flattened with a longer window.
Each event found is then re-measured on a light curve detrended with the event
masked (vetting.masked_detrend), which restores the full depth.

The orbital period of a single transit is unknown; `period_from_duration`
gives the circular, central-transit period implied by the duration and the
TIC stellar density, a rough guide (true P is usually longer for b > 0).
"""

from __future__ import annotations

import numpy as np

from ..detrend import detrend
from ..ingest import LightCurve
from .base import Detection

DURATIONS_HR = (1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0, 16.0)
G_CGS = 6.674e-8


def box_scan(t: np.ndarray, f: np.ndarray, dur: float, step_frac: float = 0.25, min_fill: float = 0.6,
             edge_fill: float = 0.5):
    """Box depth at centres spaced dur*step_frac. Returns (centres, depth, n_in, sides_ok)."""
    cad = float(np.median(np.diff(t)))
    centres = np.arange(t[0] + 0.5 * dur, t[-1] - 0.5 * dur, dur * step_frac)
    if len(centres) == 0:
        return (np.array([]),) * 4
    cs = np.concatenate([[0.0], np.cumsum(f - 1.0)])
    lo = np.searchsorted(t, centres - 0.5 * dur)
    hi = np.searchsorted(t, centres + 0.5 * dur)
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = -(cs[hi] - cs[lo]) / n
    need = min_fill * dur / cad
    # data on both sides within one duration, so the dip is not a gap edge
    l1, l2 = np.searchsorted(t, centres - 1.5 * dur), lo
    r1, r2 = hi, np.searchsorted(t, centres + 1.5 * dur)
    sides = ((l2 - l1) >= edge_fill * dur / cad) & ((r2 - r1) >= edge_fill * dur / cad)
    ok = n >= need
    depth[~ok] = np.nan
    return centres, depth, n, sides & ok


def robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def scan(flat_short: LightCurve, flat_long: LightCurve, durations_hr=DURATIONS_HR, long_split_hr: float = 4.5):
    """Yield (dur_d, centres, ses, ses_up, depth, sides_ok) per trial duration."""
    for dh in durations_hr:
        lc = flat_long if dh > long_split_hr else flat_short
        d = dh / 24
        c, dep, n, sides = box_scan(lc.time, lc.flux, d)
        if len(c) == 0:
            continue
        s = robust_std(dep)
        if not np.isfinite(s) or s <= 0:
            continue
        yield d, c, dep / s, -dep / s, dep, sides


def single_event_search(raw: LightCurve, window_short: float = 0.75, window_long: float = 2.0,
                        max_events: int = 3, min_ses: float = 6.0, mask: list[Detection] | None = None,
                        flat_short: LightCurve | None = None) -> list[Detection]:
    """Strongest lone dips in a raw light curve (after masking `mask` signals)."""
    from ..vetting import Ephemeris, masked_detrend, phase_days

    if flat_short is None:
        flat_short, _ = detrend(raw, window_length=window_short)
    flat_long, _ = detrend(raw, window_length=window_long)
    if mask:
        flat_short, flat_long = _mask(flat_short, mask), _mask(flat_long, mask)
    if len(flat_short) < 200 or len(flat_long) < 200:
        return []

    cands = []  # (ses, centre, dur, ses_up_max, depth)
    for d, c, ses, ses_up, dep, sides in scan(flat_short, flat_long):
        ses = np.where(sides, ses, np.nan)
        up_max = float(np.nanmax(ses_up)) if np.isfinite(ses_up).any() else np.nan
        for i in np.argsort(np.nan_to_num(ses, nan=-np.inf))[::-1][: 3 * max_events]:
            if np.isfinite(ses[i]) and ses[i] >= min_ses:
                cands.append((float(ses[i]), float(c[i]), d, up_max, float(dep[i])))
    cands.sort(reverse=True)
    events: list[tuple] = []
    for cand in cands:
        if all(abs(cand[1] - e[1]) > max(cand[2], e[2]) * 1.5 for e in events):
            events.append(cand)
        if len(events) >= max_events:
            break

    out = []
    r_star, m_star = _star(raw.meta)
    for rank, (ses, tc, d, up_max, dep) in enumerate(events, 1):
        # re-measure on a light curve detrended with this event masked
        eph = Ephemeris(1e4, tc, d)
        fl = masked_detrend(raw, eph)
        ph = phase_days(fl.time, eph)
        inn = np.abs(ph) < 0.5 * d
        near = (np.abs(ph) > d) & (np.abs(ph) < 3 * d)
        if inn.sum() < 3 or near.sum() < 5:
            continue
        depth = float(np.median(fl.flux[near]) - np.mean(fl.flux[inn]))
        # refine centre and duration on the masked light curve
        tc2, d2, depth2 = _refine(fl, tc, d)
        out.append(Detection(
            tic=raw.tic, period=float("nan"), t0=tc2, duration=d2, depth=max(depth, depth2), sde=ses, snr=ses,
            n_transits=1, rank=rank, method="single",
            extra={"ses_up_max": up_max, "ses_ratio_up": up_max / ses if ses > 0 else np.nan,
                   "depth_scan": dep, "p_circ_est": period_from_duration(d2, r_star, m_star)},
        ))
    return out


def _mask(lc: LightCurve, dets: list[Detection]) -> LightCurve:
    keep = np.ones(len(lc), bool)
    for m in dets:
        ph = ((lc.time - m.t0 + 0.5 * m.period) % m.period) - 0.5 * m.period
        keep &= np.abs(ph) > m.duration
    return lc.subset(keep)


def _refine(fl: LightCurve, tc: float, d: float) -> tuple[float, float, float]:
    best = (-np.inf, tc, d, 0.0)
    sig = robust_std(fl.flux - 1) or 1e-3
    for dd in d * np.array([0.7, 0.85, 1.0, 1.2, 1.4]):
        for c in tc + d * np.linspace(-0.5, 0.5, 21):
            m = np.abs(fl.time - c) < 0.5 * dd
            n = m.sum()
            if n < 3:
                continue
            nr = (np.abs(fl.time - c) > dd) & (np.abs(fl.time - c) < 3 * dd)
            b = np.median(fl.flux[nr]) if nr.sum() > 5 else 1.0
            dep = b - np.mean(fl.flux[m])
            s = dep * np.sqrt(n) / sig
            if s > best[0]:
                best = (s, c, dd, dep)
    return float(best[1]), float(best[2]), float(best[3])


def _star(meta: dict) -> tuple[float, float]:
    try:
        r = float(meta.get("RADIUS"))
    except (TypeError, ValueError):
        r = np.nan
    try:
        m = float(meta.get("MASS"))
    except (TypeError, ValueError):
        m = np.nan
    if not (np.isfinite(r) and 0.08 < r < 10):
        r = 1.0
    if not (np.isfinite(m) and 0.08 < m < 5):
        m = float(np.clip(r**1.25, 0.1, 2.0))
    return r, m


def period_from_duration(dur_d: float, r_star: float, m_star: float) -> float:
    """Circular, b = 0 period implied by a duration: P = G rho pi^2 T^3 / 3."""
    rho = 1.41 * m_star / r_star**3
    T = dur_d * 86400
    return float(G_CGS * rho * np.pi**2 * T**3 / 3 / 86400)
