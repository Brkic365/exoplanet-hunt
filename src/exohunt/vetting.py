"""Stage 4: automatic vetting of a transit signal.

Every check works on the light curve detrended *with the transits masked*
(so the trend never eats the dip being vetted) and returns numbers, not just
pass/fail, so the classifier can use them as features. Flags mark the usual
impostors:

* ``oddeven``   odd and even transits differ in depth -> eclipsing binary at 2x period
* ``secondary`` a significant dip elsewhere in phase -> eclipsing binary
* ``centroid``  the image centroid moves during transit and points off target
                -> light from a neighbouring star
* ``dump``      transits coincide with reaction-wheel momentum dumps -> systematics
* ``oneevent``  one event carries most of the signal -> outlier, not periodic
* ``toobig``    implied radius > 30 R_Earth for the TIC stellar radius -> stellar companion
* ``toolong``   duration far longer than any circular orbit around this star allows
* ``edge``      transits sit in the first day of a data segment (orbit-start ramps)
* ``noisy``     the dip sits in a stretch > 1.5x noisier than the rest (scattered light, bad orbit)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from wotan import flatten

from .ingest import LightCurve

R_SUN_RE = 109.1
G_CGS = 6.674e-8


@dataclass
class Ephemeris:
    period: float  # d
    t0: float  # BTJD
    duration: float  # d
    depth: float = float("nan")  # fractional, optional hint


def phase_days(t: np.ndarray, eph: Ephemeris) -> np.ndarray:
    """Time from nearest mid-transit, in days, in [-P/2, P/2)."""
    return ((t - eph.t0 + 0.5 * eph.period) % eph.period) - 0.5 * eph.period


def epoch_index(t: np.ndarray, eph: Ephemeris) -> np.ndarray:
    return np.round((t - eph.t0) / eph.period).astype(int)


def masked_detrend(lc: LightCurve, eph: Ephemeris, window_length: float | None = None,
                   mask_width: float = 1.5) -> LightCurve:
    """Biweight trend fitted on out-of-transit points only, interpolated across
    the transits. Window is at least 3x the transit duration."""
    w = max(0.75, 3.0 * eph.duration) if window_length is None else window_length
    oot = np.abs(phase_days(lc.time, eph)) > 0.5 * mask_width * eph.duration
    if oot.sum() < 50:
        oot = np.ones(len(lc), bool)
    _, trend = flatten(lc.time[oot], lc.flux[oot], method="biweight", window_length=w,
                       edge_cutoff=0.0, break_tolerance=0.5, return_trend=True)
    good = np.isfinite(trend)
    if good.sum() < 10:
        return lc.subset(np.zeros(len(lc), bool))
    full = np.interp(lc.time, lc.time[oot][good], trend[good])
    # do not extrapolate the trend across data gaps longer than the window
    tt = lc.time[oot][good]
    idx = np.clip(np.searchsorted(tt, lc.time), 1, len(tt) - 1)
    near = np.minimum(np.abs(lc.time - tt[idx - 1]), np.abs(tt[idx] - lc.time))
    ok = near < max(w, 2 * eph.duration)
    flat = lc.flux / full
    keep = ok & np.isfinite(flat)
    resid = flat[keep] - 1
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    keep[keep] = resid < 3 * mad  # clip upward outliers only
    return lc.subset(keep, flux=flat)


def _robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def red_noise_beta(t: np.ndarray, f: np.ndarray, timescale: float) -> float:
    """Ratio of the scatter of `timescale`-binned means to the white-noise
    expectation (>= 1). Inflates every error bar below."""
    if len(t) < 50 or timescale <= 0:
        return 1.0
    sig = _robust_std(f)
    bins = np.floor((t - t[0]) / timescale).astype(int)
    _, inv, cnt = np.unique(bins, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=f - np.median(f))
    use = cnt >= 3
    if use.sum() < 5:
        return 1.0
    means = sums[use] / cnt[use]
    expected = sig / np.sqrt(np.median(cnt[use]))
    return float(max(1.0, _robust_std(means) / expected))


def box_depth(f_in: np.ndarray, base: float, sigma: float, beta: float) -> tuple[float, float]:
    n = int(np.isfinite(f_in).sum())
    if n == 0:
        return float("nan"), float("nan")
    return float(base - np.nanmean(f_in)), float(beta * sigma / np.sqrt(n))


def implied_radius_re(depth: float, r_star: float) -> float:
    if not (np.isfinite(depth) and depth > 0 and np.isfinite(r_star) and r_star > 0):
        return float("nan")
    return float(np.sqrt(depth) * r_star * R_SUN_RE)


def max_duration_days(period: float, r_star: float, m_star: float, k: float = 0.1) -> float:
    """Central-transit duration of a circular orbit: P/pi * (1+k) * R*/a."""
    if not (np.isfinite(r_star) and r_star > 0 and np.isfinite(m_star) and m_star > 0):
        r_star, m_star = 1.0, 1.0
    rho = 1.41 * m_star / r_star**3  # g/cc
    a_rs = (G_CGS * rho * (period * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)
    return float(period / np.pi * np.arcsin(min(1.0, (1 + k) / a_rs)))


def star_params(meta: dict) -> tuple[float, float, float]:
    def num(k):
        try:
            return float(meta.get(k))
        except (TypeError, ValueError):
            return float("nan")

    return num("RADIUS"), num("MASS"), num("TEFF")


def local_noise_ratio(flat: LightCurve, eph: Ephemeris, window: float = 1.0) -> float:
    """Scatter within +-`window` d of each transit (transit excluded) over the
    light curve's overall scatter; the worst event counts. Large values mean the
    dip sits in a noisy stretch (scattered light, a bad orbit), where box
    statistics computed from the whole light curve overstate the SNR."""
    t, f = flat.time, flat.flux
    if len(t) < 100:
        return float("nan")
    ph = phase_days(t, eph)
    oot = np.abs(ph) > eph.duration
    glob = _robust_std(f[oot])
    ratios = []
    for n in np.unique(epoch_index(t[np.abs(ph) < 0.5 * eph.duration], eph)):
        tc = eph.t0 + n * eph.period
        m = oot & (np.abs(t - tc) < window)
        if m.sum() > 20:
            ratios.append(_robust_std(f[m]) / glob)
    return float(max(ratios)) if ratios else float("nan")


def edge_fraction(t: np.ndarray, eph: Ephemeris, after_start: float = 1.0, before_end: float = 0.3,
                  gap: float = 0.5) -> float:
    """Fraction of transit events within `after_start` d after a data segment
    starts (or `before_end` d before it ends). TESS orbit starts carry thermal
    and scattered-light ramps that mimic lone dips in many stars at once."""
    t = np.sort(t)
    if len(t) < 2:
        return float("nan")
    br = np.where(np.diff(t) > gap)[0]
    starts = np.concatenate([[t[0]], t[br + 1]])
    ends = np.concatenate([t[br], [t[-1]]])
    tcs = eph.t0 + eph.period * np.arange(np.ceil((t[0] - eph.t0) / eph.period - 0.5),
                                          np.floor((t[-1] - eph.t0) / eph.period + 0.5) + 1)
    tcs = [x for x in tcs if np.any((x >= starts) & (x <= ends))]
    if not tcs:
        return float("nan")
    hit = [np.any(((x - starts) >= 0) & ((x - starts) < after_start)) or np.any(((ends - x) >= 0) & ((ends - x) < before_end))
           for x in tcs]
    return float(np.mean(hit))


def vet(flat: LightCurve, eph: Ephemeris) -> dict:
    """All checks on an already masked-detrended light curve."""
    t, f = flat.time, flat.flux
    out: dict = {}
    if len(t) < 100:
        return {"vet_ok": False, "flags": "nodata"}
    P, D = eph.period, eph.duration
    ph = phase_days(t, eph)
    ep = epoch_index(t, eph)
    intr = np.abs(ph) < 0.5 * D
    near = (np.abs(ph) > 1.0 * D) & (np.abs(ph) < 3.0 * D)
    oot = np.abs(ph) > 1.0 * D
    sigma = _robust_std(f[oot])
    beta = red_noise_beta(t[oot], f[oot], D)
    base = float(np.median(f[near])) if near.sum() > 10 else float(np.median(f[oot]))
    depth, derr = box_depth(f[intr], base, sigma, beta)
    out.update(depth=depth, depth_err=derr, snr=depth / derr if derr > 0 else np.nan, sigma=sigma, beta=beta,
               n_in=int(intr.sum()))

    # per-event depths and coverage
    ev_ids = np.unique(ep)
    tmin, tmax = t.min(), t.max()
    all_ev = np.arange(np.ceil((tmin - eph.t0) / P - 0.5), np.floor((tmax - eph.t0) / P + 0.5) + 1).astype(int)
    cad = float(np.median(np.diff(t)))
    ev_depth, ev_snr, ev_time = [], [], []
    for n in all_ev:
        m = intr & (ep == n)
        cover = m.sum() * cad / D
        if cover < 0.5:
            continue
        mn = near & (ep == n)
        b = float(np.median(f[mn])) if mn.sum() > 5 else base
        d, e = box_depth(f[m], b, sigma, beta)
        ev_depth.append(d)
        ev_snr.append(d / e)
        ev_time.append(eph.t0 + n * P)
    ev_depth, ev_snr, ev_time = map(np.asarray, (ev_depth, ev_snr, ev_time))
    n_ev = len(ev_depth)
    out["n_events"] = n_ev
    if n_ev:
        pos = np.clip(ev_snr, 0, None) ** 2
        out["max_event_frac"] = float(pos.max() / pos.sum()) if pos.sum() > 0 else 1.0
        if n_ev > 1:
            w = 1 / (ev_depth / ev_snr) ** 2
            mu = np.sum(w * ev_depth) / np.sum(w)
            out["event_chi2_dof"] = float(np.sum(w * (ev_depth - mu) ** 2) / (n_ev - 1))
        else:
            out["event_chi2_dof"] = np.nan
    else:
        out["max_event_frac"], out["event_chi2_dof"] = 1.0, np.nan

    # odd / even
    odd = intr & (ep % 2 == 1)
    even = intr & (ep % 2 == 0)
    d_o, e_o = box_depth(f[odd], base, sigma, beta)
    d_e, e_e = box_depth(f[even], base, sigma, beta)
    oe = abs(d_o - d_e) / np.hypot(e_o, e_e) if np.isfinite(d_o + d_e) else np.nan
    out.update(depth_odd=d_o, depth_even=d_e, oddeven_sigma=float(oe))

    # secondary eclipse: best box of the same duration away from the primary
    centers = np.arange(-0.5 * P, 0.5 * P, D / 3)
    centers = centers[np.abs(centers) > 1.5 * D]
    sec_best, sec_phase, sec_depth = 0.0, np.nan, np.nan
    order = np.argsort(ph)
    phs, fs = ph[order], f[order]
    for c in centers:
        lo, hi = np.searchsorted(phs, [c - 0.5 * D, c + 0.5 * D])
        if hi - lo < 3:
            continue
        d, e = box_depth(fs[lo:hi], base, sigma, beta)
        if d / e > sec_best:
            sec_best, sec_phase, sec_depth = d / e, c / P, d
    lo, hi = np.searchsorted(phs, [0.5 * P - 0.5 * D, 0.5 * P])
    lo2, hi2 = np.searchsorted(phs, [-0.5 * P, -0.5 * P + 0.5 * D])
    half = np.concatenate([fs[lo:hi], fs[lo2:hi2]])
    dh, eh = box_depth(half, base, sigma, beta)
    out.update(sec_snr=float(sec_best), sec_phase=float(sec_phase), sec_depth=float(sec_depth),
               sec_half_snr=float(dh / eh) if eh > 0 else np.nan, sec_half_depth=dh)
    out["n_sec_trials"] = int(len(centers))

    # transit shape: depth of the inner half vs the outer flanks
    inner = np.abs(ph) < 0.25 * D
    flank = (np.abs(ph) >= 0.25 * D) & (np.abs(ph) < 0.5 * D)
    di, _ = box_depth(f[inner], base, sigma, beta)
    dfl, _ = box_depth(f[flank], base, sigma, beta)
    out["shape_flank_ratio"] = float(dfl / di) if di > 0 else np.nan

    # centroid motion in transit (pixels), against nearby out-of-transit cadences
    cs, cpix, coff = np.nan, np.nan, np.nan
    if flat.cx is not None and np.isfinite(flat.cx).sum() > 100:
        shifts, errs = [], []
        for c in (flat.cx, flat.cy):
            ok = np.isfinite(c)
            cn = c - np.nanmedian(c)
            # remove slow pointing drift with the same masked biweight
            try:
                _, tr = flatten(t[ok & oot], cn[ok & oot] + 10.0, method="biweight",
                                window_length=max(0.75, 3 * D), return_trend=True)
                good = np.isfinite(tr)
                cn = cn + 10.0 - np.interp(t, t[ok & oot][good], tr[good])
            except Exception:
                pass
            s = _robust_std(cn[oot & ok])
            b = red_noise_beta(t[oot & ok], cn[oot & ok], D)
            m_in = intr & ok
            m_out = near & ok
            if m_in.sum() < 3 or m_out.sum() < 3:
                shifts.append(np.nan)
                errs.append(np.nan)
                continue
            shifts.append(np.nanmean(cn[m_in]) - np.nanmedian(cn[m_out]))
            errs.append(b * s * np.sqrt(1 / m_in.sum() + 1 / m_out.sum()))
        shifts, errs = np.asarray(shifts), np.asarray(errs)
        if np.isfinite(shifts).all():
            cs = float(np.sqrt(np.sum((shifts / errs) ** 2)))
            cpix = float(np.hypot(*shifts))
            coff = float(cpix / depth) if depth > 0 else np.nan  # ~ source offset from photocentre, px
    out.update(centroid_sigma=cs, centroid_shift_px=cpix, centroid_offset_px=coff)

    # momentum dumps
    dumps = np.asarray(flat.meta.get("dump_times", []), float)
    if n_ev and len(dumps):
        hit = np.array([np.any(np.abs(dumps - tc) < max(D, 0.1)) for tc in ev_time])
        out["dump_frac"] = float(hit.mean())
    else:
        out["dump_frac"] = 0.0

    # physical plausibility
    r_star, m_star, teff = star_params(flat.meta)
    out["rp_re"] = implied_radius_re(depth, r_star)
    out["dur_ratio"] = float(D / max_duration_days(P, r_star, m_star, k=np.sqrt(max(depth, 0))))
    out.update(r_star=r_star, m_star=m_star, teff=teff)

    out["local_noise_ratio"] = local_noise_ratio(flat, eph)
    out["edge_frac"] = edge_fraction(t, eph)

    flags = []
    if np.isfinite(out["local_noise_ratio"]) and out["local_noise_ratio"] > 1.5:
        flags.append("noisy")
    if np.isfinite(out["edge_frac"]) and (out["edge_frac"] > 0.5 or (n_ev <= 1 and out["edge_frac"] > 0)):
        flags.append("edge")
    if np.isfinite(oe) and oe > 3:
        flags.append("oddeven")
    sec_thresh = max(5.0, np.sqrt(2 * np.log(max(len(centers), 2))) + 2.0)  # look-elsewhere
    if (P < (tmax - tmin) and sec_best > sec_thresh and sec_depth > 0.1 * max(depth, 0)) or (np.isfinite(dh / eh) and dh / eh > 3 and dh > 0.1 * max(depth, 0)):
        flags.append("secondary")
    if np.isfinite(cs) and cs > 4 and np.isfinite(coff) and coff > 1.0:
        flags.append("centroid")
    if out["dump_frac"] > 0.5 or (n_ev <= 2 and out["dump_frac"] > 0):
        flags.append("dump")
    if n_ev >= 3 and out["max_event_frac"] > 0.7:
        flags.append("oneevent")
    if np.isfinite(out["rp_re"]) and out["rp_re"] > 30:
        flags.append("toobig")
    if np.isfinite(out["dur_ratio"]) and out["dur_ratio"] > 2.0:
        flags.append("toolong")
    out["flags"] = ",".join(flags)
    out["vet_ok"] = not flags
    return out


def mask_others(lc: LightCurve, others: list[Ephemeris] | None, width: float = 1.5) -> LightCurve:
    """Drop the transits of the star's other signals (multi-planet systems), so
    they do not pose as secondary eclipses or odd/even differences."""
    if not others:
        return lc
    keep = np.ones(len(lc), bool)
    for o in others:
        keep &= np.abs(phase_days(lc.time, o)) > 0.5 * width * o.duration
    return lc.subset(keep)


def vet_signal(raw: LightCurve, eph: Ephemeris, others: list[Ephemeris] | None = None) -> tuple[dict, LightCurve]:
    """Masked detrend + all checks. Returns (metrics, detrended light curve)."""
    flat = mask_others(masked_detrend(raw, eph), others)
    return vet(flat, eph), flat
