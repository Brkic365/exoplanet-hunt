"""Match detections to known TOIs by period (incl. simple harmonics) and epoch."""

from __future__ import annotations

import numpy as np
import pandas as pd

# 3/2 and 2/3 appear when data gaps hide every other transit
HARMONICS = (1.0, 0.5, 2.0, 1 / 3, 3.0, 1.5, 2 / 3)


RANK = {"match": 0, "alias": 1, "period_only": 2}


def match_one(det_period, det_t0, toi_period, toi_t0, toi_duration_d, period_tol=0.01,
              toi_period_err=0.0, toi_epoch_err=0.0):
    """Return ('match' | 'alias' | None, ratio).

    'match': period within `period_tol` (fractional) and a predicted TOI transit
    lands within one TOI duration of a detected one.
    'alias': same, but the detected period is a 1/2, 2, 1/3, 3, 3/2 or 2/3 multiple.
    'period_only': period matches exactly but the epoch doesn't (stale
    ephemeris or TTVs). Not counted as recovered.
    """
    if not (np.isfinite(toi_period) and toi_period > 0):
        return None, np.nan
    for h in HARMONICS:
        if abs(det_period / toi_period - h) / h < period_tol:
            # Propagate the (precise, long-baseline) TOI ephemeris to our epoch and
            # compare modulo the shorter of the two periods. Detected transits
            # are a subset of TOI transits (h >= 1) or vice versa (h < 1).
            # TLS's T0 can be an extrapolated epoch inside a data gap, so test the
            # first few detected epochs and accept if any lands on the ephemeris.
            p = toi_period * min(h, 1.0)
            ks = np.arange(4)
            dphis = ((det_t0 + ks * det_period - toi_t0 + 0.5 * p) % p) - 0.5 * p
            dphi = dphis[np.argmin(np.abs(dphis))]
            n_cyc = abs(det_t0 - toi_t0) / toi_period
            sig = np.hypot(n_cyc * np.nan_to_num(toi_period_err), np.nan_to_num(toi_epoch_err))
            dur = toi_duration_d if np.isfinite(toi_duration_d) else 0.1
            tol = max(dur, 3 * sig, 0.05)
            if abs(dphi) < tol:
                return ("match" if h == 1.0 else "alias"), h
            if h == 1.0:
                return "period_only", h
    return None, np.nan


def crossmatch(detections: pd.DataFrame, tois: pd.DataFrame, min_sde: float = 9.0) -> pd.DataFrame:
    """One row per TOI with the best-matching detection on that star, if any."""
    rows = []
    for _, t in tois.iterrows():
        d = detections[(detections["tic"] == t["tic"]) & (detections["sde"] >= min_sde)]
        best = None
        for _, row in d.iterrows():
            kind, h = match_one(row["period"], row["t0"], t["period"], t["epoch_btjd"], t["duration_hr"] / 24,
                                toi_period_err=t.get("period_err", 0.0), toi_epoch_err=t.get("epoch_err", 0.0))
            if kind and (best is None or RANK[kind] < RANK[best[0]]):
                best = (kind, h, row)
        out = {
            "toi": t["toi"],
            "tic": t["tic"],
            "disposition": t["disposition"],
            "toi_period": t["period"],
            "toi_depth_ppm": t["depth_ppm"],
            "toi_radius_re": t.get("radius_re"),
            "tmag": t["tmag"],
            "n_transits_expected": t.get("n_transits_expected", np.nan),
            "status": best[0] if best else "missed",
        }
        if best:
            r = best[2]
            out.update(
                det_period=r["period"], det_sde=r["sde"], det_snr=r["snr"], det_rank=r["rank"], harmonic=best[1]
            )
        rows.append(out)
    return pd.DataFrame(rows)


def recall_summary(cm: pd.DataFrame, min_transits: int = 2) -> pd.DataFrame:
    """Recall per disposition, over TOIs with >= `min_transits` transits in the
    data (fewer can't be found by a periodic search; that's the single-event
    detector's job)."""
    if "n_transits_expected" in cm:
        cm = cm[cm["n_transits_expected"] >= min_transits]
    g = cm.groupby("disposition")["status"]
    s = pd.DataFrame(
        {
            "n": g.size(),
            "match": g.apply(lambda x: (x == "match").sum()),
            "alias": g.apply(lambda x: (x == "alias").sum()),
            "period_only": g.apply(lambda x: (x == "period_only").sum()),
        }
    )
    s.loc["ALL"] = s.sum()
    s["recall"] = (s["match"] + s["alias"]) / s["n"]
    s["recall_exact"] = s["match"] / s["n"]
    return s
