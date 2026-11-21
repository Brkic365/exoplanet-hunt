"""Stage 2: detrend with a sliding biweight (wotan), then clip upward outliers.

Rule of thumb (Hippke et al. 2019): a window ~3x the longest transit duration
you care about preserves the dip. The default 0.75 d keeps transits up to ~6 h.
"""

from __future__ import annotations

import numpy as np
from wotan import flatten

from .ingest import LightCurve


def detrend(
    lc: LightCurve,
    window_length: float = 0.75,
    method: str = "biweight",
    edge_cutoff: float = 0.25,
    break_tolerance: float = 0.5,
    sigma_upper: float = 3.0,
) -> tuple[LightCurve, np.ndarray]:
    """Return (flattened light curve, trend). Upward outliers are clipped;
    downward points are kept because they may be transits."""
    flat, trend = flatten(
        lc.time,
        lc.flux,
        method=method,
        window_length=window_length,
        edge_cutoff=edge_cutoff,
        break_tolerance=break_tolerance,
        return_trend=True,
    )
    ok = np.isfinite(flat)
    resid = flat[ok] - np.median(flat[ok])
    mad = 1.4826 * np.median(np.abs(resid))
    ok[ok] = resid < sigma_upper * mad
    out = LightCurve(lc.tic, lc.time[ok], flat[ok], lc.flux_err[ok], list(lc.sectors), dict(lc.meta))
    return out, trend
