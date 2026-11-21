"""Transit Least Squares (Hippke & Heller 2019) as the CPU reference search."""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
from transitleastsquares import transitleastsquares

from ..ingest import LightCurve
from .base import Detection


def tls_search(
    lc: LightCurve,
    period_min: float = 0.25,
    use_stellar_params: bool = False,  # see README: finer short-P grid lowered SDE elsewhere
    period_max: float | None = None,
    oversampling_factor: int = 3,
    duration_grid_step: float = 1.1,
    threads: int = 1,
    min_transits: int = 2,
) -> Detection | None:
    baseline = float(lc.time.max() - lc.time.min())
    if period_max is None:
        period_max = baseline / min_transits
    if period_max <= period_min:
        return None
    R_star, M_star = stellar_params(lc.meta) if use_stellar_params else (1.0, 1.0)
    model = transitleastsquares(lc.time, lc.flux, lc.flux_err)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        r = model.power(
            period_min=period_min,
            period_max=period_max,
            oversampling_factor=oversampling_factor,
            R_star=R_star,
            M_star=M_star,
            R_star_min=min(0.13, R_star),
            R_star_max=max(3.5, R_star),
            M_star_min=min(0.1, M_star),
            M_star_max=max(1.0, M_star),
            duration_grid_step=duration_grid_step,
            use_threads=threads,
            show_progress_bar=False,
        )
    if not np.isfinite(r.period) or not np.isfinite(r.SDE):
        return None
    return Detection(
        tic=lc.tic,
        period=float(r.period),
        t0=float(r.T0),
        duration=float(r.duration),
        depth=float(1.0 - r.depth),
        sde=float(r.SDE),
        snr=float(r.snr),
        n_transits=int(r.distinct_transit_count),
        method="tls",
        extra={
            "odd_even_mismatch": float(r.odd_even_mismatch),
            "period_uncertainty": float(r.period_uncertainty),
            "rp_rs": float(r.rp_rs),
        },
    )


def stellar_params(meta: dict) -> tuple[float, float]:
    """TIC radius and mass from the QLP header; solar if missing. They only
    shape TLS's period/duration grids, so small stars get a shorter minimum
    period (down to `period_min`)."""

    def num(key):
        try:
            v = float(meta.get(key))
        except (TypeError, ValueError):
            return float("nan")
        return v

    r, m = num("RADIUS"), num("MASS")
    if not (np.isfinite(r) and 0.08 < r < 5):
        r = 1.0
    if not (np.isfinite(m) and 0.08 < m < 3):
        m = float(np.clip(r**1.25, 0.1, 2.0))
    return r, m
