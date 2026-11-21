"""Full pipeline on one star: detrend -> periodic BLS -> single-event search
-> vetting -> views. Classifier scoring happens afterwards in one batch."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from . import ingest
from .classify import signal_record
from .detrend import detrend
from .search import iterative_search
from .search.bls import bls_search
from .search.single import single_event_search
from .vetting import Ephemeris

BLS_SDE_MIN = 6.0  # same false-alarm rate as TLS SDE 9 (week 2)
SES_MIN = 9.0  # 7.6% of quiet stars per sector; see results/s0075/injections/single_events.csv


def is_harmonic(p1: float, p2: float, tol: float = 0.01) -> bool:
    r = p1 / p2
    return any(abs(r / h - 1) < tol for h in (1, 2, 0.5, 3, 1 / 3, 1.5, 2 / 3))


def hunt_star(path: Path, max_signals: int = 2, ses_min: float = SES_MIN) -> list[dict]:
    warnings.filterwarnings("ignore")
    raw = ingest.read_qlp(path)
    if len(raw) < 500:
        return []
    flat, _ = detrend(raw, window_length=0.75)
    dets = iterative_search(flat, bls_search, max_signals=max_signals, min_sde=BLS_SDE_MIN)
    periodic = [d for d in dets if d.sde >= BLS_SDE_MIN]
    singles = [d for d in single_event_search(raw, mask=periodic, flat_short=flat, max_events=2) if d.sde >= ses_min]
    out = []
    for d in periodic + singles:
        P = d.period if np.isfinite(d.period) else 1e4
        eph = Ephemeris(P, d.t0, d.duration, d.depth)
        # mask the star's other signals, except harmonics of this one: an eclipsing
        # binary's secondary is often found as a second signal at the same period
        others = [Ephemeris(o.period, o.t0, o.duration) for o in periodic
                  if o is not d and not (np.isfinite(d.period) and is_harmonic(o.period, d.period))]
        try:
            rec = signal_record(raw, eph, others)
        except Exception:
            continue
        if not rec:
            continue
        rec.update(tic=raw.tic, method=d.method, search_rank=d.rank, search_sde=d.sde, search_snr=d.snr,
                   search_depth=d.depth, n_transits_search=d.n_transits,
                   **{f"x_{k}": v for k, v in d.extra.items()})
        if not np.isfinite(d.period):
            rec["period"] = np.nan
        out.append(rec)
    return out
