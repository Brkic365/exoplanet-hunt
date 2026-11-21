"""The four panels the portfolio walkthrough animates: raw, detrended,
periodogram, folded transit."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transitleastsquares import transitleastsquares

from .detrend import detrend
from .ingest import LightCurve


def stage_figure(lc: LightCurve, out: Path, title: str = "", window_length: float = 0.75) -> dict:
    flat, trend = detrend(lc, window_length=window_length)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = transitleastsquares(flat.time, flat.flux, flat.flux_err).power(
            period_min=0.5, period_max=(flat.time.ptp() if hasattr(flat.time, "ptp") else np.ptp(flat.time)) / 2,
            oversampling_factor=3, duration_grid_step=1.1, use_threads=4, show_progress_bar=False)

    fig, ax = plt.subplots(4, 1, figsize=(9, 11))
    ax[0].plot(lc.time, lc.flux, ".", ms=1.5, color="0.4")
    ax[0].plot(lc.time, trend, "-", lw=1, color="C1")
    ax[0].set(title="1. raw SAP flux + biweight trend", xlabel="BTJD", ylabel="rel. flux")
    ax[1].plot(flat.time, flat.flux, ".", ms=1.5, color="0.2")
    ax[1].set(title="2. detrended", xlabel="BTJD", ylabel="rel. flux")
    ax[2].plot(r.periods, r.power, lw=0.7, color="k")
    ax[2].axvline(r.period, color="C3", alpha=0.4, lw=3)
    ax[2].set(title=f"3. TLS periodogram  (P = {r.period:.4f} d, SDE = {r.SDE:.1f})", xlabel="period [d]",
              ylabel="SDE", xscale="log")
    ph = ((flat.time - r.T0 + 0.5 * r.period) % r.period - 0.5 * r.period) * 24
    win = np.abs(ph) < 3 * r.duration * 24
    ax[3].plot(ph[win], flat.flux[win], ".", ms=2, color="0.3")
    ax[3].plot((np.asarray(r.model_folded_phase) - 0.5) * r.period * 24, r.model_folded_model, color="C3")
    ax[3].set(title=f"4. folded transit  (depth = {(1 - r.depth) * 1e6:.0f} ppm)", xlabel="hours from mid-transit",
              ylabel="rel. flux", xlim=(-3 * r.duration * 24, 3 * r.duration * 24))
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return {"period": r.period, "sde": r.SDE}
