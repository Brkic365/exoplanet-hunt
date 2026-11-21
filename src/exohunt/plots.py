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


def heatmap_figure(cells, out: Path, title: str = "Injection-recovery, sector 75") -> None:
    """Recovery fraction over (period, radius) cells from `injection.heatmap`."""
    xs = np.unique(np.r_[cells["x_lo"], cells["x_hi"]])
    ys = np.unique(np.r_[cells["y_lo"], cells["y_hi"]])
    grid = np.full((len(ys) - 1, len(xs) - 1), np.nan)
    counts = np.zeros_like(grid, dtype=int)
    for c in cells.itertuples():
        i, j = np.searchsorted(ys, c.y_lo), np.searchsorted(xs, c.x_lo)
        grid[i, j], counts[i, j] = c.frac, c.n
    fig, ax = plt.subplots(figsize=(8, 6))
    m = ax.pcolormesh(xs, ys, grid, cmap="viridis", vmin=0, vmax=1, shading="flat")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if counts[i, j]:
                xc, yc = np.sqrt(xs[j] * xs[j + 1]), np.sqrt(ys[i] * ys[i + 1])
                ax.text(xc, yc, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="w" if grid[i, j] < 0.6 else "k")
    ax.set(xscale="log", yscale="log", xlabel="period [d]", ylabel="planet radius [R⊕]", title=title)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{y:g}" for y in ys])
    ax.minorticks_off()
    fig.colorbar(m, ax=ax, label="fraction recovered (SDE ≥ 9, period + epoch match)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def recall_curve_figure(curves: dict, out: Path, xlabel: str = "expected SNR", title: str = "") -> None:
    """`curves`: label -> DataFrame with index bins (pd.Interval) and a `recall` column."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, df in curves.items():
        x = [iv.left * 1.5 if iv.right > 1e3 else np.sqrt(iv.left * iv.right) if iv.left > 0 else iv.right / 2
             for iv in df.index]
        ax.plot(x, df["recall"], "o-", label=label)
    ax.set(xscale="log", xlabel=xlabel, ylabel="recall", ylim=(-0.02, 1.02), title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
