"""Per-signal inputs for the classifier: views + scalar features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import LightCurve
from ..vetting import Ephemeris, vet_signal
from ..views import make_views

# "Plain BLS statistics": what any box search reports about a signal.
BLS_FEATURES = ["log_period", "log_duration_hr", "log_depth", "log_snr", "n_events"]
# Vetting metrics (vetting.py) on top of those.
VET_FEATURES = BLS_FEATURES + [
    "oddeven_sigma", "sec_snr", "sec_half_snr", "sec_rel_depth", "centroid_sigma", "log_centroid_offset",
    "shape_flank_ratio", "log_rp_re", "dur_ratio", "max_event_frac", "beta", "dump_frac", "tmag", "teff", "r_star",
]


def signal_record(raw: LightCurve, eph: Ephemeris, others: list[Ephemeris] | None = None) -> dict:
    m, flat = vet_signal(raw, eph, others)
    if not m.get("n_in"):
        return {}
    glob, local = make_views(flat, eph, m.get("sec_phase"))
    rec = dict(m)
    rec.update(period=eph.period, t0=eph.t0, duration=eph.duration, tmag=_f(raw.meta.get("TESSMAG")),
               global_view=glob, local_view=local)
    return rec


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def scalar_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Derived columns + NaN filled with training-free defaults; returns float32."""
    d = pd.DataFrame(index=df.index)
    lg = lambda x, lo: np.log10(np.clip(np.asarray(x, float), lo, None))
    d["log_period"] = lg(df["period"], 1e-2)
    d["log_duration_hr"] = lg(df["duration"] * 24, 1e-2)
    d["log_depth"] = lg(df["depth"], 1e-6)
    d["log_snr"] = lg(df["snr"], 0.1)
    d["n_events"] = np.asarray(df["n_events"], float)
    d["oddeven_sigma"] = np.clip(df["oddeven_sigma"], 0, 50)
    d["sec_snr"] = np.clip(df["sec_snr"], 0, 50)
    d["sec_half_snr"] = np.clip(df["sec_half_snr"], -20, 50)
    d["sec_rel_depth"] = np.clip(df["sec_depth"] / np.clip(df["depth"], 1e-6, None), -2, 2)
    d["centroid_sigma"] = np.clip(df["centroid_sigma"], 0, 50)
    d["log_centroid_offset"] = lg(df["centroid_offset_px"], 1e-2)
    d["shape_flank_ratio"] = np.clip(df["shape_flank_ratio"], -1, 2)
    d["log_rp_re"] = lg(df["rp_re"], 0.3)
    d["dur_ratio"] = np.clip(df["dur_ratio"], 0, 5)
    d["max_event_frac"] = df["max_event_frac"]
    d["beta"] = np.clip(df["beta"], 1, 5)
    d["dump_frac"] = df["dump_frac"]
    d["tmag"] = df["tmag"]
    d["teff"] = np.clip(df["teff"], 2500, 12000) / 1000
    d["r_star"] = np.clip(df["r_star"], 0.1, 10)
    fill = {"log_rp_re": 1.0, "teff": 5.5, "r_star": 1.0, "tmag": 11.0, "log_centroid_offset": -1.0}
    d = d[cols].replace([np.inf, -np.inf], np.nan)
    for c in cols:
        d[c] = d[c].fillna(fill.get(c, 0.0))
    return d.to_numpy(np.float32)
