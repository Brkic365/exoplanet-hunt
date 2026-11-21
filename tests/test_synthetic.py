"""Inject a box transit into white noise and check every stage end to end."""

import numpy as np
import pandas as pd

from exohunt.crossmatch import crossmatch, match_one
from exohunt.detrend import detrend
from exohunt.ingest import LightCurve, qlp_url
from exohunt.search import iterative_search, tls_search


def synthetic(period=3.21, t0=1001.3, depth=2e-3, dur=0.12, noise=5e-4, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(1000.0, 1027.0, 200 / 86400)
    t = t[(t < 1013.0) | (t > 1014.0)]  # orbit gap
    f = 1 + 0.004 * np.sin(2 * np.pi * t / 4.7) + rng.normal(0, noise, t.size)  # spot-like variability
    ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    f[np.abs(ph) < dur / 2] -= depth
    return LightCurve(1, t, f, np.full_like(t, noise), [1]), period, t0, dur


def test_qlp_url():
    u = qlp_url(150428135, 13)
    assert u.endswith("s0013/0000/0001/5042/8135/hlsp_qlp_tess_ffi_s0013-0000000150428135_tess_v01_llc.fits")


def test_detrend_keeps_transit_and_tls_recovers_it():
    lc, P, T0, dur = synthetic()
    flat, _ = detrend(lc)
    assert np.std(flat.flux) < 1e-3  # variability removed
    dets = iterative_search(flat, tls_search, max_signals=1, threads=2)
    d = dets[0]
    assert d.sde > 9
    assert abs(d.period - P) / P < 0.01
    assert match_one(d.period, d.t0, P, T0, dur)[0] == "match"


def test_match_harmonics():
    assert match_one(2.0, 100.0, 1.0, 100.0, 0.1)[0] == "alias"
    assert match_one(1.0, 100.0, 1.0, 100.5, 0.1)[0] == "period_only"
    assert match_one(1.7, 100.0, 1.0, 100.0, 0.1)[0] is None
    assert match_one(1.0, 100.0, 1.0, 102.0, 0.1)[0] == "match"


def test_crossmatch_table():
    det = pd.DataFrame([{"tic": 5, "period": 3.0, "t0": 10.0, "sde": 12.0, "snr": 20.0, "rank": 1}])
    toi = pd.DataFrame([{"toi": 1.01, "tic": 5, "disposition": "CP", "period": 3.0, "epoch_btjd": 13.0,
                         "duration_hr": 2.0, "depth_ppm": 1000, "radius_re": 2.0, "tmag": 10.0}])
    cm = crossmatch(det, toi)
    assert cm.loc[0, "status"] == "match"


def test_period_only_and_three_halves():
    assert match_one(1.0, 100.3, 1.0, 100.0, 0.1)[0] == "period_only"
    assert match_one(1.5, 100.0, 1.0, 103.0, 0.1)[0] == "alias"


def test_t0_in_gap():
    # TOI-1148-like: detected P = 1.5 x true, T0 extrapolated half a cycle off
    assert match_one(8.32741, 3341.935861, 5.5514888, 3361.366521, 0.2)[0] == "alias"
