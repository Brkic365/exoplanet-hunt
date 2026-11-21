import numpy as np

from exohunt.ingest import LightCurve
from exohunt.search.single import single_event_search
from exohunt.vetting import Ephemeris, edge_fraction, vet


def _lc(seed=0, n=12000, span=27.0, noise=1e-3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, span, n)
    f = 1 + rng.normal(0, noise, n)
    return t, f


def _box(t, P, t0, D, depth, parity=None):
    ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    ep = np.round((t - t0) / P).astype(int)
    m = np.abs(ph) < 0.5 * D
    if parity is not None:
        m &= ep % 2 == parity
    return np.where(m, depth, 0.0)


def test_planet_passes_and_eb_flagged():
    t, f = _lc()
    eph = Ephemeris(3.0, 1.0, 0.1)
    planet = LightCurve(1, t, f - _box(t, 3.0, 1.0, 0.1, 3e-3), np.full_like(t, 1e-3), meta={"RADIUS": 1.0, "MASS": 1.0})
    m = vet(planet, eph)
    assert m["vet_ok"], m["flags"]
    assert abs(m["depth"] - 3e-3) < 4e-4
    # eclipsing binary found at half its period: alternating depths
    eb = f - _box(t, 3.0, 1.0, 0.1, 6e-3, parity=0) - _box(t, 3.0, 1.0, 0.1, 2e-3, parity=1)
    m = vet(LightCurve(1, t, eb, np.full_like(t, 1e-3)), eph)
    assert "oddeven" in m["flags"]
    # secondary eclipse at phase 0.5
    eb2 = f - _box(t, 3.0, 1.0, 0.1, 6e-3) - _box(t, 3.0, 2.5, 0.1, 2e-3)
    m = vet(LightCurve(1, t, eb2, np.full_like(t, 1e-3)), eph)
    assert "secondary" in m["flags"]


def test_edge_fraction():
    t = np.concatenate([np.linspace(0, 10, 3000), np.linspace(12, 22, 3000)])
    assert edge_fraction(t, Ephemeris(1e4, 12.3, 0.1)) == 1.0
    assert edge_fraction(t, Ephemeris(1e4, 16.0, 0.1)) == 0.0


def test_single_event_found():
    t, f = _lc(seed=3)
    f = f - _box(t, 1e4, 15.0, 0.25, 4e-3)
    lc = LightCurve(2, t, f, np.full_like(t, 1e-3), meta={"RADIUS": 0.5, "MASS": 0.5})
    dets = single_event_search(lc, min_ses=7)
    assert dets and abs(dets[0].t0 - 15.0) < 0.1
    assert abs(dets[0].depth - 4e-3) < 1e-3
