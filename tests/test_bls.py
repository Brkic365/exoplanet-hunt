"""BLS: CPU path finds an injected transit; the CUDA kernel (run in numba's
CUDA simulator when no GPU is present) returns the same periodogram."""

import os
import subprocess
import sys
import textwrap

import numpy as np

from exohunt.detrend import detrend
from exohunt.search import iterative_search
from exohunt.search.bls import bls_search
from exohunt.crossmatch import match_one
from test_synthetic import synthetic


def test_cpu_bls_recovers_transit():
    lc, P, T0, dur = synthetic(depth=1.5e-3)
    flat, _ = detrend(lc)
    d = iterative_search(flat, bls_search, max_signals=1)[0]
    assert d.sde > 9
    assert abs(d.period - P) / P < 0.01
    assert match_one(d.period, d.t0, P, T0, dur)[0] == "match"
    assert 0.5 < d.depth / 1.5e-3 < 1.2


def test_cuda_kernel_matches_cpu():
    # The simulator flag must be set before numba.cuda is imported, so run in a
    # subprocess. On a machine with a GPU, set EXOHUNT_REAL_CUDA=1 to test the device.
    code = textwrap.dedent(
        """
        import numpy as np, sys
        sys.path.insert(0, %r)
        from test_synthetic import synthetic
        from exohunt.ingest import LightCurve
        from exohunt.search import bls as B
        lc, P, T0, dur = synthetic(depth=3e-3)
        lc = LightCurve(1, lc.time[::10], lc.flux[::10], lc.flux_err[::10], [1])
        lc2 = LightCurve(2, lc.time, 2 - lc.flux, lc.flux_err, [1])  # inverted: no dip at P
        g = B.make_grid(27.0, 3.15, 3.27)
        pc, jc, kc, _ = B.bls_periodogram_cpu(lc, g)
        pg, jg, kg, _ = B.bls_periodogram_cuda([lc, lc2], g, threads=16)
        assert np.allclose(pg[0], pc, rtol=1e-4, atol=1e-6 * pc.max()), np.max(np.abs(pg[0] - pc))
        assert np.argmax(pg[0]) == np.argmax(pc)
        assert (kg[0] == kc).mean() > 0.9
        p2, _, _, _ = B.bls_periodogram_cpu(lc2, g)
        assert np.allclose(pg[1], p2, rtol=1e-4, atol=1e-6 * max(p2.max(), 1e-30))
        print("ok", len(g.periods))
        """
        % os.path.dirname(__file__)
    )
    env = dict(os.environ)
    if not env.get("EXOHUNT_REAL_CUDA"):
        env["NUMBA_ENABLE_CUDASIM"] = "1"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ok" in r.stdout
