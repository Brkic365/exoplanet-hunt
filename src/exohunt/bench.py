"""Search-speed benchmark: stars per second for TLS (CPU reference), our BLS on
CPU (numba, all cores), and our BLS CUDA kernel (batched, one launch per batch).

All backends search the same detrended light curves over the same period grid
(TLS's Ofir grid, P = 0.6 d to baseline/2). Detrending and file reading are
timed separately and excluded from the search rate.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import ingest
from .detrend import detrend
from .search import bls as B
from .search.tls import tls_search


def load_flat(paths: list[Path], window_length: float = 0.75):
    lcs = []
    for p in paths:
        lc = ingest.read_qlp(p)
        if len(lc) >= 200:
            lcs.append(detrend(lc, window_length=window_length)[0])
    return lcs


def _tls_one(lc):
    d = tls_search(lc)
    return None if d is None else (d.period, d.sde)


def bench_tls(lcs, workers: int) -> dict:
    t = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_tls_one, lcs, chunksize=1))
    dt = time.perf_counter() - t
    return dict(backend="tls_cpu", workers=workers, n=len(lcs), seconds=dt, stars_per_s=len(lcs) / dt,
                periods=[r[0] if r else np.nan for r in res])


def bench_bls_cpu(lcs, grid) -> dict:
    B.bls_periodogram_cpu(lcs[0], grid)  # JIT warm-up
    t = time.perf_counter()
    dets = B.bls_batch(lcs, grid, backend="cpu")
    dt = time.perf_counter() - t
    return dict(backend="bls_cpu_numba", workers=os.cpu_count(), n=len(lcs), seconds=dt, stars_per_s=len(lcs) / dt,
                periods=[d.period if d else np.nan for d in dets])


def bench_bls_cuda(lcs, grid, batch: int = 256) -> dict:
    B.bls_periodogram_cuda(lcs[:2], grid)  # compile + warm-up
    t = time.perf_counter()
    dets = []
    for i in range(0, len(lcs), batch):
        dets += B.bls_batch(lcs[i:i + batch], grid, backend="cuda")
    dt = time.perf_counter() - t
    from numba import cuda

    name = cuda.get_current_device().name
    name = name.decode() if isinstance(name, bytes) else name
    return dict(backend="bls_cuda", workers=name, n=len(lcs), seconds=dt, stars_per_s=len(lcs) / dt,
                periods=[d.period if d else np.nan for d in dets])


def run(paths: list[Path], backends=("tls", "bls_cpu", "bls_cuda"), workers: int | None = None,
        out_csv: Path | None = None) -> pd.DataFrame:
    workers = workers or os.cpu_count() or 1
    t = time.perf_counter()
    lcs = load_flat(paths)
    t_prep = time.perf_counter() - t
    baseline = max(float(lc.time.max() - lc.time.min()) for lc in lcs)
    grid = B.make_grid(baseline)
    n_pts = int(np.median([len(lc) for lc in lcs]))
    print(f"{len(lcs)} stars, median {n_pts} points, {len(grid.periods)} trial periods; "
          f"read+detrend {t_prep:.1f} s ({len(lcs) / t_prep:.1f} stars/s, 1 core)")
    rows = []
    if "tls" in backends:
        rows.append(bench_tls(lcs, workers))
    if "bls_cpu" in backends:
        rows.append(bench_bls_cpu(lcs, grid))
    if "bls_cuda" in backends:
        if B.cuda_available():
            rows.append(bench_bls_cuda(lcs, grid))
        else:
            print("bls_cuda: no CUDA device found, skipped")
    df = pd.DataFrame(rows)
    if len(df) > 1:  # agreement of best periods with the first backend
        ref = np.array(df.iloc[0]["periods"])
        df["same_period_as_" + df.iloc[0]["backend"]] = [
            float(np.mean(np.abs(np.array(p) - ref) / ref < 0.01)) for p in df["periods"]]
    df = df.drop(columns=["periods"]).assign(n_points_median=n_pts, n_periods=len(grid.periods))
    print(df.to_string(index=False))
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
