"""Run ingest -> detrend -> search for one sector and write results."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import ingest
from .detrend import detrend
from .search import SEARCHERS, iterative_search


def process_star(path: Path, searcher: str = "tls", max_signals: int = 2, window_length: float = 0.75,
                 flux_column: str = "SAP_FLUX", **search_kw) -> list[dict]:
    t0 = time.time()
    lc = ingest.read_qlp(path, flux_column=flux_column)
    if len(lc) < 200:
        return []
    flat, _ = detrend(lc, window_length=window_length)
    dets = iterative_search(flat, SEARCHERS[searcher], max_signals=max_signals, **search_kw)
    dt = time.time() - t0
    rows = []
    for d in dets:
        r = d.to_row()
        r.update(sector=lc.sectors[0], n_points=len(flat), runtime_s=dt)
        rows.append(r)
    return rows


def run_sector(paths: list[Path], out_csv: Path, workers: int | None = None, **kw) -> pd.DataFrame:
    """Search every light curve in `paths` in parallel; resumable via `out_csv`."""
    workers = workers or os.cpu_count() or 1
    done: set[int] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(prev["tic"].astype(int))
    todo = [p for p in paths if int(p.stem.split("_")[-1]) not in done]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = not out_csv.exists()
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(process_star, p, **kw): p for p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            p = futs[f]
            try:
                rows = f.result()
            except Exception as e:  # keep going; record the failure
                print(f"[{i}/{len(todo)}] {p.name}: FAILED {e!r}", flush=True)
                continue
            if not rows:
                rows = [{"tic": int(p.stem.split("_")[-1]), "sde": float("nan")}]
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=header, index=False)
            header = False
            best = rows[0]
            print(f"[{i}/{len(todo)}] TIC {best['tic']}: P={best.get('period', float('nan')):.4f} SDE={best['sde']:.1f}",
                  flush=True)
    return pd.read_csv(out_csv)
