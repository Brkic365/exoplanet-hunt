"""Labelled TOIs for training the classifier: ExoFOP TFOPWG dispositions.

planet (1) = CP / KP, not planet (0) = FP / FA. PC / APC are unlabelled and
only scored. Each TOI gets the QLP light curves of up to `max_sectors` of its
most recent sectors; the host star decides the train/val/test split so the
same star never sits on both sides.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import catalog, ingest

LABEL = {"CP": 1, "KP": 1, "FP": 0, "FA": 0}


def labelled_tois(toi: pd.DataFrame, max_tmag: float = 13.5, exclude_sectors: set[int] = frozenset()) -> pd.DataFrame:
    t = toi[toi["disposition"].isin(LABEL) & (toi["tmag"] <= max_tmag)].copy()
    t = t[np.isfinite(t["period"]) & (t["period"] > 0) & np.isfinite(t["epoch_btjd"])
          & np.isfinite(t["duration_hr"]) & (t["duration_hr"] > 0)]
    t["label"] = t["disposition"].map(LABEL)
    t["split"] = t["tic"].map(split_of)
    t["sector_list"] = t["sector_list"].map(lambda ss: [s for s in ss if s not in exclude_sectors])
    return t[t["sector_list"].map(len) > 0]


def split_of(tic: int, val: float = 0.1, test: float = 0.2) -> str:
    """Deterministic star-level split from a hash of the TIC id."""
    u = int(hashlib.md5(str(int(tic)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else ("val" if u < test + val else "train")


def download_for(t: pd.DataFrame, max_sectors: int = 2, workers: int = 12) -> pd.DataFrame:
    """Fetch QLP files, newest sectors first; returns (tic, sector, path) rows."""
    stars = t.groupby("tic")["sector_list"].agg(lambda s: sorted({x for ss in s for x in ss}, reverse=True))

    def fetch(item):
        tic, sectors = item
        got = []
        for s in sectors:
            try:
                p = ingest.download_qlp(int(tic), int(s))
            except Exception:
                p = None
            if p:
                got.append((int(tic), int(s), str(p)))
            if len(got) >= max_sectors:
                break
        return got

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for g in ex.map(fetch, stars.items()) for r in g]
    return pd.DataFrame(rows, columns=["tic", "sector", "path"])
