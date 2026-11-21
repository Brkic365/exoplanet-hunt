"""ExoFOP TOI catalogue: download, cache, and select targets per sector."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from . import config

KNOWN_PLANET = {"CP", "KP"}  # confirmed / known planet
CANDIDATE = {"PC", "APC"}
FALSE_POSITIVE = {"FP", "FA"}


def fetch_toi(path: Path | None = None, max_age_hours: float = 24.0) -> pd.DataFrame:
    """Download the ExoFOP TOI table (cached for `max_age_hours`)."""
    config.ensure_dirs()
    path = path or config.CATALOG_DIR / "toi.csv"
    if not path.exists() or (time.time() - path.stat().st_mtime) > max_age_hours * 3600:
        r = requests.get(config.TOI_URL, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return load_toi(path)


def load_toi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "TIC ID": "tic",
            "TOI": "toi",
            "TFOPWG Disposition": "disposition",
            "TESS Disposition": "tess_disposition",
            "Period (days)": "period",
            "Period (days) err": "period_err",
            "Epoch (BJD)": "epoch_bjd",
            "Epoch (BJD) err": "epoch_err",
            "Duration (hours)": "duration_hr",
            "Depth (ppm)": "depth_ppm",
            "Planet Radius (R_Earth)": "radius_re",
            "TESS Mag": "tmag",
            "Sectors": "sectors",
        }
    )
    df["epoch_btjd"] = df["epoch_bjd"] - config.BTJD_OFFSET
    df["sector_list"] = df["sectors"].astype(str).map(_parse_sectors)
    return df


def _parse_sectors(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x.isdigit()]


def tois_in_sector(
    toi: pd.DataFrame,
    sector: int,
    dispositions: set[str] | None = None,
    max_tmag: float = 13.5,
) -> pd.DataFrame:
    """TOIs observed in `sector` with the given TFOPWG dispositions."""
    m = toi["sector_list"].map(lambda ss: sector in ss)
    if dispositions is not None:
        m &= toi["disposition"].isin(dispositions)
    m &= toi["tmag"] <= max_tmag
    return toi[m].copy()


def expected_transits(time, period, epoch_btjd, duration_d, min_coverage=0.5):
    """Number of TOI transits whose centre +- duration/2 is at least
    `min_coverage` covered by cadences in `time` (assumes regular cadence)."""
    import numpy as np

    time = np.sort(np.asarray(time))
    if not (np.isfinite(period) and period > 0 and np.isfinite(epoch_btjd)):
        return 0
    if not np.isfinite(duration_d) or duration_d <= 0:
        duration_d = 0.1
    cad = np.median(np.diff(time))
    n0 = np.ceil((time[0] - epoch_btjd) / period)
    n1 = np.floor((time[-1] - epoch_btjd) / period)
    count = 0
    for n in np.arange(n0, n1 + 1):
        tc = epoch_btjd + n * period
        lo, hi = np.searchsorted(time, [tc - duration_d / 2, tc + duration_d / 2])
        if (hi - lo) * cad >= min_coverage * duration_d:
            count += 1
    return count


def qlp_field_stars(sector: int, ra_ranges=((150, 152), (30, 31), (260, 261)), path: Path | None = None) -> pd.DataFrame:
    """Every QLP target of `sector` inside a few RA strips, with TIC Tmag, radius,
    mass and Teff. Used as host stars for injection tests (cached as CSV)."""
    import numpy as np
    from astroquery.mast import Catalogs, Observations

    config.ensure_dirs()
    path = path or config.CATALOG_DIR / f"qlp_s{sector:04d}_field.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for ra in ra_ranges:
        t = Observations.query_criteria(provenance_name="QLP", sequence_number=sector, s_ra=list(ra))
        rows.append(pd.DataFrame({"tic": np.array(t["target_name"]).astype(int), "ra": np.array(t["s_ra"]),
                                  "dec": np.array(t["s_dec"])}))
    df = pd.concat(rows).drop_duplicates("tic")
    ids = list(df["tic"].astype(str))
    out = []
    for i in range(0, len(ids), 500):
        c = Catalogs.query_criteria(catalog="Tic", ID=ids[i:i + 500])
        out.append(pd.DataFrame({k: np.array(c[c_], float) for k, c_ in
                                 (("tmag", "Tmag"), ("rad", "rad"), ("mass", "mass"), ("teff", "Teff"))}
                                ).assign(tic=np.array(c["ID"]).astype(int)))
    df = df.merge(pd.concat(out), on="tic")
    df.to_csv(path, index=False)
    return df
