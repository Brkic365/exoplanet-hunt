"""Stage 1: ingest QLP light curves from MAST.

QLP HLSP files live at a path derived from the 16-digit TIC id, so we build the
URL directly instead of querying MAST for every star. Cadences with a non-zero
QUALITY flag are dropped.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

from . import config


@dataclass
class LightCurve:
    tic: int
    time: np.ndarray  # BTJD
    flux: np.ndarray  # normalised to median 1
    flux_err: np.ndarray
    sectors: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    # Flux-weighted centroid (pixels), same length as time; None if unknown.
    cx: np.ndarray | None = None
    cy: np.ndarray | None = None

    def subset(self, keep: np.ndarray, flux: np.ndarray | None = None) -> "LightCurve":
        """Copy with only the cadences in boolean/int index `keep`."""
        f = self.flux if flux is None else flux
        return LightCurve(self.tic, self.time[keep], f[keep], self.flux_err[keep], list(self.sectors),
                          dict(self.meta),
                          None if self.cx is None else self.cx[keep],
                          None if self.cy is None else self.cy[keep])

    def __len__(self) -> int:
        return len(self.time)


MOMENTUM_DUMP = 32

QLP_VERSIONS = ("v02", "v01")  # later sectors are v02; try newest first


def qlp_url(tic: int, sector: int, version: str = "v01") -> str:
    p = f"{int(tic):016d}"
    return (
        f"{config.QLP_BASE}/s{sector:04d}/{p[0:4]}/{p[4:8]}/{p[8:12]}/{p[12:16]}/"
        f"hlsp_qlp_tess_ffi_s{sector:04d}-{p}_tess_{version}_llc.fits"
    )


def qlp_path(tic: int, sector: int) -> Path:
    return config.LC_DIR / f"s{sector:04d}" / f"qlp_s{sector:04d}_{int(tic)}.fits"


def download_qlp(tic: int, sector: int, overwrite: bool = False) -> Path | None:
    """Download one QLP light curve. Returns None if QLP has no file for it."""
    path = qlp_path(tic, sector)
    if path.exists() and not overwrite:
        return path
    for version in QLP_VERSIONS:
        r = requests.get(qlp_url(tic, sector, version), timeout=120)
        if r.status_code != 404:
            break
    else:
        return None
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return path


def download_many(tics, sector: int, workers: int = 8) -> dict[int, Path | None]:
    config.ensure_dirs()
    tics = [int(t) for t in tics]
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(lambda t: _safe_download(t, sector), tics))
    return dict(zip(tics, paths))


def _safe_download(tic: int, sector: int) -> Path | None:
    try:
        return download_qlp(tic, sector)
    except requests.RequestException:
        return None


def read_qlp(path: Path, flux_column: str = "SAP_FLUX") -> LightCurve:
    """Read a QLP file, drop flagged/non-finite cadences, normalise.

    `SAP_FLUX` is the raw aperture flux, so detrending is entirely ours.
    `SYS_RM_FLUX` (v02 only) has spacecraft systematics removed but no stellar
    detrending. `KSPSAP_FLUX` (v01) / `DET_FLUX` (v02) is QLP's own
    spline-detrended flux, useful as a reference.
    """
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        d = hdul[1].data
        t = np.asarray(d["TIME"], float)
        q = np.asarray(d["QUALITY"], int)
        names = d.columns.names
        err_col = "DET_FLUX_ERR" if "DET_FLUX_ERR" in names else "KSPSAP_FLUX_ERR"  # v02 vs v01
        if flux_column not in names:
            flux_column = {"KSPSAP_FLUX": "DET_FLUX", "DET_FLUX": "KSPSAP_FLUX"}.get(flux_column, flux_column)
        f = np.asarray(d[flux_column], float)
        e = np.asarray(d[err_col], float)
        tic = int(hdr.get("TICID") or path.stem.split("_")[-1])
        sector = int(hdr.get("SECTOR"))
        meta = {k: hdr.get(k) for k in ("TESSMAG", "TEFF", "RADIUS", "MASS", "LOGG", "RA_OBJ", "DEC_OBJ")}
        cx = np.asarray(d["SAP_X"], float) if "SAP_X" in names else np.full_like(t, np.nan)
        cy = np.asarray(d["SAP_Y"], float) if "SAP_Y" in names else np.full_like(t, np.nan)

    # Bit 6 (value 32) marks reaction-wheel momentum dumps; keep their times for
    # vetting before the flagged cadences are dropped.
    dumps = t[((q & MOMENTUM_DUMP) != 0) & np.isfinite(t)]
    meta["dump_times"] = _cluster_times(dumps)
    good = (q == 0) & np.isfinite(t) & np.isfinite(f) & (f > 0)
    t, f, e, cx, cy = t[good], f[good], e[good], cx[good], cy[good]
    med = np.nanmedian(f)
    f = f / med
    # KSPSAP errors are already relative to ~1; fall back to scatter if missing.
    if not np.isfinite(e).all() or np.nanmedian(e) <= 0:
        e = np.full_like(f, _mad_std(np.diff(f)) / np.sqrt(2))
    return LightCurve(tic, t, f, e, sectors=[sector], meta=meta, cx=cx, cy=cy)


def stitch(lcs: list[LightCurve]) -> LightCurve:
    """Concatenate per-sector light curves of one star (each already normalised)."""
    if not lcs:
        raise ValueError("nothing to stitch")
    order = np.argsort([lc.time[0] for lc in lcs])
    lcs = [lcs[i] for i in order]
    return LightCurve(
        lcs[0].tic,
        np.concatenate([lc.time for lc in lcs]),
        np.concatenate([lc.flux for lc in lcs]),
        np.concatenate([lc.flux_err for lc in lcs]),
        sectors=sorted(s for lc in lcs for s in lc.sectors),
        meta={**lcs[0].meta, "dump_times": [x for lc in lcs for x in lc.meta.get("dump_times", [])]},
        cx=None if any(lc.cx is None for lc in lcs) else np.concatenate([lc.cx - np.nanmedian(lc.cx) for lc in lcs]),
        cy=None if any(lc.cy is None for lc in lcs) else np.concatenate([lc.cy - np.nanmedian(lc.cy) for lc in lcs]),
    )


def _cluster_times(t: np.ndarray, gap: float = 0.1) -> list[float]:
    """Collapse runs of flagged cadences into one time per event."""
    t = np.sort(t)
    if len(t) == 0:
        return []
    groups = np.split(t, np.where(np.diff(t) > gap)[0] + 1)
    return [float(np.mean(g)) for g in groups]


def _mad_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))
