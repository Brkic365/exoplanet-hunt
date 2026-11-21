"""Paths and defaults. Override the data directory with EXOHUNT_DATA."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("EXOHUNT_DATA", Path.cwd() / "data"))
LC_DIR = DATA_DIR / "lightcurves"
CATALOG_DIR = DATA_DIR / "catalogs"

# TESS times are BTJD = BJD - 2457000
BTJD_OFFSET = 2457000.0

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
QLP_BASE = "https://archive.stsci.edu/hlsps/qlp"


def ensure_dirs() -> None:
    for d in (LC_DIR, CATALOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
