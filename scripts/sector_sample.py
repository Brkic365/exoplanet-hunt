"""M/K dwarf sample for a sector, same cuts as sector 105 (scripts/s105_targets.py), but with
TIC criteria queries over cones covering the sector instead of 2M ID lookups:
M dwarfs Teff < 3900 K, T <= 13.5; K dwarfs 3900-5300 K, T <= 12.
Usage: python scripts/sector_sample.py SECTOR [--download]"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from astroquery.mast import Catalogs
from exohunt import config, ingest

COLS = ["ID", "Tmag", "Teff", "rad", "mass", "logg", "rho", "lumclass", "d", "contratio"]
RADIUS = 4.5  # deg; Fibonacci centres ~6 deg apart leave gaps up to ~4.3 deg


def centres(n=1323):
    i = np.arange(n) + 0.5
    dec = np.degrees(np.arcsin(1 - 2 * i / n))
    ra = np.degrees(np.pi * (1 + 5 ** 0.5) * i) % 360
    return ra, dec


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def cone(c, radius=RADIUS):
    ra, dec = c
    for attempt in range(3):
        try:
            t = Catalogs.query_criteria(catalog="Tic", coordinates=f"{ra} {dec}", radius=radius,
                                        Teff=[1000, 5300], Tmag=[-5, 13.5], lumclass="DWARF")
            return pd.DataFrame({k: np.array(t[k]) for k in COLS})
        except Exception:
            time.sleep(10 * (attempt + 1))
    if radius < 0.5:
        raise RuntimeError(f"cone {ra} {dec} r={radius} failed")
    # dense fields (Galactic bulge) time out: cover the disc with 7 cones of ~half the radius
    sub = [(ra, dec)] + [(ra + 0.866 * radius * np.cos(np.radians(a)) / np.cos(np.radians(dec)),
                          dec + 0.866 * radius * np.sin(np.radians(a))) for a in range(0, 360, 60)]
    return pd.concat([cone((r % 360, d), 0.55 * radius) for r, d in sub])


def sample(sector):
    tl = pd.read_csv(config.CATALOG_DIR / f"qlp_targets_s{sector:04d}.csv"); tl.columns = ["tic", "ra", "dec"]
    cra, cdec = centres()
    U = unit(cra, cdec); T = unit(tl.ra.values, tl.dec.values)
    near = np.concatenate([np.argmax(T[i:i + 200000] @ U.T, 1) for i in range(0, len(T), 200000)])
    sep = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", T, U[near]), -1, 1)))
    assert sep.max() < RADIUS, sep.max()
    use = np.unique(near)
    with ThreadPoolExecutor(6) as ex:
        tic = pd.concat(ex.map(cone, zip(cra[use], cdec[use]))).drop_duplicates("ID")
    tic = tic.rename(columns={"ID": "tic"}).astype({"tic": int, "Teff": float, "Tmag": float})
    d = tl.merge(tic, on="tic")
    m = d[(d.Teff < 3900) & (d.Tmag <= 13.5)].assign(group="M")
    k = d[(d.Teff >= 3900) & (d.Teff < 5300) & (d.Tmag <= 12)].assign(group="K")
    s = pd.concat([m, k]).drop_duplicates("tic")
    s.to_csv(config.CATALOG_DIR / f"s{sector:04d}_sample.csv", index=False)
    print(sector, "cones", len(use), "sample", len(s), s.group.value_counts().to_dict(), flush=True)
    return s


if __name__ == "__main__":
    sector = int(sys.argv[1])
    s = sample(sector)
    if "--download" in sys.argv:
        got = ingest.download_many(s.tic.tolist(), sector, workers=16)
        print("downloaded", sum(1 for v in got.values() if v), flush=True)
