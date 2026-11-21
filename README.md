# exoplanet-hunt

An end-to-end transit search on NASA TESS light curves, written from scratch:
ingest, detrend, search, vet, classify. The goal is a pipeline whose accuracy is
**measured** (injection-recovery, known-planet recall, false-positive rate), then
pointed at sectors and stars that have had few independent searches.

Anything this pipeline flags is a *candidate*. Confirmation needs follow-up
observations by others.

## Status: week 1 (rediscover known planets)

Sector 75 (2024), QLP full-frame-image light curves, every ExoFOP TOI host with
TFOPWG disposition CP / KP / PC / APC and T ≤ 13.5: **250 TOIs on 227 stars**.

| step | what it does |
|---|---|
| ingest | QLP `SAP_FLUX` from MAST; drop cadences with `QUALITY != 0`; normalise |
| detrend | wotan sliding biweight, 0.75 d window; clip upward outliers only |
| search | Transit Least Squares, P = 0.6 d to baseline/2, up to 2 signals per star (mask and repeat) |
| cross-match | detected period vs TOI period (incl. 1/2, 2, 1/3, 3, 3/2, 2/3 aliases), TOI ephemeris propagated to our epoch |

**Runtime:** about 75 s for all 227 stars on 4 CPU cores (≈0.3 s per star-signal).

### Recall on known TOIs

Only TOIs with **at least 2 transits inside the sector's good data** can be found by
a periodic search, so recall is computed over those 173 (the other 77 have 0 or 1
transit in this sector; that is the single-event detector's job in week 3).

| TFOPWG disposition | n | recovered, SDE ≥ 9 | recall | recovered, SDE ≥ 7 | recall |
|---|---|---|---|---|---|
| CP (confirmed) | 33 | 14 | 42% | 17 | 52% |
| KP (known planet) | 17 | 9 | 53% | 9 | 53% |
| APC | 18 | 15 | 83% | 15 | 83% |
| PC | 105 | 38 | 36% | 44 | 42% |
| **all** | **173** | **76** | **44%** | **85** | **49%** |

"Recovered" = exact period match or a simple alias, with transits landing on the
TOI ephemeris. One more (TOI-1136.01, a strong-TTV system) matches in period but
its transits are offset by ~6 h from the linear ephemeris, so it is not counted.

Recall by TOI depth (SDE ≥ 9):

| depth | n | recall |
|---|---|---|
| < 500 ppm | 25 | 8% |
| 500 – 1000 ppm | 32 | 28% |
| 1000 – 3000 ppm | 49 | 37% |
| > 3000 ppm | 67 | 70% |

Smallest planets recovered: TOI-4481.01 (1.7 R⊕, 950 ppm, P = 0.93 d),
TOI-1430.01 (2.0 R⊕, 700 ppm, P = 7.43 d), TOI-1339.01 (3.2 R⊕, P = 8.88 d).

Full tables: [`results/s0075/`](results/s0075/) (`detections.csv`, `recovery_sde9.csv`, `recall_summary_sde9.csv`).

![TOI-4481.01 through the four stages](results/s0075/figures/tic239332587.png)

### What week 1 taught

* **QLP v02 files** (later sectors) rename `KSPSAP_FLUX` to `DET_FLUX`; the reader handles both.
* **Data gaps create 3/2 aliases.** TOI-1148.01 (P = 5.55 d) shows only two transits
  16.65 d apart, so TLS picks 8.33 d. TLS's reported T0 can also sit inside a gap,
  so the cross-match checks several detected epochs, not just T0.
* **Stellar-density period grids cut both ways.** Using each star's TIC radius and
  mass lets TLS search below 0.6 d (it recovered the 0.41 d TOI-1442.01) but the
  denser grid lowered SDEs elsewhere and lost six other recoveries, so the default
  stays solar (`use_stellar_params=False`). Week 2's injection
  tests will settle the grid properly.
* **Shallow signals are the gap.** Recall falls below 30% under 1000 ppm. That,
  plus single transits and long periods, is what injection-recovery has to map.

## Usage

```bash
pip install -e .[dev]
exohunt ingest --sector 75          # TOI hosts in the sector -> data/lightcurves/s0075
exohunt search --sector 75          # detrend + TLS -> results/s0075/detections.csv (resumable)
exohunt report --sector 75 --min-sde 9
exohunt plot   --sector 75 --tic 239332587
pytest                              # synthetic-transit end-to-end test
```

`EXOHUNT_DATA` overrides the data directory (default `./data`, git-ignored).

## Layout

```
src/exohunt/
  catalog.py     ExoFOP TOI table, per-sector targets, expected transit counts
  ingest.py      QLP download (URL from TIC id), quality mask, stitching
  detrend.py     wotan biweight
  search/        Detection type, iterative masking, TLS (GPU BLS slots in here)
  crossmatch.py  period/epoch matching with aliases
  pipeline.py    parallel per-sector runner
  plots.py       raw -> detrended -> periodogram -> folded figure
```

## Plan

1. ~~Week 1: rediscover known planets on one sector (CPU TLS)~~
2. Week 2: injection-recovery harness, GPU BLS kernel, vetting (odd/even, secondary, centroid, momentum dumps)
3. Week 3: AstroNet-style CNN on ExoFOP dispositions; recent sectors; single-transit detector
4. Week 4: portfolio page and any CTOI submission

## Data credits

TESS QLP light curves (Huang et al. 2020; Kunimoto et al. 2022) via MAST.
TOI catalogue from ExoFOP-TESS. Transit Least Squares (Hippke & Heller 2019),
wotan (Hippke et al. 2019).
