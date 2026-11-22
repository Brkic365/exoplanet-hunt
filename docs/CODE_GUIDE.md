# Code guide: what each part does

A plain-language walk through every file, in the order data flows through the
pipeline. For what the project is and what it found, see [`HANDOVER.md`](../HANDOVER.md).

Layout:

```
src/exohunt/        the pipeline as an installable Python package
  config.py           paths and constants
  catalog.py          known-planet tables (ExoFOP)
  ingest.py           stage 1: get and clean light curves
  detrend.py          stage 2: remove the star's own variability
  search/             stage 3: find dips
    base.py             shared Detection type + "find, mask, repeat"
    tls.py              reference search (Transit Least Squares)
    bls.py              own Box Least Squares, CPU and CUDA GPU
    single.py           lone-dip detector
  vetting.py          stage 4: impostor checks
  views.py            folded pictures of a signal for the CNN
  classify/           stage 5: rank signals
    features.py         one row of inputs per signal
    model.py            the CNN and the ensemble
  labels.py           training labels from ExoFOP dispositions
  crossmatch.py       "did we find a known planet?"
  injection.py        plant fake planets and measure recovery
  pipeline.py         stages 1-3 over a whole sector (week 1-2 search)
  hunt.py             stages 2-5 on one star (week 3+ search)
  bench.py            speed benchmark
  plots.py            figures and vetting sheets
  cli.py              the `exohunt` command
scripts/            one-off runs that produced results/ (training, hunts, exports)
tests/              automated checks
site/               the web page (static HTML + JS)
results/            outputs: tables (CSV), figures (PNG), models (.pt)
```

Common terms: a **light curve** is a star's brightness over time; **BTJD** is
TESS's time unit (days); **SDE** and **SNR** are "how strong is this signal"
scores; an **ephemeris** is (period, time of first transit, duration, depth).

---

## Package: `src/exohunt/`

### `config.py`: settings
Where data lives on disk (`data/`, override with `EXOHUNT_DATA`), the TESS
time offset, and the ExoFOP download URL. Every other module imports paths from here.

### `catalog.py`: the list of known planets
Downloads the ExoFOP TOI table (every TESS Object of Interest with its
period, depth and disposition) and caches it for a day. Functions pick the
TOIs observed in a given sector, predict how many transits should fall in the
data, and pick "field stars" with no known planet for the fake-planet tests.
**In:** ExoFOP CSV. **Out:** pandas tables of targets.

### `ingest.py`: stage 1, get the data
Builds the MAST download URL for a star's QLP light curve directly from its
TIC id (no slow archive query), downloads many in parallel, and reads the
FITS file into a small `LightCurve` object (time, flux, error, metadata). Bad
measurements (non-zero quality flag) are dropped and flux is normalised to 1.
Handles both QLP file versions (older `KSPSAP_*` and newer `DET_FLUX` columns).
**In:** TIC id + sector. **Out:** `LightCurve`.

### `detrend.py`: stage 2, flatten the star
Runs a sliding biweight filter (from the `wotan` library) with a 0.75-day
window: it follows slow wiggles from starspots and the spacecraft, then
divides them out. A transit lasting a few hours is too short for the filter
to follow, so it survives. Upward outliers (flares) are clipped; downward
points are kept because they might be transits. The 0.75 d window was chosen
by paired tests against 0.5, 1.0 and 1.5 d (README week 2).
**In:** raw `LightCurve`. **Out:** flat `LightCurve` + the trend.

### `search/`: stage 3, find dips

* **`base.py`**: the `Detection` record every searcher returns (period,
  time of transit, duration, depth, SDE, SNR, number of transits), plus
  `iterative_search`: find the strongest signal, cut its transits out, search
  again (up to 2 signals per star).
* **`tls.py`**: wraps Transit Least Squares, the standard published tool,
  as the reference to compare against. It fits a realistic transit shape;
  accurate but slow.
* **`bls.py`**: Box Least Squares written from scratch. For each trial
  period it folds the light curve onto one orbit, sums it into phase bins,
  and slides a box (0.5 to 8 h) over the bins looking for the deepest
  significant dip. Prefix sums make each box O(1). Two backends run the same
  maths:
  * CPU with numba (compiled Python, parallel over periods);
  * **GPU with a hand-written CUDA kernel**: one thread block per (period,
    star) pair; the block bins the data in fast shared memory, prefix-sums it
    together, each thread scores a slice of box positions, and a reduction
    keeps the best. Hundreds of stars run in one launch.
  The periodogram is turned into an SDE score the same way TLS does it, so
  thresholds are comparable (BLS SDE 6 ≈ TLS SDE 9 in false alarms).
* **`single.py`**: finds lone dips (planets on orbits longer than the
  27-day sector). Slides boxes of 1 to 16 h along the light curve and scores
  each by its depth relative to the spread of *all* box depths of that length
  in the same star, so a jittery star needs a deeper dip to count. Long boxes
  use a light curve flattened with a longer filter so long dips are not eaten.
  Each event is re-measured with the event masked during detrending.

**In:** flat `LightCurve`. **Out:** list of `Detection`.

### `vetting.py`: stage 4, catch impostors
For one signal, re-detrends the light curve with that signal's transits
masked (so the filter cannot shrink the dip being judged), then measures:

| check | the impostor it catches |
|---|---|
| odd vs even transit depth | eclipsing binary found at half its period |
| dip elsewhere in the orbit (secondary) | eclipsing binary |
| image centroid moves during the dip | light from a neighbouring star (blend) |
| transits line up with momentum dumps | spacecraft pointing glitches |
| one event carries most of the signal | a single outlier, not a repeating dip |
| implied size > 30 Earth radii | a star, not a planet |
| dip too long for this star's density | wrong star, giant, or binary |
| scatter around the dip vs the whole curve | dip sits in a bad stretch |
| lone dip right at the start of a data segment | orbit-start ramps |

Each check returns a number (corrected for correlated "red" noise) and a
flag. Numbers go to the classifier; flags give pass/fail.
**In:** raw `LightCurve` + `Ephemeris`. **Out:** dict of metrics and flags.

### `views.py`: pictures for the neural network
Folds the light curve at the signal's period and bins it into fixed-length
arrays, after the AstroNet paper: a **global view** (the whole orbit, 201
bins) and a **local view** (±2 transit durations, 61 bins) with five
channels: all transits, odd transits, even transits, the window around the
strongest secondary dip, and centroid motion. Flux is rescaled so the dip
bottom is −1: the network learns shape; depth and SNR are given separately.
Also refines a catalogue ephemeris to the data before folding.

### `classify/`: stage 5, rank signals

* **`features.py`**: `signal_record` runs vetting + views for one signal
  and returns one row: the arrays plus 20 numbers (5 plain BLS statistics +
  vetting metrics + star brightness, temperature, radius). Defines which
  columns count as "plain BLS statistics" (the baseline) vs "with vetting".
* **`model.py`**: the CNN (PyTorch). Two towers of 1-D convolutions, one
  for each view, joined with the scalar features into a small dense head that
  outputs a planet probability; ~151k parameters, trains in minutes on CPU.
  `Ensemble` averages 5 seeds. The final score is the mean of this CNN and a
  gradient-boosting model on the scalar features.

### `labels.py`: training labels
Picks every ExoFOP TOI with a final verdict (CP/KP = planet, FP/FA = not),
downloads light curves from its two most recent sectors, and assigns each host
star to train / validation / test by a hash of its TIC id, so the same star
never lands on both sides.

### `crossmatch.py`: did we find a known planet?
Compares a detected period and transit time with a TOI's, allowing simple
multiples (½, 2, ⅓, 3, and 3/2 or 2/3 when data gaps hide every other transit),
and checks the transits land where the TOI's ephemeris predicts. Produces the
recall tables.

### `injection.py`: plant fake planets
Generates realistic transit shapes (`batman` library, limb darkening,
averaged over the 200 s exposure) with random period, size and impact
parameter, using each star's real radius and mass; adds them to the **raw**
flux; runs detrend + search + crossmatch; records whether it came back, its
expected SNR, and how much depth survived detrending. Also builds the heatmap
and recall-vs-SNR tables. This is how the pipeline's detection limits were measured.

### `pipeline.py`: week-1/2 sector search
Runs ingest → detrend → search for every star in a sector in parallel and
writes one CSV of detections. Used for the known-TOI recall work.

### `hunt.py`: full pipeline on one star (week 3+)
detrend → periodic BLS (SDE ≥ 6) → single-event search (SES ≥ 9, periodic
signals masked) → vetting + views for each signal. Classifier scoring is done
afterwards in one batch for speed. This is what ran on all 15,511 sector-105 stars.

### `bench.py`: speed benchmark
Times TLS, CPU BLS and CUDA BLS on the same detrended light curves and period
grid; reading and detrending are timed separately. Produced the 148 stars/s, 41× numbers.

### `plots.py`: figures
The four-panel stage figure (raw, detrended, periodogram, fold), the
injection heatmap, recall curves, and the one-page **vetting sheet** used for
hand-vetting (light curve, fold, odd/even, secondary, centroid, numbers).

### `cli.py`: the command line
`exohunt ingest | search | report | plot | inject | bench --sector N`, thin
wrappers around the modules above.

---

## `scripts/`: the runs behind the results

One-off scripts, each documented at its top. Roughly in order:

| script | does |
|---|---|
| `week2_summary.py` | rebuilds week-2 figures and tables from CSVs |
| `fetch_week3_data.py` | downloads sector-75 and training light curves |
| `build_training_set.py` | views + vetting features for every labelled TOI |
| `build_noise_set.py`, `build_noise_set_mk.py` | "noise" training examples: everything the hunt finds on stars with no known planet |
| `week3_train.py` | trains and compares all classifiers, writes `results/classifier/` |
| `week3_vet_s75.py` | vetting on known sector-75 signals (planets vs known FPs) |
| `week3_single_inject.py`, `week3_single_tois.py` | single-transit injection test and real-TOI test |
| `tic_params.py` | attaches star parameters (TIC) to a sector's target list |
| `s105_targets.py` | picks the sector-105 M/K-dwarf sample and downloads it |
| `week3_hunt.py` | runs `hunt.py` over a whole sector in parallel |
| `week3_score.py` | systematics filter, classifier scores, matching against known TOI/CTOI/TCE lists |
| `week3_sheets.py` | vetting sheets for signals that survive, for hand-vetting |
| `week4_export.py` | writes the compact JSON the web page reads into `site/data/` |

---

## `tests/`

* `test_synthetic.py`: plants a box transit in fake noise and checks every
  stage end to end, plus the period-matching rules.
* `test_bls.py`: CPU BLS finds an injected transit; the CUDA kernel gives the
  same periodogram as the CPU (in numba's simulator without a GPU, on the real
  GPU with `EXOHUNT_REAL_CUDA=1`).
* `test_vetting.py`: a fake planet passes vetting while a fake eclipsing
  binary is flagged; edge and single-event checks.

Run with `pytest`.

---

## `site/`: the web page

No framework and no build step; open with any static server.

| file | does |
|---|---|
| `index.html` | page structure and text |
| `style.css` | layout; colours and fonts are CSS variables at the top |
| `js/main.js` | loads the data files and starts each section |
| `js/sky.js` | WebGL 1 sky map: every searched star as a point on an equal-area projection, coloured by classifier score, clickable |
| `js/walkthrough.js` | scroll-driven five-stage view of one star's real data |
| `js/heatmap.js` | interactive injection-recovery grid with found/missed example folds |
| `js/results.js` | scoreboard, recall charts, list of hand-vetted signals |
| `js/plot.js` | shared Canvas helpers: axes, scales, ticks, colour ramp |
| `data/*.json`, `data/stars/`, `data/sheets/` | exported by `scripts/week4_export.py` |

Each section reads only its own data and its own part of the page, so it can
be lifted into another site on its own. See [`site/README.md`](../site/README.md).

---

## `results/`

* `s0075/`: sector-75 detections, recall tables, injection campaign,
  benchmarks, vetting of known signals, figures.
* `s0105/`: sector-105 signals, hand-vetting verdicts, vetting sheets.
* `classifier/`: trained CNN weights, test metrics, predictions, PR curves.

Raw light curves (`data/`) are not committed; scripts re-download them from MAST.
