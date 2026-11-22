# Exoplanet Hunt: handover for a case study

This file is written for someone (human or AI agent) who has never seen the
project and needs to turn it into a portfolio case study. Everything here is
checked against the code and result files in this repository as of
2026-11-22. Paths are relative to the repo root unless they say otherwise.

A file-by-file explanation of the code is in [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md).

---

## 0. Ground rules for the case study

* **Say "candidate", never "discovered".** Nothing this pipeline flags is a
  confirmed planet. Confirming a planet takes follow-up observations by other
  teams (ground-based photometry, imaging, spectroscopy). So far the search
  has produced **no new candidate**; every signal that reached hand-vetting
  was rejected or turned out to be already known (section 6).
* **The headline is measured performance**, not a discovery: how often the
  pipeline finds planets of a given strength, how well it separates planets
  from impostors, and how fast it runs.
* Use the numbers in section 5 as written. Several older numbers circulated
  during the project (for example a classifier AUC of 0.935 vs 0.79); those
  came from an earlier model version and were superseded. Section 5 is current.
* Antonio built this project with Claude (Anthropic's AI assistant) as a coding
  partner. See section 9 for how to describe roles.

---

## 1. One-paragraph summary

Exoplanet Hunt is an end-to-end search for planets around other stars in data
from NASA's TESS space telescope, written from scratch in Python with a custom
GPU kernel. It downloads a star's brightness measurements, removes the star's
own wobbles, searches for the small regular dips a passing planet causes,
runs automatic checks that catch the usual impostors, and ranks what is left
with a neural network trained on thousands of human-labelled examples. Every
stage is **measured**: 5,000 fake planets were planted in real data to see
which ones come back, the pipeline was scored against every known planet in a
test sector, and the classifier was scored on stars it never saw. It was then
pointed at a fresh 2026 sector (15,511 small, cool stars). A static web page
shows the whole thing on real data, including a WebGL map of every star searched.

---

## 2. The science, in plain language

**Transits.** When a planet passes in front of its star (as seen from Earth),
it blocks a tiny fraction of the star's light. A Jupiter-sized planet in front
of a Sun-like star dims it by about 1%; an Earth-sized planet by about 0.01%
(100 parts per million, "ppm"). If the orbit lines up, the dip repeats every
orbit: same depth, same duration, same spacing. Finding those repeating dips
in a long record of a star's brightness (a **light curve**) is the transit
method, which has found most known exoplanets.

**TESS.** The Transiting Exoplanet Survey Satellite (NASA, launched 2018)
photographs most of the sky in strips called **sectors**, each watched for
about 27 days. A camera image is taken every few minutes, so for each of
millions of stars there is a brightness measurement every 200 s (in recent
sectors). This project uses light curves produced by MIT's **Quick-Look
Pipeline (QLP)** from those full-frame images, downloaded from NASA's MAST archive.

**Why it is hard.**

* The dips are tiny compared with the noise. One 200 s measurement of a
  typical star here scatters by ~5,000 ppm; a small planet's dip is a few
  hundred ppm. Only by stacking many in-transit points does a dip emerge.
* Stars vary on their own (spots rotating in and out of view, pulsations),
  and the spacecraft adds trends (scattered sunlight, pointing jitter).
* Most dips that look like planets are not. The common impostors:
  * **eclipsing binaries**: two stars orbiting each other; one gives deep
    and shallow eclipses that can look like one planet at half the period;
  * **blends**: a binary on a neighbouring star whose light leaks into the
    same TESS pixel (each pixel is 21 arcseconds wide, huge on the sky);
  * **systematics**: instrument artefacts that dip many stars at once.

**What happens to a real signal.** A promising dip becomes a **TOI** (TESS
Object of Interest) if the TESS team flags it, or a **CTOI** (Community TOI)
if someone outside the team submits it to ExoFOP, the follow-up coordination
site. Follow-up teams then confirm or reject it. Labels used in this project:
CP (confirmed planet), KP (known planet), PC/APC (planet candidate), FP/FA
(false positive / false alarm).

---

## 3. The problem and the gap

Large teams already search TESS data: the official SPOC and QLP pipelines,
and independent efforts such as the T16 search of Cycle 1 and a 2025
transformer-based search of sectors 1 to 26. Redoing those would add little.
This project aims at places with fewer independent searches:

* **Recent extended-mission sectors** (sector 105, observed mid-2026, was the
  newest QLP sector on MAST and had no TOIs from it yet).
* **Single transits**: planets on long orbits that dip only once in a 27-day
  sector, which period-folding searches cannot find by design.
* **Hard host stars**: M and K dwarfs (small, cool stars where a small planet
  makes a deeper dip, but which are often active and noisy).

The second aim is credibility: a hobby pipeline that claims candidates without
knowing its own detection limits and false-alarm rate is not useful to anyone.
So the pipeline was measured before it was trusted.

---

## 4. Pipeline architecture

Five stages, each its own module, run per star. Measurement harnesses sit
around them.

```
 MAST (QLP light curves)      ExoFOP (TOI / CTOI tables)    MAST (SPOC TCE catalogues)
          |                            |                              |
  1. INGEST   download FITS, drop flagged cadences, normalise         |
          |                            |                              |
  2. DETREND  sliding biweight filter (0.75 d window) removes         |
              stellar variability, keeps dips up to ~6 h              |
          |                                                           |
  3. SEARCH   periodic: own Box Least Squares (CPU numba / CUDA GPU)  |
              or Transit Least Squares (reference), 2 signals/star    |
              single events: sliding-box scan, 1 to 16 h boxes        |
          |                                                           |
  4. VET      odd/even depth, secondary eclipse, centroid shift,      |
              momentum dumps, one-event dominance, implied radius,    |
              duration vs star density, local noise, orbit edges      |
          |                                                           |
  5. CLASSIFY CNN on folded light curve "views" + gradient boosting   |
              on vetting numbers, averaged (score 0 to 1)             |
          |                                                           |
  sector filters: cross-star systematics, known TOI/CTOI/TCE match <--+
          |
  hand-vetting with one-page "vetting sheets"  ->  verdict per signal

 Measurement harnesses
   injection-recovery: plant 5,000 synthetic planets in raw flux, run stages 2-3
   known-TOI recall: score against every TOI in sector 75
   false-alarm rate: run on 500 stars with no known planet
   classifier: train on 2,578 labelled TOIs, test on held-out host stars
   benchmark: stars/second for TLS vs own BLS on CPU vs own BLS on GPU
```

Key design choices worth telling:

* **Inject into raw data, before detrending.** That way the test measures
  everything the real search does to a signal, including what the filter eats.
* **Measure against expected signal-to-noise (SNR), not planet size.** A 2
  Earth-radius planet is easy on a bright small star and invisible on a faint
  big one. Recall against expected SNR transfers to any star.
* **Own BLS with a CUDA kernel.** One GPU thread block per (trial period,
  star); the block bins the light curve into shared memory, prefix-sums it,
  scores every box, and reduces to the best. Hundreds of stars go into one launch.
* **Vetting returns numbers, not just pass/fail**, so the classifier can learn
  from them.
* **Split training data by host star**, so the classifier is never tested on
  a star it trained on.
* **Cross-star systematics filter.** If many stars dip at the same moment, or
  share the same "period", it is the instrument, not planets.

---

## 5. Key measured results (verified)

All numbers are from files in `results/` and the README. Source in brackets.

### 5.1 Rediscovering known planets (sector 75)

* Sector 75 (2024), 250 known TOIs (CP/KP/PC/APC, T mag ≤ 13.5) on 227 stars.
* **44% recall** (76 of 173) of known planets that have at least 2 transits in
  the data, at the standard threshold (TLS SDE ≥ 9); 49% at SDE ≥ 7.
  Over all magnitudes: 79 of 181, also 44%. [`results/s0075/recovery_sde9.csv`, README week 1/2]
* Why not higher: 87 of those 181 have expected SNR below 9 in one sector,
  i.e. they are too faint to see in this data at all. **Among known planets
  with SNR ≥ 12 and P ≤ 13.5 d, recall is 91.5%.**
* Runtime: about 75 s for 227 stars on 4 CPU cores (TLS).

### 5.2 Injection-recovery (how good, and why it misses)

* 5,000 synthetic planets planted in the raw light curves of 500 sector-75
  stars with no known planet. [`results/s0075/injections/`]
* **93% recovered at expected SNR ≥ 12; 50% at SNR ≈ 9.5.**
* Recall by expected SNR (TLS): 5–7: 6%, 7–9: 36%, 9–12: 65%, 12–16: 80%,
  16–25: 92%, 25–50: 96%, > 50: 98%. Known TOIs follow the same curve,
  which shows the injections are realistic.
* Conclusion: **the search works at the noise limit**; missed shallow planets
  are too faint for one sector, not thrown away by the pipeline.
* False alarms: 2.8% of planet-free stars produce a signal above threshold
  (an upper bound; some may be real variables or binaries).

### 5.3 Own BLS search and GPU speed

* Same accuracy as the reference TLS at the same false-alarm rate:
  **26.2% vs 26.0%** of the same 5,000 injections.
* Speed, 227 stars, search only [`results/s0075/benchmark_gpu.csv`, `benchmark_cpu4.csv`]:

  | search | hardware | stars/s | vs TLS |
  |---|---|---|---|
  | TLS (reference) | 4-core cloud VM | 5.9 | 1× |
  | own BLS, numba CPU | 4-core cloud VM | 23.7 | 4× |
  | TLS (reference) | laptop, 16 threads | 3.6 | 1× |
  | own BLS, numba CPU | laptop, 16 threads | 52.0 | 14× |
  | **own BLS, CUDA** | **NVIDIA RTX 4050 Laptop GPU** | **148** | **41×** |

* The kernel is compute-bound (same throughput for 100 and 227 stars); file
  reading and detrending (~44 stars/s on one core) are now the bottleneck.

### 5.4 Single-transit detector

* 3,000 lone transits (P = 30–400 d) planted: **77% recovered at SNR 20–30,
  91% above 50**; 7.6% of planet-free stars also cross the threshold.
* On real data: finds **17 of 54** sector-75 TOIs that show exactly one
  transit (all the deep ones).

### 5.5 Vetting

On sector-75 detections that match known TOIs [`results/s0075/vetting_known.csv`]:
known false positives flagged **45%** of the time, confirmed/known planets
**12%**. Most FPs that pass were ruled out by ground-based follow-up at scales
TESS pixels cannot resolve.

### 5.6 Classifier

Trained on **2,578 ExoFOP TOIs** with a final disposition (1,261 planets vs
1,317 false positives), plus 1,249 "noise" signals from planet-free stars.
Tested on 502 TOIs around host stars never seen in training
[`results/classifier/metrics_test_planet_vs_fp.csv`]:

| model | ROC AUC (planet vs false positive) |
|---|---|
| logistic regression on plain BLS statistics | 0.67 |
| gradient boosting on plain BLS statistics (**baseline**) | **0.78** |
| CNN on folded light curves only | 0.84 |
| gradient boosting on BLS stats + vetting numbers | 0.93 |
| CNN on views + vetting numbers | 0.91 |
| **mean of the two above (used)** | **0.93** |

* Ensemble at score 0.5: precision 0.83, recall 0.90; average precision 0.91.
* Planet vs noise on held-out stars: AUC 0.999.
* Takeaway: hand-built vetting numbers are worth as much as the neural
  network; combined is best. Baseline → used model: **0.78 → 0.93 AUC**.
* **Noise class**: adding noise examples cut weak (SNR < 10) sector-105
  false alarms that score ≥ 0.5 from **69 to 22**. It cannot reach zero,
  because real planets also have SNR 5–8 in one sector; an SNR ≥ 10 gate
  before hand-vetting stays.

### 5.7 Sector 105 search (M and K dwarfs, 2026 data)

15,511 stars (7,152 M dwarfs, 8,359 K dwarfs), about an hour on 4 cloud cores.

| step | signals |
|---|---|
| detected (1,943 periodic + 6,750 single events) | 8,693 |
| removed as sector-wide systematics | −7,800 |
| fail vetting (overlapping) | −3,368 |
| pass everything | 149 |
| not matching any known TOI / CTOI / SPOC TCE | 135 |
| SNR ≥ 10, hand-vetted | **21** |

Hand-vetting verdicts [`results/s0105/hand_vetting.csv`]: 7 blends/eclipsing
binaries, 4 stellar variability, 7 noise/systematics, 1 inconclusive (too weak),
2 new transits on stars that already host single-transit CTOIs.
Sanity check: 19 of 43 known TOIs on sample stars recovered.

---

## 6. Interesting individual stars (good story material)

* **TIC 7145074**: a clean 5,200 ppm, 3.8 h single transit at BTJD 4217.50.
  The star already hosts a community single-transit candidate (CTOI
  7145074.01, from the WINE team, at BTJD 2111.52). If the same object, this
  is a second transit 2,106 days later, constraining the period to 2,106/n
  days. It is **someone else's candidate**; this pipeline independently
  caught another transit. Figure 4.
* **TIC 409696823**: a lone 5,100 ppm dip on a star with a single-transit
  CTOI; depth and duration disagree, so it may not be the same object.
* **TIC 139526205**: the strongest periodic signal in sector 105 looked like
  a 5.4 Earth-radius planet (SNR 44). The centroid check showed the light
  centre moving 22σ toward a neighbour during the dip; older sectors show the
  same period at 6% depth with odd/even differences. Verdict: an eclipsing
  binary next door, diluted. The classifier scored it 0.07. A great example of
  why vetting matters.
* **TOI-4481.01 (TIC 239332587)**: a confirmed 22-hour ultra-short-period
  planet, recovered at SDE 21 with a 938 ppm dip. Used for the four-stage figure.

---

## 7. Candidates from the multi-sector GPU search

> **⚠ PLACEHOLDER: TO BE UPDATED.** A multi-sector GPU search (roughly
> sectors 90 to 105) was running on Antonio's laptop when this was written.
> Do not invent results for it. When it lands, this section gets: sectors and
> star count, runtime on the RTX 4050, the filter funnel, the hand-vetted
> list with verdicts, and any signal judged worth submitting as a CTOI (with
> TIC, period, depth, SNR and vetting sheet path). Until then, the honest
> statement is: **"no new candidates so far; a wider multi-sector search is
> in progress."**

---

## 8. Honest limits

* One sector of QLP data limits sensitivity: most small planets around these
  stars are below the noise. More sectors stacked, or 2-minute SPOC data,
  would help more than a better algorithm.
* The periodic search needs at least 2 transits and P ≤ half the data span;
  longer periods rely on the single-transit detector, which needs high SNR.
* Classifier labels come from TOI dispositions, which teach "planet vs
  astrophysical false positive" better than "signal vs noise". Low-SNR
  signals still need a hard SNR gate and human judgement.
* Recent sectors are rough: 90% of sector-105 detections were instrument
  systematics.
* Not cross-checked: the T16 and 2025 transformer-search candidate tables,
  VizieR and the NASA Exoplanet Archive were unreachable from the cloud
  environment. None of the 21 hand-vetted signals needed them.
* No new candidate has been submitted. Nothing here is a discovery.

---

## 9. Tech stack and Antonio's role

**Language and libraries:** Python 3.10+, NumPy, pandas, SciPy, Astropy,
astroquery, numba and **numba-cuda (custom CUDA kernel)**, wotan (detrending),
transitleastsquares (reference search), batman (transit models for
injections), PyTorch (CNN), scikit-learn (gradient boosting, baselines),
matplotlib, pytest.

**Web page:** static HTML, vanilla JavaScript ES modules, **raw WebGL 1** (no
three.js) for the sky map, Canvas 2D for all charts, CSS custom properties.
No build step, no framework.

**Data sources:** TESS light curves via MIT QLP on MAST; ExoFOP-TESS TOI and
CTOI tables; SPOC TCE catalogues on MAST; TIC stellar parameters via MAST.

**Hardware:** cloud VMs (4 cores) for most runs; Antonio's laptop (Windows,
NVIDIA RTX 4050 Laptop GPU, 6 GB) for GPU benchmarks and the multi-sector search.

**Antonio's role:** project owner and lead. He chose the project and its
goals (an end-to-end, measured pipeline aimed at under-searched data, with a
public page), set the four-week plan and the scope decisions (what to aim at,
what to avoid redoing, "candidate, never discovered"), reviewed and merged
every stage (PRs #1–#5), ran the GPU work on his own hardware, and directs
what happens with any candidate. Implementation was done in collaboration
with Claude as an AI coding partner. A truthful phrasing: *"Designed and led;
built with an AI coding assistant."* Antonio should adjust this paragraph to
how he wants his contribution described.

Timeline: four part-time weekly milestones, all reached in November 2026:
W1 rediscover known planets on one sector; W2 measurement harness and GPU
kernel; W3 vetting, classifier, single transits, recent sector; W4 web page.

---

## 10. Figures

Ready-made figures (copies in `docs/figures/`; originals also in the project
files folder `exoplanet-figures/`):

| file | shows | caption idea |
|---|---|---|
| `docs/figures/1_four_stages_TOI-4481.png` | TIC 239332587 (TOI-4481.01), sector 75: raw flux with trend, detrended, TLS periodogram (P = 0.926 d, SDE 21.3), folded 938 ppm transit | "One star through the pipeline: a 22-hour planet emerges from the noise." |
| `docs/figures/2_injection_heatmap.png` | injection-recovery heatmap, period vs radius | "Where the pipeline can and cannot find planets, from 5,000 planted fakes." |
| `docs/figures/3_classifier_vs_baseline.png` | precision-recall curves for all classifier variants on held-out stars | "Vetting numbers plus a CNN beat plain search statistics." |
| `docs/figures/4_vetting_sheet_TIC7145074.png` | one-page vetting sheet for TIC 7145074's single transit | "A second transit on a known community candidate, 2,106 days later." |

More in the repo:

* `results/s0075/figures/recall_vs_snr.png`: recall vs expected SNR, injections and known TOIs.
* `results/s0075/figures/single_event_recall.png`: single-transit recall.
* `results/s0075/injections/heatmap_tls_w0.75.png`: the main heatmap (and variants).
* `results/classifier/pr_curves.png`: same as figure 3.
* `results/s0105/sheets/*.png`: vetting sheets for all 21 hand-vetted signals.
* `results/s0075/figures/tic*.png`: four-stage plots for other known planets.

Screenshots of the live page (sky map, walkthrough, heatmap) are the best hero images.

---

## 11. The project page

Static page in [`site/`](site/), built from the results above:

1. **Sky map**: WebGL plot of all 16,238 stars searched, coloured by
   classifier score; glowing stars open the walkthrough.
2. **One star, five stages**: scroll-driven walkthrough of real data (raw →
   detrended → periodogram or single-event scan → folded dip → verdict) for
   known planets and the sector-105 signals.
3. **What it can and cannot find**: interactive injection-recovery heatmap,
   with one found and one missed example per cell; recall vs SNR.
4. **The measured pipeline**: scoreboard (recall, classifier AUC, speed).
5. **Everything worth a look, and why each was rejected**: the 21 hand-vetted
   signals with vetting sheets.
6. **What worked, what did not**: the write-up.

Run locally: `cd site && python -m http.server 8000`.

---

## 12. Links

* Repository: https://github.com/Brkic365/exoplanet-hunt
* Live page: **TO BE FILLED IN.** No public deployment was verified when this
  was written (GitHub Pages status unknown). If Pages is enabled on `main`
  from the root, the page would be at
  `https://brkic365.github.io/exoplanet-hunt/site/`; check before linking.
* Data: MAST QLP HLSP (https://archive.stsci.edu/hlsp/qlp), ExoFOP-TESS
  (https://exofop.ipac.caltech.edu/tess/).
* Method references: Shallue & Vanderburg 2018 (AstroNet CNN), Hippke &
  Heller 2019 (TLS), Hippke et al. 2019 (wotan detrending).

---

## 13. Suggested case-study angles

* **"Measure before you trust."** The interesting part is not the search, it
  is knowing exactly what the search can and cannot see.
* **GPU engineering**: a from-scratch BLS kernel, 41× faster than the standard
  tool on a laptop GPU with the same accuracy.
* **ML with honest baselines**: a CNN that only matters in combination with
  domain features; a noise class added after the first version fooled itself.
* **A detective story**: TIC 139526205, the "planet" that was a neighbour's
  eclipsing binary.
* **Integrity**: 8,693 signals, 21 looked at by hand, zero claimed.
