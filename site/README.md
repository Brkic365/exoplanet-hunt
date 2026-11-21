# Portfolio page

A static, dependency-free page that shows the pipeline on real data. Open it with any
static server:

```bash
cd site && python -m http.server 8000   # http://localhost:8000
```

(`fetch()` and ES modules do not work from `file://`.)

## Sections and their files

| section | module | data |
|---|---|---|
| sky map of every star searched (WebGL points, coloured by classifier score; glowing stars open the walkthrough) | `js/sky.js` | `data/sky.json`, `data/backdrop.json` |
| one star, five stages (raw → detrended → periodogram / single-event scan → fold → verdict), driven by scroll | `js/walkthrough.js` | `data/stars.json`, `data/stars/<tic>.json` |
| injection-recovery heatmap with a found and a missed example per cell, recall vs SNR | `js/heatmap.js`, `js/results.js` | `data/heatmap.json`, `data/metrics.json` |
| scoreboard (known-TOI recall, classifier AUC, search speed) and the 21 hand-vetted sector-105 signals | `js/results.js` | `data/metrics.json`, `data/stars.json`, `data/sheets/*.png` |

`js/plot.js` holds the shared canvas helpers. `js/main.js` wires the sections together;
each section reads only its own data and its own DOM root, so it can be lifted into
another site (or a React/Svelte component) on its own.

## Moving it into the Field site

* Colours and fonts are CSS custom properties at the top of `style.css`; swap them for
  the site's tokens. The page is dark-only on purpose (a sky map).
* Copy `data/` as-is and keep the relative paths, or change the `fetch()` URLs.
* The sky uses raw WebGL 1, no three.js, so it can share a canvas setup with the site's
  particle field if needed.

## Regenerating the data

```bash
for s in 0075 0105; do curl -so data/catalogs/qlp_targets_s$s.csv https://archive.stsci.edu/hlsps/qlp/target_lists/s$s.csv; done
python scripts/tic_params.py 105     # TIC parameters for the sector-105 target list (for sky.json)
python scripts/week4_export.py       # downloads ~50 QLP light curves, writes site/data/ (~7 MB)
```

The sector-105 TIC target list and QLP target lists come from MAST
(`archive.stsci.edu/hlsps/qlp/target_lists/`); everything else is read from `results/`.
