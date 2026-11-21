"""Vetting sheets for every signal that passes vetting and the systematics filter, SNR >= 10, not already known."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from exohunt import ingest
from exohunt.plots import vet_sheet
from exohunt.vetting import Ephemeris, vet_signal
sector = int(sys.argv[1])
d = pd.read_csv(f"results/s{sector:04d}/signals.csv")
c = d[d.rank_ok & ~d.known & ~d.toi_host & (d.snr >= 10)]
for r in c.itertuples():
    raw = ingest.read_qlp(ingest.qlp_path(r.tic, sector))
    P = r.period if np.isfinite(r.period) else 1e4
    eph = Ephemeris(P, r.t0, r.duration, r.depth)
    m, _ = vet_signal(raw, eph)
    lab = f"P = {r.period:.4f} d" if np.isfinite(r.period) else "single event"
    vet_sheet(raw, eph, m, f"results/s{sector:04d}/sheets/tic{r.tic}_{r.method}_{r.t0:.2f}.png",
              title=f"TIC {r.tic}  sector {sector}  {lab}  T={r.tmag:.1f}  Teff={r.teff:.0f}  R*={r.r_star:.2f}  "
                    f"Rp~{r.rp_re:.1f} Re  score {r.score:.2f}")
