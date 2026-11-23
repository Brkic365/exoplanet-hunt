"""Combine scored sectors into one ranked candidate list + vetting sheets.
Candidate = passes vetting and the systematics filter, SNR >= 10, host has no TOI, and the
period matches no TOI / CTOI / SPOC TCE on that star. CTOI hosts are kept but flagged.
Usage: python scripts/multi_rank.py 90 91 ... 105"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from exohunt import ingest
from exohunt.plots import vet_sheet
from exohunt.vetting import Ephemeris, vet_signal

CENTROID_MAX = 5.0  # sigma; every sector-105 hand-vetted signal above this was a blend, EB or systematic
out = Path("results/multi"); (out / "sheets").mkdir(parents=True, exist_ok=True)
sectors = [int(s) for s in sys.argv[1:]]
d = pd.concat([pd.read_csv(f"results/s{s:04d}/signals.csv").assign(sector=s) for s in sectors
               if Path(f"results/s{s:04d}/signals.csv").exists()], ignore_index=True)
c = d[d.rank_ok & ~d.known & ~d.toi_host & (d.snr >= 10)].copy()
c["centroid_ok"] = ~(c.centroid_sigma > CENTROID_MAX)

# same star flagged in more than one sector at the same period (or both single events)
def repeats(g):
    n = np.zeros(len(g), int)
    P, S = g.period.values, g.sector.values
    for i in range(len(g)):
        for j in range(len(g)):
            if S[i] != S[j] and ((np.isnan(P[i]) and np.isnan(P[j])) or abs(P[i] / P[j] - 1) < 0.01):
                n[i] += 1
    return pd.Series(n, index=g.index)

c["n_other_sectors"] = c.groupby("tic", group_keys=False).apply(repeats) if len(c) else []
# periodic: classifier score; single events (which the classifier scores low, see README): SNR
c["kind"] = np.where(c.method == "single", "single", "periodic")
# stars that already host a CTOI or SPOC TCE drop to tier 2: in hand-vetting, a period mismatch on
# such a star was almost always the known signal seen as a single event or an alias (e.g. 4x)
top = c.centroid_ok & (((c.kind == "periodic") & (c.score >= 0.5)) | ((c.kind == "single") & (c.snr >= 12)))
c["tier"] = np.select([top & ~c.ctoi_host & ~c.tce_host, c.centroid_ok], [1, 2], 3)
c = c.sort_values(["tier", "n_other_sectors", "score", "snr"], ascending=[True, False, False, False])
hv = out / "hand_vetting.csv"  # keep sheets for everything already hand-vetted
c["sheet"] = (c.tier == 1) | (c.tic.isin(pd.read_csv(hv).tic) if hv.exists() else False)
keep = ["tier", "sector", "tic", "kind", "period", "t0", "duration", "depth", "snr", "score", "p_cnn", "p_gbm",
        "rp_re", "tmag", "teff", "r_star", "centroid_sigma", "oddeven_sigma", "sec_snr", "n_events",
        "n_other_sectors", "ctoi_host", "tce_host", "local_noise_ratio", "flags", "sheet"]
c[keep].to_csv(out / "candidates.csv", index=False)
print("signals", len(d), "candidates", len(c), c.tier.value_counts().sort_index().to_dict(), flush=True)

for r in c[c.sheet].itertuples():
    p = ingest.qlp_path(r.tic, r.sector)
    if not p.exists():
        print("missing", p); continue
    raw = ingest.read_qlp(p)
    P = r.period if np.isfinite(r.period) else 1e4
    eph = Ephemeris(P, r.t0, r.duration, r.depth)
    m, _ = vet_signal(raw, eph)
    lab = f"P = {r.period:.4f} d" if np.isfinite(r.period) else "single event"
    vet_sheet(raw, eph, m, out / "sheets" / f"s{r.sector}_tic{r.tic}_{r.kind}_{r.t0:.2f}.png",
              title=f"TIC {r.tic}  sector {r.sector}  {lab}  T={r.tmag:.1f}  Teff={r.teff:.0f}  R*={r.r_star:.2f}  "
                    f"Rp~{r.rp_re:.1f} Re  score {r.score:.2f}")
