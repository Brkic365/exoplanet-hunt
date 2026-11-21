"""Command line entry point: `exohunt <command> --sector N`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import catalog, config, ingest


def _targets(sector: int, dispositions: str, limit: int | None) -> pd.DataFrame:
    toi = catalog.fetch_toi()
    disp = set(dispositions.split(",")) if dispositions else None
    t = catalog.tois_in_sector(toi, sector, disp)
    if limit:
        t = t.sort_values("tmag").head(limit)
    return t


def cmd_ingest(a):
    t = _targets(a.sector, a.dispositions, a.limit)
    tics = sorted(set(t["tic"].astype(int)))
    print(f"sector {a.sector}: {len(t)} TOIs on {len(tics)} stars -> downloading QLP")
    got = ingest.download_many(tics, a.sector, workers=a.workers)
    ok = [p for p in got.values() if p]
    print(f"downloaded {len(ok)}/{len(tics)} light curves into {config.LC_DIR / f's{a.sector:04d}'}")


def cmd_search(a):
    from .pipeline import run_sector

    paths = sorted((config.LC_DIR / f"s{a.sector:04d}").glob("*.fits"))
    out = Path(a.out or f"results/s{a.sector:04d}/detections.csv")
    print(f"searching {len(paths)} light curves with {a.searcher} -> {out}")
    run_sector(paths, out, workers=a.workers, searcher=a.searcher, max_signals=a.max_signals,
               window_length=a.window)


def cmd_report(a):
    from .crossmatch import crossmatch, recall_summary

    outdir = Path(f"results/s{a.sector:04d}")
    det = pd.read_csv(outdir / "detections.csv")
    searched = set(det["tic"].astype(int))
    t = _targets(a.sector, a.dispositions, None)
    t = t[t["tic"].astype(int).isin(searched)].copy()
    t["n_transits_expected"] = [
        catalog.expected_transits(ingest.read_qlp(ingest.qlp_path(r.tic, a.sector)).time, r.period, r.epoch_btjd,
                                  r.duration_hr / 24)
        for r in t.itertuples()
    ]
    cm = crossmatch(det, t, min_sde=a.min_sde)
    tag = f"sde{a.min_sde:g}"
    cm.to_csv(outdir / f"recovery_{tag}.csv", index=False)
    s = recall_summary(cm)
    s.to_csv(outdir / f"recall_summary_{tag}.csv")
    print(s.to_string())


def cmd_plot(a):
    from .plots import stage_figure

    p = ingest.qlp_path(a.tic, a.sector)
    if not p.exists():
        p = ingest.download_qlp(a.tic, a.sector)
    lc = ingest.read_qlp(p)
    out = Path(a.out or f"results/s{a.sector:04d}/figures/tic{a.tic}.png")
    r = stage_figure(lc, out, title=a.title or f"TIC {a.tic}, sector {a.sector}")
    print(f"wrote {out}  P={r['period']:.4f} SDE={r['sde']:.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exohunt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--sector", type=int, required=True)
        p.add_argument("--dispositions", default="CP,KP,PC,APC", help="TFOPWG dispositions to target")

    p = sub.add_parser("ingest", help="download QLP light curves for TOI hosts in a sector")
    common(p)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="detrend + transit search every downloaded light curve")
    common(p)
    p.add_argument("--searcher", default="tls")
    p.add_argument("--max-signals", type=int, default=2)
    p.add_argument("--window", type=float, default=0.75, help="biweight window [days]")
    p.add_argument("--workers", type=int)
    p.add_argument("--out")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("report", help="cross-match detections with the TOI list")
    common(p)
    p.add_argument("--min-sde", type=float, default=9.0)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("plot", help="four-stage figure for one star")
    p.add_argument("--sector", type=int, required=True)
    p.add_argument("--tic", type=int, required=True)
    p.add_argument("--title")
    p.add_argument("--out")
    p.set_defaults(func=cmd_plot)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
