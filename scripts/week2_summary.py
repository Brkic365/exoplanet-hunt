"""Rebuild the week-2 figures and tables from the CSVs in results/s0075/.

    python scripts/week2_summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from exohunt import catalog
from exohunt import injection as I
from exohunt.plots import recall_curve_figure

R = Path("results/s0075")
INJ = R / "injections"


def main():
    tls = pd.read_csv(INJ / "tls_w0.75.csv")
    bls = pd.read_csv(INJ / "bls_w0.75_sde6.csv")
    bls["snr_expected"] = tls.set_index("id").loc[bls["id"], "snr_expected"].values

    # known TOIs (all magnitudes) with the same expected-SNR definition
    cm = pd.read_csv(R / "recovery_sde9_all_tmag99.csv")
    cm = cm[cm["n_transits_expected"] >= 2]
    cm = I.toi_expected_snr(cm, catalog.fetch_toi(), 75)
    cm["recovered"] = cm["status"].isin(["match", "alias"])
    cm["status"] = cm["status"].astype(str)
    cm.to_csv(R / "toi_expected_snr.csv", index=False)

    edges = np.array([1, 3, 5, 7, 9, 12, 16, 25, 50, 1e4])
    curves = {
        "injections, TLS (SDE ≥ 9)": I.recovery_vs(tls, "snr_expected", edges),
        "injections, own BLS (same false-alarm rate)": I.recovery_vs(bls, "snr_expected", edges),
        "known TOIs, TLS": I.recovery_vs(cm.assign(depth_retained=cm["depth_retained"]), "snr_expected", edges),
    }
    recall_curve_figure(curves, R / "figures" / "recall_vs_snr.png",
                        title="Sector 75: recall vs expected transit SNR")
    tab = pd.concat({k: v[["n", "recall"]] for k, v in curves.items()}, axis=1)
    tab.to_csv(R / "recall_vs_snr.csv")
    print(tab.round(3).to_string())

    # paired detrending / flux variants on the first 1500 injections
    base = tls[tls["id"] < 1500].set_index("id").sort_index()
    rows = []
    for tag, label in [("tls_w0.5_sub", "biweight 0.5 d"), (None, "biweight 0.75 d (default)"),
                       ("tls_w1.0_sub", "biweight 1.0 d"), ("tls_w1.5_sub", "biweight 1.5 d"),
                       ("tls_sysrm_sub", "0.75 d on QLP SYS_RM_FLUX")]:
        d = base if tag is None else pd.read_csv(INJ / f"{tag}.csv").set_index("id").sort_index()
        long_ = base["duration_d"] * 24 > 4
        rows.append(dict(variant=label, recall=d["recovered"].mean(),
                         recall_snr_7_12=d["recovered"][(base.snr_expected >= 7) & (base.snr_expected < 12)].mean(),
                         recall_snr_ge12=d["recovered"][base.snr_expected >= 12].mean(),
                         recall_snr_ge12_dur_gt4h=d["recovered"][(base.snr_expected >= 12) & long_].mean(),
                         depth_kept_dur_gt4h=d["depth_retained"][long_].median()))
    v = pd.DataFrame(rows)
    v.to_csv(INJ / "detrend_variants.csv", index=False)
    print(v.round(3).to_string(index=False))

    # false alarms on un-injected field stars
    fa = []
    for s, thr in (("tls", 9.0), ("bls", 6.0)):
        d = pd.read_csv(R / f"field_noinj_{s}.csv")
        r1 = d[d["rank"] == 1] if "rank" in d else d
        fa.append(dict(searcher=s, threshold=thr, n_stars=len(r1), flagged=(r1["sde"] >= thr).mean()))
    fa = pd.DataFrame(fa)
    fa.to_csv(R / "false_alarms_field.csv", index=False)
    print(fa.to_string(index=False))


if __name__ == "__main__":
    main()
