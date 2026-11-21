"""Sector 105 search sample: M dwarfs (Teff < 3900 K, T <= 13.5) + K dwarfs (3900-5300 K, T <= 12), then download."""
import pandas as pd
from exohunt import config, ingest
d = pd.read_csv(config.CATALOG_DIR / "qlp_targets_s0105_tic.csv")
dw = d[d.lumclass == "DWARF"]
m = dw[(dw.Teff < 3900) & (dw.Tmag <= 13.5)].assign(group="M")
k = dw[(dw.Teff >= 3900) & (dw.Teff < 5300) & (dw.Tmag <= 12)].assign(group="K")
s = pd.concat([m, k]).drop_duplicates("tic")
s.to_csv(config.CATALOG_DIR / "s0105_sample.csv", index=False)
print(len(s), s.group.value_counts().to_dict(), flush=True)
got = ingest.download_many(s.tic.tolist(), 105, workers=16)
print("downloaded", sum(1 for v in got.values() if v), flush=True)
