"""Cross-match catalogues and QLP target lists: ExoFOP CTOIs, every SPOC TCE stats file on MAST,
and the QLP target list of each sector given on the command line."""
import re, sys
from concurrent.futures import ThreadPoolExecutor
import requests
from exohunt import config

config.ensure_dirs()
C = config.CATALOG_DIR

def get(url, path):
    if path.exists():
        return
    r = requests.get(url, timeout=600); r.raise_for_status(); path.write_bytes(r.content)

get("https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv", C / "ctoi.csv")
base = "https://archive.stsci.edu/missions/tess/catalogs/tce/"
names = sorted(set(re.findall(r'href="([^"]+dvr-tcestats\.csv)"', requests.get(base, timeout=120).text)))
jobs = [(base + n, C / n) for n in names]
jobs += [(f"https://archive.stsci.edu/hlsps/qlp/target_lists/s{int(s):04d}.csv", C / f"qlp_targets_s{int(s):04d}.csv")
         for s in sys.argv[1:]]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda j: get(*j), jobs))
print("TCE files", len(names), "target lists", len(sys.argv) - 1)
