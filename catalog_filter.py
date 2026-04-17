"""
catalog_filter.py
=================
Combines all preprocessing chunk CSVs into a single catalog and filters
to stars that are eligible for metallicity fitting.

Eligibility criteria (all must be met):
  - logg_ok  : logg finite and > 4.0  (main-sequence cut)
  - teff_ok  : TEFFADOP finite
  - sn_ivar  : finite  (ivar-based S/N successfully computed)
  - FEHADOP  : finite  (SSPP metallicity available for downstream comparison)

Input
-----
  outputs/csv/results_chunk_*.csv  — produced by chunk_runner.py

Output
------
  outputs/csv/catalog_fit_stars.csv  — filtered catalog, ready for
                                       spectrum_export.py
"""

import pandas as pd
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
CHUNK_DIR = Path(_cfg["output_dir"]) / "csv"
OUT_FILE  = CHUNK_DIR / "catalog_fit_stars.csv"

# ---------------------------------------------------------------------------
# Load and combine all chunk CSVs
# ---------------------------------------------------------------------------
chunk_files = sorted(CHUNK_DIR.glob("results_chunk_*.csv"))
if not chunk_files:
    raise FileNotFoundError(f"No chunk CSVs found in {CHUNK_DIR}")

dfs = []
for f in chunk_files:
    print(f"Reading: {f.name}")
    dfs.append(pd.read_csv(f))

all_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows loaded: {len(all_df)}")

# ---------------------------------------------------------------------------
# Filter: keep only stars eligible for metallicity fitting
# ---------------------------------------------------------------------------
fit_df = all_df[
    (all_df["logg_ok"]  == True) &
    (all_df["teff_ok"]  == True) &
    (all_df["sn_ivar"].notna()) &
    (all_df["FEHADOP"].notna())
].copy()

print(f"Stars eligible for metallicity fitting: {len(fit_df)}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fit_df.to_csv(OUT_FILE, index=False)
print(f"\nFiltered catalog written to:\n  {OUT_FILE}")
