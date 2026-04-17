"""
spectrum_export.py
==================
Exports continuum-normalised, rest-frame-corrected SDSS spectra to plain
text files for use in the PHOENIX chi-square fitting pipeline.

For each star in the filtered catalog (catalog_fit_stars.csv), this script:

  1. Fetches the observed spectrum from SDSS via astroquery
  2. Normalises the flux using the same continuum method as the
     preprocessing pipeline (continuum_envelope from utils.py)
  3. Propagates flux errors from the SDSS inverse-variance:
         σ_flux = 1 / sqrt(ivar)
         σ_norm = σ_flux / continuum
  4. Shifts the wavelength axis to the stellar rest frame using
     ELODIERVFINAL from SSPP
  5. Applies a wavelength window cut (5000–6500 Å) appropriate for
     the metallicity-sensitive region used in chi-square fitting
  6. Saves each spectrum as a 3-column plain text file:
         wavelength_A  flux_normalised  sigma_flux_normalised
  7. Writes an index CSV mapping star IDs to spectrum files and
     SSPP stellar parameters

Output
------
  outputs/spectra/  — {plate}-{mjd}-{fiber}.txt per star
  outputs/index.csv — index of exported spectra
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from astroquery.sdss import SDSS
from utils import continuum_envelope, shift_to_restframe

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))

INPUT_CSV  = Path(_cfg["output_dir"]) / "csv" / "catalog_fit_stars.csv"
OUT_DIR    = Path(_cfg["output_dir"]) / "spectra"
INDEX_CSV  = Path(_cfg["output_dir"]) / "index.csv"

# Wavelength window: metallicity-sensitive region, free of strong tellurics
WL_MIN, WL_MAX = 5000, 6500

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_wavelength_from_loglam(loglam: np.ndarray) -> np.ndarray:
    return 10.0 ** loglam


# ---------------------------------------------------------------------------
# Main export loop
# ---------------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)
print(f"Stars to export: {len(df)}\n")

index_rows = []

for i, row in df.iterrows():
    plate = int(row["plate"])
    mjd   = int(row["mjd"])
    fiber = int(row["fiber"])
    rv    = float(row["ELODIERVFINAL"])
    tag   = f"{plate}-{mjd}-{fiber}"

    print(f"[{i + 1}/{len(df)}] {tag}")

    try:
        # ------------------------------------------------------------------
        # 1. Fetch spectrum
        # ------------------------------------------------------------------
        spec = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiber)[0]
        data = spec[1].data

        wl_obs = build_wavelength_from_loglam(data["loglam"])
        flux   = data["flux"].astype(float)
        ivar   = data["ivar"].astype(float)

        # ------------------------------------------------------------------
        # 2. Continuum normalisation
        # ------------------------------------------------------------------
        cont      = continuum_envelope(wl_obs, flux)
        flux_norm = flux / cont

        # ------------------------------------------------------------------
        # 3. Error propagation
        #    σ_flux = 1 / sqrt(ivar)    (from SDSS inverse-variance)
        #    σ_norm = σ_flux / cont     (propagated through normalisation)
        # ------------------------------------------------------------------
        sigma_flux             = np.zeros_like(flux)
        good_ivar              = ivar > 0
        sigma_flux[good_ivar]  = 1.0 / np.sqrt(ivar[good_ivar])
        sigma_norm             = sigma_flux / cont

        # ------------------------------------------------------------------
        # 4. Rest-frame shift using SSPP RV (ELODIERVFINAL)
        # ------------------------------------------------------------------
        wl_rest = shift_to_restframe(wl_obs, rv)

        # ------------------------------------------------------------------
        # 5. Wavelength window + quality mask
        # ------------------------------------------------------------------
        mask = (
            (wl_rest >= WL_MIN)      &
            (wl_rest <= WL_MAX)      &
            np.isfinite(flux_norm)   &
            np.isfinite(sigma_norm)  &
            (sigma_norm > 0)
        )

        wl  = wl_rest[mask]
        fn  = flux_norm[mask]
        err = sigma_norm[mask]

        if len(wl) < 500:
            raise ValueError(f"Too few spectral points after filtering ({len(wl)})")

        # ------------------------------------------------------------------
        # 6. Save spectrum
        # ------------------------------------------------------------------
        out_file = OUT_DIR / f"{tag}.txt"
        np.savetxt(
            out_file,
            np.column_stack([wl, fn, err]),
            fmt="%.6f",
            header="wavelength_A flux_normalized sigma_flux_normalized",
            comments="",
        )

        # ------------------------------------------------------------------
        # 7. Index entry
        # ------------------------------------------------------------------
        index_rows.append({
            "id":            tag,
            "spectrum_file": out_file.name,
            "TEFF":          row["TEFFADOP"],
            "LOGG":          row["LOGGADOP"],
            "FEH_SSPP":      row["FEHADOP"],
        })

    except Exception as e:
        print(f"  skipped: {e}")

# ---------------------------------------------------------------------------
# Write index
# ---------------------------------------------------------------------------
pd.DataFrame(index_rows).to_csv(INDEX_CSV, index=False)
print(f"\nExport complete.")
print(f"  Spectra written : {len(index_rows)}")
print(f"  Index saved to  : {INDEX_CSV}")
