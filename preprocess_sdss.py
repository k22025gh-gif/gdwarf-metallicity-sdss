"""
preprocess_sdss.py
==================
SDSS G-type star spectral preprocessing pipeline.

For each star in the input catalog (produced by the CasJobs query in
query/sdss_gstars_query.sql), this module:

  1. Fetches the observed spectrum from SDSS via astroquery
  2. Masks telluric bands and bad wavelength regions
  3. Estimates a continuum via a chunked upper-envelope spline
  4. Normalises the spectrum (flux / continuum)
  5. Shifts the normalised spectrum to the rest frame using the
     SSPP radial velocity (ELODIERVFINAL) from sppParams
  6. Computes a signal-to-noise estimate from the SDSS inverse-variance array
  7. Saves per-star outputs:
       outputs/csv/   — appended to the batch results CSV
       outputs/json/  — {plate}-{mjd}-{fiber}.json
       outputs/plots/ — {plate}-{mjd}-{fiber}.png  (3-panel QC plot)

Public API
----------
    from preprocess_sdss import process_one, process_many
"""

import json
import yaml
from pathlib import Path

from astroquery.sdss import SDSS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import continuum_envelope, shift_to_restframe, mask_tellurics_and_bad

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(_CONFIG_PATH) as _f:
    _cfg = yaml.safe_load(_f)

OUTPUT_BASE = Path(_cfg["output_dir"])
CSV_DIR     = OUTPUT_BASE / "csv"
JSON_DIR    = OUTPUT_BASE / "json"
PLOT_DIR    = OUTPUT_BASE / "plots"

for _d in (CSV_DIR, JSON_DIR, PLOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference absorption lines used for visual QC (vacuum wavelengths, Å)
# Fe I and Mg I lines characteristic of G-type stellar atmospheres
# ---------------------------------------------------------------------------
QC_LINES = [
    4383.544, 4404.750, 4957.596, 5006.119,
    5167.321, 5172.684, 5183.604, 5328.039,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_wavelength_from_loglam(loglam: np.ndarray) -> np.ndarray:
    """Convert SDSS log10-wavelength array to linear wavelength in Angstrom."""
    return 10.0 ** loglam


def estimate_sn_from_ivar(
    flux: np.ndarray,
    ivar: np.ndarray,
    wl: np.ndarray = None,
    wl_min: float = 5000.0,
    wl_max: float = 6000.0,
) -> float:
    """
    Estimate per-pixel S/N from the SDSS inverse-variance array.

    Computes flux * sqrt(ivar) for each pixel, then returns the median
    absolute value within the wavelength window [wl_min, wl_max].

    The 5000–6000 Å window is used by default as it is free of strong
    telluric features and covers a relatively clean continuum region for
    G-type stars. This is the standard SDSS approach and is more reliable
    than estimating noise from flux fluctuations directly.

    Parameters
    ----------
    flux   : array  Observed flux
    ivar   : array  SDSS inverse-variance
    wl     : array or None
    wl_min : float  Blue edge of S/N window (Å, default 5000)
    wl_max : float  Red edge of S/N window (Å, default 6000)

    Returns
    -------
    sn : float  Median S/N per pixel, or NaN if undefined
    """
    if ivar is None:
        return np.nan
    mask = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    if wl is not None:
        mask &= (wl >= wl_min) & (wl <= wl_max)
    if np.sum(mask) == 0:
        return np.nan
    return float(np.nanmedian(np.abs(flux[mask] * np.sqrt(ivar[mask]))))


def _pycast(obj):
    """Coerce numpy scalars to Python natives for JSON serialisation."""
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    return obj


# ---------------------------------------------------------------------------
# QC plot
# ---------------------------------------------------------------------------

def plot_3panel(
    wl_obs: np.ndarray,
    flux_obs: np.ndarray,
    cont: np.ndarray,
    flux_norm: np.ndarray,
    wl_rest: np.ndarray,
    flux_rest: np.ndarray,
    ref_lines: list,
    meta: dict = None,
    savepath: str = None,
) -> None:
    """
    Save a 3-panel quality-control plot for one spectrum.

    Panel 1 — Observed flux + continuum estimate
    Panel 2 — Continuum-normalised flux (observed frame) with reference
              line positions marked in red
    Panel 3 — Continuum-normalised flux in the rest frame

    Note: flux_rest is numerically identical to flux_norm — the rest-frame
    shift moves only the wavelength axis (wl_rest), not the flux values.

    Parameters
    ----------
    wl_obs    : array  Observed wavelength (Å)
    flux_obs  : array  Observed flux
    cont      : array  Continuum estimate
    flux_norm : array  Normalised flux (observed frame)
    wl_rest   : array  Rest-frame wavelength (Å)
    flux_rest : array  Normalised flux (rest frame, same values as flux_norm)
    ref_lines : list   Reference wavelengths to mark (Å)
    meta      : dict   {'plate', 'mjd', 'fiber'} for plot title
    savepath  : str    Path to save PNG; if None, calls plt.show()
    """
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    ax0.plot(wl_obs, flux_obs, lw=0.6, color='0.2')
    ax0.plot(wl_obs, cont, lw=1.0, color='C1', label='continuum')
    ax0.set_ylabel("Flux")
    ax0.legend(fontsize='small')

    ax1.plot(wl_obs, flux_norm, lw=0.7, color='0.05')
    ax1.set_ylabel("Normalised flux")
    for wlab in ref_lines:
        ax1.axvline(wlab, color='red', linestyle='-', lw=0.8, alpha=0.7)
        ax1.text(wlab, 1.95, f"{wlab:.1f}", rotation=90,
                 va='top', ha='center', fontsize=7, color='red')

    ax2.plot(wl_rest, flux_rest, lw=0.7, color='0.05')
    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel("Rest-frame norm.")
    for wlab in ref_lines:
        ax2.axvline(wlab, color='red', linestyle='-', lw=0.8, alpha=0.7)

    if meta is not None:
        fig.suptitle(
            f"Plate {meta.get('plate')}  MJD {meta.get('mjd')}  Fiber {meta.get('fiber')}",
            fontsize=10,
        )
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------

def process_one(row: pd.Series, save_plots: bool = True) -> dict:
    """
    Process a single star from the input catalog.

    Fetches the SDSS spectrum identified by (plate, mjd, fiberID), performs
    continuum normalisation, shifts to the rest frame using the SSPP radial
    velocity, computes the ivar-based S/N, saves a QC plot and a JSON file,
    and returns a results dictionary.

    Parameters
    ----------
    row        : pd.Series  One row of the input catalog CSV.
                            Required columns: plate, mjd, fiber,
                            ELODIERVFINAL, ELODIERVFINALERR,
                            TEFFADOP, LOGGADOP
    save_plots : bool       Save the 3-panel QC PNG (default True)

    Returns
    -------
    result : dict  Keys: plate, mjd, fiber, sn_ivar, rv_sspp, rv_sspp_err,
                         teff, teff_ok, logg, logg_ok
    """
    plate = int(row['plate'])
    mjd   = int(row['mjd'])
    fiber = int(row['fiber'])

    # RV from SSPP — heliocentric-corrected, from sppParams.ELODIERVFINAL
    rv_sspp     = float(row['ELODIERVFINAL'])
    rv_sspp_err = float(row.get('ELODIERVFINALERR', np.nan))

    # Stellar parameters from SSPP (used for downstream quality flags)
    logg = float(row['LOGGADOP']) if pd.notna(row['LOGGADOP']) else np.nan
    teff = float(row['TEFFADOP']) if pd.notna(row['TEFFADOP']) else np.nan

    # ------------------------------------------------------------------
    # 1. Fetch spectrum from SDSS
    # ------------------------------------------------------------------
    spec_list = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiber)
    if not spec_list:
        raise ValueError(f"No SDSS spectrum returned for {plate}-{mjd}-{fiber}")

    data     = spec_list[0][1].data
    wl_obs   = build_wavelength_from_loglam(data['loglam'])
    flux_obs = data['flux'].astype(float)
    ivar     = data['ivar'].astype(float) if 'ivar' in data.columns.names else None

    # ------------------------------------------------------------------
    # 2. Continuum normalisation
    # ------------------------------------------------------------------
    cont      = continuum_envelope(wl_obs, flux_obs)
    flux_norm = flux_obs / cont

    # ------------------------------------------------------------------
    # 3. S/N estimate (ivar method, 5000–6000 Å window)
    # ------------------------------------------------------------------
    sn_ivar = estimate_sn_from_ivar(flux_obs, ivar, wl_obs)

    # ------------------------------------------------------------------
    # 4. Rest-frame shift using SSPP RV
    # ------------------------------------------------------------------
    wl_rest   = shift_to_restframe(wl_obs, rv_sspp)
    flux_rest = flux_norm.copy()

    # ------------------------------------------------------------------
    # 5. QC plot
    # ------------------------------------------------------------------
    if save_plots:
        plot_3panel(
            wl_obs, flux_obs, cont, flux_norm,
            wl_rest, flux_rest,
            QC_LINES,
            meta=dict(plate=plate, mjd=mjd, fiber=fiber),
            savepath=str(PLOT_DIR / f"{plate}-{mjd}-{fiber}.png"),
        )

    # ------------------------------------------------------------------
    # 6. Quality flags
    # ------------------------------------------------------------------
    logg_ok = bool(np.isfinite(logg) and logg > 4.0)
    teff_ok = bool(np.isfinite(teff))

    # ------------------------------------------------------------------
    # 7. Build result dict
    # ------------------------------------------------------------------
    result = {
        'plate':       plate,
        'mjd':         mjd,
        'fiber':       fiber,
        'sn_ivar':     float(sn_ivar) if np.isfinite(sn_ivar) else None,
        'rv_sspp':     rv_sspp,
        'rv_sspp_err': rv_sspp_err,
        'teff':        float(teff) if np.isfinite(teff) else None,
        'teff_ok':     teff_ok,
        'logg':        float(logg) if np.isfinite(logg) else None,
        'logg_ok':     logg_ok,
    }

    # ------------------------------------------------------------------
    # 8. Save per-star JSON
    # ------------------------------------------------------------------
    with open(JSON_DIR / f"{plate}-{mjd}-{fiber}.json", "w") as f:
        json.dump({k: _pycast(v) for k, v in result.items()}, f, indent=2)

    return result


def process_many(input_csv: str, output_csv: str = "results.csv") -> None:
    """
    Batch-process all stars in an input catalog CSV.

    Reads ``input_csv``, calls ``process_one()`` for each row, and writes
    a combined results CSV to ``CSV_DIR / output_csv``. Failures for
    individual stars are caught, logged to the 'error' column, and do not
    abort the batch.

    Parameters
    ----------
    input_csv  : str  Path to the input catalog CSV
    output_csv : str  Filename for the output CSV written into CSV_DIR
    """
    df = pd.read_csv(input_csv)
    output_rows = []

    for i, row in df.iterrows():
        plate = row["plate"]
        mjd   = row["mjd"]
        fiber = row["fiber"]
        print(f"[{i + 1}/{len(df)}] {plate}-{mjd}-{fiber}")

        base = row.to_dict()

        try:
            res = process_one(row)
            base.update({
                "sn_ivar":     res.get("sn_ivar"),
                "rv_sspp":     res.get("rv_sspp"),
                "rv_sspp_err": res.get("rv_sspp_err"),
                "teff":        res.get("teff"),
                "teff_ok":     res.get("teff_ok"),
                "logg":        res.get("logg"),
                "logg_ok":     res.get("logg_ok"),
                "error":       "",
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            base.update({
                "sn_ivar":     None,
                "rv_sspp":     None,
                "rv_sspp_err": None,
                "teff":        None,
                "teff_ok":     False,
                "logg":        None,
                "logg_ok":     False,
                "error":       str(e),
            })

        output_rows.append(base)

    out_path = CSV_DIR / output_csv
    pd.DataFrame(output_rows).to_csv(out_path, index=False)
    print(f"\nPipeline complete → {out_path}")
