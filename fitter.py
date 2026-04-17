"""
fitter.py
=========
SDSS–PHOENIX chi-square metallicity fitter.

For each star in the cleaned catalog, fits a grid of PHOENIX synthetic
spectra to derive [Fe/H] by minimising chi-square. The best-fit
metallicity is refined by fitting a parabola through the three chi-square
values surrounding the grid minimum.

Method summary
--------------
1.  Load the observed normalised spectrum (.txt, 3 columns: wl, flux, err)
2.  For each logg in LOGG_GRID, evaluate chi-square at each [Fe/H] in Z_GRID
    at the two PHOENIX Teff grid points bracketing the star's SSPP Teff
3.  Interpolate chi-square linearly in Teff
4.  Fit a parabola through the three points around the chi-square minimum
    to derive a sub-grid metallicity and formal uncertainty
5.  Select the best logg by minimum chi-square
6.  Write results to a CSV

PHOENIX model preprocessing (per model)
----------------------------------------
- Wavelength: reconstructed from FITS header as exp(CRVAL1 + i*CDELT1)
              (log-linear spacing in ln(λ))
- Air → vacuum conversion: Morton (1991) formula
- Convolution to SDSS resolution: Gaussian kernel with
      σ_pixels = (1 / (R_target * 2.355)) / Δln(λ)_model
  where R_target = 2000 (SDSS).

  In principle the kernel should subtract the native PHOENIX resolution
  (R ≳ 10000) in quadrature. Because R_PHOENIX ≫ R_SDSS, the correction
  is only ~2% and does not significantly affect the fitting results.

- Interpolated to the observed wavelength grid (interp1d)
- Normalised with the same continuum_envelope() function used on the
  observed spectra, ensuring that any systematic bias in the continuum
  method cancels in the chi-square comparison

Cross-correlation alignment
---------------------------
The observed spectra are already shifted to the stellar rest frame using
ELODIERVFINAL from the SDSS SSPP. The cross-correlation step therefore
corrects only small residual wavelength misalignments between the observed
spectra and the synthetic models. The resulting pixel shifts correspond to
approximately one wavelength sampling interval (~70 km/s), which reflects
the discrete sampling of the SDSS wavelength grid rather than a physically
meaningful radial velocity measurement.

Fitting region
--------------
5050–6450 Å. Mg b (5167–5185 Å) and Na D (5885–5900 Å) are masked
because these strong lines are sensitive to non-LTE effects and chromospheric
activity not modelled in the PHOENIX grid.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from utils import continuum_envelope

---------
# Configuration
---------
INDEX_PATH   = "outputs/index_clean.csv"
META_PATH    = "outputs/catalog_clean.csv"
SPECTRA_DIR  = "outputs/spectra"
PHOENIX_BASE = "/home/shree/Downloads"
OUTPUT_PATH  = "outputs/metallicity_results.csv"

# PHOENIX metallicity grid (dex)
Z_GRID = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

# Surface gravity grid (log g)
# Main-sequence G stars: logg > 4.0; two values bracket the SSPP range
LOGG_GRID = [4.0, 4.5]

# Fitting region (Å)
FIT_WL_MIN = 5050
FIT_WL_MAX = 6450

# Lines masked due to non-LTE sensitivity
MASK_REGIONS = [
    (5167, 5185),   # Mg b triplet
    (5885, 5900),   # Na D doublet
]

---------
# Load and merge input tables
---------
index_df = pd.read_csv(INDEX_PATH)
meta_df  = pd.read_csv(META_PATH)

meta_df['id'] = (
    meta_df['plate'].astype(str) + '-' +
    meta_df['mjd'].astype(str)   + '-' +
    meta_df['fiber'].astype(str)
)

input_df = pd.merge(index_df, meta_df, on='id', how='inner')
print(f"Stars to fit: {len(input_df)}")


---------
# PHOENIX file locator
---------

def get_phoenix_path(teff: float, logg: float, feh: float) -> str | None:
    """
    Construct the local file path for a PHOENIX model spectrum.

    Returns None if the file does not exist, allowing the calling code
    to skip missing grid points gracefully.
    """
    feh_str    = f"-{abs(feh):.1f}" if feh <= 0 else f"+{feh:.1f}"
    folder     = f"PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z{feh_str}"
    filename   = (
        f"lte{int(teff):05d}-{logg:.2f}{feh_str}"
        f".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    )
    full_path  = os.path.join(PHOENIX_BASE, folder, filename)
    return full_path if os.path.exists(full_path) else None


---------
# Chi-square computation for one model
---------

def compute_chi_square(
    obs_wl: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    model_path: str,
) -> tuple:
    """
    Compute chi-square between one observed spectrum and one PHOENIX model.

    Steps
    -----
    1. Read model FITS, reconstruct log-linear wavelength grid
    2. Convert air → vacuum wavelengths (Morton 1991)
    3. Convolve model to SDSS resolution (R=2000)
    4. Interpolate model onto observed wavelength grid
    5. Normalise model with continuum_envelope (same function as obs)
    6. Cross-correlate to correct residual wavelength misalignment
       (shift is at most ~1 pixel; see module docstring)
    7. Compute chi-square over the fitting region, excluding masked lines

    Parameters
    ----------
    obs_wl    : array  Observed rest-frame wavelength (Å)
    obs_flux  : array  Observed normalised flux
    obs_err   : array  Observed flux uncertainty
    model_path: str    Path to PHOENIX FITS file

    Returns
    -------
    (chi2, reduced_chi2, n_pixels) or (nan, nan, nan) on failure
    """
    if not model_path or not os.path.exists(model_path):
        return np.nan, np.nan, np.nan

    
    # 1. Read PHOENIX model
    
    with fits.open(model_path) as hdul:
        model_flux_native = hdul[0].data.astype(float)
        header            = hdul[0].header
        # PHOENIX stores wavelength as log-linear: ln(λ) = CRVAL1 + i*CDELT1
        ln_wl_model     = header["CRVAL1"] + np.arange(header["NAXIS1"]) * header["CDELT1"]
        wl_air          = np.exp(ln_wl_model)
        delta_ln_lambda = header["CDELT1"]   # pixel size in ln(λ)

    # Restrict to relevant wavelength range (with margin)
    wl_mask            = (wl_air >= obs_wl.min() - 50) & (wl_air <= obs_wl.max() + 50)
    wl_air             = wl_air[wl_mask]
    model_flux_native  = model_flux_native[wl_mask]

    
    # 2. Air → vacuum conversion (Morton 1991)
    
    s2     = (1e4 / wl_air) ** 2
    wl_vac = wl_air * (
        1.0
        + (0.05792105 / (238.0185 - s2))
        + (0.00167917 / (57.362   - s2))
    )

    
    # 3. Convolve to SDSS resolution (R = 2000)
    #
    #    PHOENIX resolution: R ≳ 10000 (much higher than SDSS).
    #    In principle the kernel should use quadrature subtraction of
    #    native and target resolutions. Because R_PHOENIX ≫ R_SDSS,
    #    the correction is ~2% and is neglected (see module docstring).
    #
    #    σ in ln(λ)-pixel units: σ = 1 / (R_target × 2.355 × Δln(λ))
    
    sigma_pixels      = 1.0 / (2000.0 * 2.355 * delta_ln_lambda)
    model_flux_smooth = gaussian_filter1d(model_flux_native, sigma_pixels)

    
    # 4. Interpolate model onto observed wavelength grid
    
    interp_fn         = interp1d(wl_vac, model_flux_smooth,
                                 bounds_error=False, fill_value=1.0)
    model_resampled   = interp_fn(obs_wl)

    
    # 5. Normalise model (same continuum_envelope as observations)
    
    model_cont        = continuum_envelope(obs_wl, model_resampled)
    model_norm        = model_resampled / model_cont

    
    # 6. Cross-correlation residual alignment
    #
    #    Spectra are already in the rest frame (ELODIERVFINAL applied).
    #    This step corrects only sub-pixel residual misalignments between
    #    the observed spectrum and the model. The lag is restricted to
    #    ±50 pixels to prevent spurious large shifts.
    #    Edge pixels introduced by np.roll are excluded from the fit
    #    by the wavelength window cut applied below.
    
    corr        = np.correlate(obs_flux - 1.0, model_norm - 1.0, mode="same")
    center      = len(corr) // 2
    window      = slice(center - 50, center + 50)
    best_lag    = (np.arange(len(corr)) - center)[window][np.argmax(corr[window])]
    model_shift = np.roll(model_norm, best_lag)

    
    # 7. Chi-square over fitting region
    
    fit_mask = (obs_wl > FIT_WL_MIN) & (obs_wl < FIT_WL_MAX)

    # Mask strong non-LTE sensitive lines
    for wl_lo, wl_hi in MASK_REGIONS:
        fit_mask &= ~((obs_wl >= wl_lo) & (obs_wl <= wl_hi))

    # Exclude edge pixels that may contain roll wrap-around artefacts
    n_edge   = max(abs(best_lag) + 1, 5)
    fit_mask[:n_edge]  = False
    fit_mask[-n_edge:] = False

    valid = (
        fit_mask
        & np.isfinite(obs_err)   & (obs_err > 0)
        & np.isfinite(model_shift)
        & np.isfinite(obs_flux)
    )

    n_pix = np.sum(valid)
    if n_pix < 10:
        return np.nan, np.nan, np.nan

    chi2     = float(np.sum(((obs_flux[valid] - model_shift[valid]) / obs_err[valid]) ** 2))
    dof      = max(n_pix - 3, 1)
    chi2_red = chi2 / dof

    return chi2, chi2_red, int(n_pix)


---------
# Main fitting loop
---------
results = []

for idx, row in input_df.iterrows():
    star_id = row['id']
    print(f"[{idx + 1}/{len(input_df)}] {star_id}")

    # Load observed spectrum
    try:
        obs_wl, obs_flux, obs_err = np.loadtxt(
            os.path.join(SPECTRA_DIR, row['spectrum_file']), unpack=True
        )
    except Exception as e:
        print(f"  Could not load spectrum: {e}")
        continue

    # Restrict to fitting wavelength range
    wl_mask  = (obs_wl >= 5000) & (obs_wl <= 6500)
    obs_wl   = obs_wl[wl_mask]
    obs_flux = obs_flux[wl_mask]
    obs_err  = obs_err[wl_mask]

    # Bracket SSPP Teff on the PHOENIX 100 K grid
    teff              = row['TEFFADOP']
    teff_lo           = int(np.floor(teff / 100) * 100)
    teff_hi           = teff_lo + 100
    teff_weight       = (teff - teff_lo) / 100.0   # linear interpolation weight

    best_solution = None

    for logg in LOGG_GRID:

        chi2_lo = []
        chi2_hi = []

        for feh in Z_GRID:
            c2_lo, _, _  = compute_chi_square(obs_wl, obs_flux, obs_err,
                                               get_phoenix_path(teff_lo, logg, feh))
            c2_hi, _, _  = compute_chi_square(obs_wl, obs_flux, obs_err,
                                               get_phoenix_path(teff_hi, logg, feh))
            chi2_lo.append(c2_lo)
            chi2_hi.append(c2_hi)

        chi2_lo = np.array(chi2_lo)
        chi2_hi = np.array(chi2_hi)

        # Teff-interpolated chi-square curve
        chi2_interp = (1.0 - teff_weight) * chi2_lo + teff_weight * chi2_hi

        finite = np.isfinite(chi2_interp)
        if np.sum(finite) < 3:
            continue

        z_valid   = np.array(Z_GRID)[finite]
        chi2_valid= chi2_interp[finite]

        # Parabolic refinement around the grid minimum
        # Note: sample metallicities span −1.14 to −0.018 dex (well within
        # the grid), so the minimum never falls at a grid edge and the
        # three-point parabola always has distinct neighbours.
        min_idx   = np.argmin(chi2_valid)
        idx_range = np.clip([min_idx - 1, min_idx, min_idx + 1],
                            0, len(z_valid) - 1)
        coeffs    = np.polyfit(z_valid[idx_range], chi2_valid[idx_range], 2)

        if coeffs[0] <= 0:
            continue   # parabola opens downward — pathological case

        feh_best    = -coeffs[1] / (2.0 * coeffs[0])
        chi2_min    = float(np.poly1d(coeffs)(feh_best))
        sigma_feh   = float(np.sqrt(1.0 / coeffs[0]))

        # Diagnostics from the closest grid model
        best_grid_idx = np.argmin(np.abs(np.array(Z_GRID) - feh_best))
        if teff_weight > 0.5:
            _, chi2_red_diag, npix_diag = compute_chi_square(
                obs_wl, obs_flux, obs_err,
                get_phoenix_path(teff_hi, logg, Z_GRID[best_grid_idx])
            )
        else:
            _, chi2_red_diag, npix_diag = compute_chi_square(
                obs_wl, obs_flux, obs_err,
                get_phoenix_path(teff_lo, logg, Z_GRID[best_grid_idx])
            )

        if best_solution is None or chi2_min < best_solution['chi2']:
            best_solution = {
                'feh_derived':  feh_best,
                'sigma_feh':    sigma_feh,
                'best_logg':    logg,
                'chi2':         chi2_min,
                'chi2_red':     chi2_red_diag,
                'n_pix':        npix_diag,
            }

    if best_solution is None:
        print(f"  No valid solution found.")
        continue

    results.append({
        'id':              star_id,
        'subClass':        row.get('subClass'),
        'ra':              row['ra'],
        'dec':             row['dec'],
        'psfMag_r':        row['psfMag_r'],
        'snMedian_r':      row['snMedian_r'],
        'sn_ivar':         row['sn_ivar'],
        'TEFFADOP':        row['TEFFADOP'],
        'TEFFADOPUNC':     row.get('TEFFADOPUNC'),
        'LOGGADOP':        row['LOGGADOP'],
        'LOGGADOPUNC':     row.get('LOGGADOPUNC'),
        'best_logg':       best_solution['best_logg'],
        'SSPP_RV':         row['ELODIERVFINAL'],
        'SSPP_RV_err':     row.get('ELODIERVFINALERR'),
        'FEHADOP':         row['FEHADOP'],
        'FEHADOPUNC':      row.get('FEHADOPUNC'),
        'feh_derived':     best_solution['feh_derived'],
        'sigma_feh':       best_solution['sigma_feh'],
        'chi2':            best_solution['chi2'],
        'chi2_red':        best_solution['chi2_red'],
        'n_pix_valid':     best_solution['n_pix'],
    })

---------
# Save results
---------
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nFitting complete. Results saved to:\n  {OUTPUT_PATH}")
print(f"  Stars fitted: {len(out_df)}")
