"""
utils.py
========
Shared utility functions used across the SDSS G-star preprocessing
and fitting pipeline.

Importing from here ensures that continuum normalisation is applied
identically to both observed spectra (preprocess_sdss.py) and synthetic
PHOENIX models (fitter.py), which is a hard requirement for a valid
chi-square comparison.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from astropy.constants import c


# ---------------------------------------------------------------------------
# Continuum estimation
# ---------------------------------------------------------------------------

def continuum_envelope(
    wl: np.ndarray,
    flux: np.ndarray,
    n_chunks: int = 180,
    perc: float = 0.92,
    spline_k: int = 3,
    spline_s: float = None,
) -> np.ndarray:
    """
    Estimate the spectral continuum via a chunked upper-envelope spline.

    The spectrum is divided into ``n_chunks`` wavelength bins. In each bin
    the ``perc``-th percentile of the flux is taken as a continuum anchor
    point. A cubic spline (k=3) is then fitted through these anchor points.
    A 3rd-degree polynomial is used as a fallback if the spline fails.

    The 92nd percentile (perc=0.92) is chosen empirically so that absorption
    lines depress anchor values only weakly, while emission spikes or noise
    peaks do not dominate.

    The smoothing factor ``spline_s`` is set adaptively from the spread of
    anchor values and the wavelength span when not provided explicitly,
    preventing over-fitting on noisy spectra.

    This function is applied identically to both observed SDSS spectra and
    PHOENIX synthetic models so that the normalisation is consistent and
    any systematic bias in the continuum method cancels in the chi-square
    comparison.

    Parameters
    ----------
    wl       : array  Wavelength (Å), sorted ascending
    flux     : array  Observed or model flux
    n_chunks : int    Number of wavelength bins (default 180)
    perc     : float  Percentile used as anchor (default 0.92)
    spline_k : int    Spline degree (default 3)
    spline_s : float  Spline smoothing factor; auto-computed if None

    Returns
    -------
    cont : array  Continuum estimate, same shape as flux
    """
    indices = np.array_split(np.arange(len(wl)), max(8, int(n_chunks)))
    centers, vals = [], []

    for idx in indices:
        if len(idx) == 0:
            continue
        sub = flux[idx]
        sub = sub[np.isfinite(sub)]
        if len(sub) == 0:
            continue
        centers.append(np.median(wl[idx]))
        vals.append(np.percentile(sub, perc * 100.0))

    if len(centers) < 5:
        med = np.nanmedian(flux[np.isfinite(flux)]) if np.any(np.isfinite(flux)) else 1.0
        return np.ones_like(flux) * med

    centers = np.array(centers)
    vals    = np.array(vals)

    if spline_s is None:
        span     = wl.max() - wl.min()
        spline_s = (np.nanstd(vals) ** 2) * max(1.0, span / 1000.0) * 1e-2

    try:
        spl  = UnivariateSpline(centers, vals, k=spline_k, s=spline_s)
        cont = spl(wl)
    except Exception:
        cont = np.polyval(np.polyfit(centers, vals, deg=3), wl)

    cont[~np.isfinite(cont)] = np.nanmedian(vals)
    cont[cont <= 0]          = np.nanmedian(vals)
    return cont


# ---------------------------------------------------------------------------
# Rest-frame shift
# ---------------------------------------------------------------------------

def shift_to_restframe(wl: np.ndarray, rv_kms: float) -> np.ndarray:
    """
    Shift an observed wavelength array to the stellar rest frame.

    Uses the non-relativistic Doppler formula:
        wl_rest = wl_obs / (1 + v/c)

    The radial velocity used here is ELODIERVFINAL from the SDSS SSPP
    (sppParams table), which is already corrected to the heliocentric frame.

    Parameters
    ----------
    wl     : array  Observed wavelength (Å)
    rv_kms : float  Radial velocity in km/s (positive = receding)

    Returns
    -------
    wl_rest : array  Rest-frame wavelength (Å)
    """
    beta = rv_kms * 1000.0 / c.value
    return wl / (1.0 + beta)


# ---------------------------------------------------------------------------
# Wavelength masking
# ---------------------------------------------------------------------------

# Telluric absorption bands (Å):  O2 A-band and B-band
TELLURIC_REGIONS = [(6860, 6885), (7580, 7700)]

# Wavelength ranges discarded due to poor SDSS sensitivity
BAD_RANGES = [(0, 3500), (9200, 20000)]


def mask_tellurics_and_bad(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray = None,
    telluric_regions: list = None,
    bad_ranges: list = None,
) -> np.ndarray:
    """
    Build a boolean mask that is True where pixels are usable.

    Excludes non-finite / non-positive flux, bad ivar pixels,
    telluric absorption bands, and SDSS sensitivity edges.

    Parameters
    ----------
    wave             : array  Wavelength (Å)
    flux             : array  Observed flux
    ivar             : array or None
    telluric_regions : list of (min, max) tuples, optional
    bad_ranges       : list of (min, max) tuples, optional

    Returns
    -------
    mask : boolean array, True = good pixel
    """
    if telluric_regions is None:
        telluric_regions = TELLURIC_REGIONS
    if bad_ranges is None:
        bad_ranges = BAD_RANGES

    mask = np.isfinite(flux) & (flux > 0.0)
    if ivar is not None:
        mask &= np.isfinite(ivar) & (ivar > 0)
    for a, b in telluric_regions + bad_ranges:
        mask &= ~((wave >= a) & (wave <= b))
    return mask
