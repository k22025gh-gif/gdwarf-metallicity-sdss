# gdwarf-metallicity-sdss
Full-spectrum χ² metallicity estimation for SDSS DR17 G-dwarf stars using PHOENIX synthetic spectra. MS Physics project end-to-end computational Pipeline.
# SDSS G-Star Spectral Preprocessing and Metallicity Fitting Pipeline

End-to-end pipeline for deriving [Fe/H] of SDSS DR17 G-type main-sequence
stars by chi-square fitting of PHOENIX synthetic spectra, developed as part
of a stellar spectroscopy thesis project.
## Pipeline Overview

```
CasJobs SQL query (query/sdss_gstars_query.sql)
        ↓  download catalog CSV, split into chunks
chunk_runner.py  +  preprocess_sdss.py
        ↓  fetch spectra, normalise, rest-frame shift, compute S/N
catalog_filter.py
        ↓  combine chunks, filter eligible stars
spectrum_export.py
        ↓  re-fetch, normalise, propagate errors, export .txt spectra + index
cleaner.py
        ↓  canonicalise IDs, deduplicate by highest SNR
fitter.py
        ↓  PHOENIX chi-square grid fit → derived [Fe/H]
diagnosis.py
        ↓  calibration plots, Bland–Altman, residuals
```

## Data Source
Run the query in [`query/sdss_gstars_query.sql`](query/sdss_gstars_query.sql)
on [SDSS CasJobs (DR17)](https://skyserver.sdss.org/casjobs/).
**Selection criteria:**
- Spectral class: STAR, subClass LIKE 'G%'
- Median r-band S/N ≥ 15
- PSF r magnitude 10–15
- SSPP parameters available (sppParams join)
Download the result as a CSV and split into chunks if needed.


## Directory Structure

```
sdss_gstar_pipeline/
├── utils.py               # Shared: continuum_envelope, shift_to_restframe, masking
├── preprocess_sdss.py     # Stage 1: fetch, normalise, rest-frame shift, S/N
├── chunk_runner.py        # Batch runner for Stage 1
├── catalog_filter.py      # Stage 2: combine chunks, filter fit-eligible stars
├── spectrum_export.py     # Stage 3: export normalised spectra + index
├── cleaner.py             # Stage 4: deduplication and ID canonicalisation
├── fitter.py              # Stage 5: PHOENIX chi-square metallicity fitting
├── diagnosis.py           # Stage 6: calibration plots and statistics
├── config.yaml            # Input/output path configuration
├── requirements.txt       # Python dependencies
├── query/
│   └── sdss_gstars_query.sql
└── example/
    └── example_input.csv  # 3-row sample for testing Stage 1
```
## Installation
```bash
git clone https://github.com/your-username/sdss_gstar_pipeline.git
cd sdss_gstar_pipeline
pip install -r requirements.txt
```
Python ≥ 3.9 recommended.


## Configuration

Edit `config.yaml`:

```yaml
input_dir: "data/CSV_chunks"   # CasJobs CSV chunk files
output_dir: "outputs"          # root for csv/, json/, plots/, spectra/
```

PHOENIX model paths and fitting parameters are configured at the top of
`fitter.py`.


## Running the Pipeline

```bash
# Stage 1 — preprocess all chunks
python chunk_runner.py

# Stage 2 — combine and filter
python catalog_filter.py

# Stage 3 — export normalised spectra
python spectrum_export.py

# Stage 4 — clean and deduplicate
python cleaner.py

# Stage 5 — fit metallicities
python fitter.py

# Stage 6 — diagnostic plots
python diagnosis.py
```

All stages accept `--help` for path overrides.


## Key Design Choices
**Shared continuum function (`utils.py`):**
`continuum_envelope()` is defined once and imported by both
`preprocess_sdss.py` and `fitter.py`. This guarantees that observed spectra
and PHOENIX models are normalised identically, so any systematic bias in
the continuum method cancels in the chi-square comparison.

**Radial velocity correction:**
The pipeline uses `ELODIERVFINAL` from the SDSS SSPP (`sppParams`) for
rest-frame correction. This is already heliocentric-corrected. The
cross-correlation step inside `fitter.py` corrects only small residual
wavelength misalignments between the observed spectra and synthetic models
(typically ≲1 pixel, ~70 km/s); it does not measure a stellar radial velocity.

**PHOENIX convolution:**
Models are convolved with a Gaussian kernel to match SDSS resolution
(R = 2000). In principle the kernel should subtract the native PHOENIX
resolution (R ≳ 10000) in quadrature; because R_PHOENIX ≫ R_SDSS the
correction is ~2% and is neglected.

**S/N metric:**
`flux × sqrt(ivar)` in the 5000–6000 Å window follows the standard SDSS
convention. A separate Gray-style continuum S/N was explored but found to
lack astrophysical and instrumental justification and was not used.

**Deduplication:**
When a star appears on multiple plates, the observation with the highest
`snMedian_r` is retained.


## Output CSV Columns (`metallicity_results.csv`)
| Column | Description |
|||
| `id` | `plate-mjd-fiber` identifier |
| `feh_derived` | Pipeline-derived [Fe/H] (dex) |
| `sigma_feh` | Formal uncertainty from chi-square parabola curvature (dex) |
| `best_logg` | Best-fit log g from LOGG_GRID |
| `chi2` | Minimum interpolated chi-square value |
| `chi2_red` | Reduced chi-square at closest grid model |
| `n_pix_valid` | Number of pixels used in the fit |
| `FEHADOP` | SSPP reference metallicity for comparison |
| `SSPP_RV` | ELODIERVFINAL used for rest-frame correction (km/s) |

## Requirements
See [`requirements.txt`](requirements.txt).
