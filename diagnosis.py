"""
diagnosis.py
============
Diagnostic plots and statistics comparing SSPP metallicities (FEHADOP)
with pipeline-derived metallicities (feh_derived).

Produces
--------
  1.  Scatter plot: derived vs SSPP [Fe/H] with linear regression
  2.  Bland–Altman plot: difference vs mean metallicity
  3.  Residual vs [Fe/H]
  4.  Residual vs Teff
  5.  Residual vs log g
  6.  Residual vs SDSS SNR (snMedian_r)
  7.  Residual vs SNR with running median
  8.  Chi-square distribution
  9.  Reduced chi-square distribution
  10. Derived metallicity histogram
  11. SSPP metallicity histogram
  12. Residual histogram

Usage
-----
    python diagnosis.py --results outputs/metallicity_results.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
import yaml


def parse_args() -> argparse.Namespace:
    _cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    parser = argparse.ArgumentParser(description="Metallicity fitter diagnostics.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/metallicity_results.csv"),
        help="Path to the fitter output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.results)
    df = df.dropna(subset=["FEHADOP", "feh_derived"])
    df["residual"] = df["feh_derived"] - df["FEHADOP"]

    x_sspp    = df["FEHADOP"].values
    y_derived = df["feh_derived"].values

    # ------------------------------------------------------------------
    # Linear regression
    # ------------------------------------------------------------------
    model     = LinearRegression().fit(x_sspp.reshape(-1, 1), y_derived)
    slope     = model.coef_[0]
    intercept = model.intercept_
    y_pred    = model.predict(x_sspp.reshape(-1, 1))
    r2        = r2_score(y_derived, y_pred)
    bias      = df["residual"].mean()
    sigma     = df["residual"].std()

    print("\n--- Calibration relation ---")
    print(f"[Fe/H]_derived = {slope:.4f} * [Fe/H]_SSPP + {intercept:.4f}")
    print("\n--- Performance ---")
    print(f"  Bias  (mean residual) = {bias:+.3f} dex")
    print(f"  Sigma (std residual)  = {sigma:.3f} dex")
    print(f"  R²                    = {r2:.3f}")
    print(f"  N stars               = {len(df)}")

    minv = min(x_sspp.min(), y_derived.min())
    maxv = max(x_sspp.max(), y_derived.max())

    # ------------------------------------------------------------------
    # 1. Scatter + regression
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_sspp, y_derived, alpha=0.6, s=15)
    ax.plot([minv, maxv], [minv, maxv], 'k--', label='Identity')
    ax.plot(x_sspp, y_pred, 'r',
            label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
    ax.set_xlabel("SSPP [Fe/H]")
    ax.set_ylabel("Derived [Fe/H]")
    ax.set_title(f"Metallicity Comparison  (R² = {r2:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 2. Bland–Altman
    # ------------------------------------------------------------------
    mean_feh = (df["FEHADOP"] + df["feh_derived"]) / 2.0
    fig, ax  = plt.subplots(figsize=(7, 6))
    ax.scatter(mean_feh, df["residual"], alpha=0.6, s=15)
    ax.axhline(bias,               color='red',  label=f'Bias = {bias:+.3f}')
    ax.axhline(bias + 1.96 * sigma, color='gray', ls='--',
               label=f'±1.96σ = ±{1.96*sigma:.3f}')
    ax.axhline(bias - 1.96 * sigma, color='gray', ls='--')
    ax.set_xlabel("Mean [Fe/H]")
    ax.set_ylabel("Derived − SSPP  [Fe/H]")
    ax.set_title("Bland–Altman Plot")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 3–6. Residual vs various parameters
    # ------------------------------------------------------------------
    residual_panels = [
        ("FEHADOP",   "SSPP [Fe/H]",    "Residual vs Metallicity"),
        ("TEFFADOP",  "T_eff (K)",       "Residual vs Effective Temperature"),
        ("LOGGADOP",  "log g",           "Residual vs Surface Gravity"),
        ("snMedian_r","snMedian_r",      "Residual vs SDSS SNR"),
    ]

    for col, xlabel, title in residual_panels:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df[col], df["residual"], alpha=0.6, s=15)
        ax.axhline(0, color='red', lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residual [Fe/H]  (derived − SSPP)")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 7. Residual vs SNR with running median
    # ------------------------------------------------------------------
    snr    = df["snMedian_r"]
    resid  = df["residual"]
    bins   = np.linspace(snr.min(), snr.max(), 15)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    medians = [
        np.median(resid[(snr >= bins[i]) & (snr < bins[i + 1])].values)
        if np.sum((snr >= bins[i]) & (snr < bins[i + 1])) > 3 else np.nan
        for i in range(len(bins) - 1)
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(snr, resid, alpha=0.4, s=12)
    ax.plot(bin_centers, medians, color='red', lw=2, label='Running median')
    ax.axhline(0, color='black', ls='--', lw=1)
    ax.set_xlabel("SNR (snMedian_r)")
    ax.set_ylabel("Residual [Fe/H]")
    ax.set_title("Metallicity Residual vs SNR")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 8–9. Chi-square distributions
    # ------------------------------------------------------------------
    for col, xlabel, title in [
        ("chi2",     "χ²",          "Chi-square Distribution"),
        ("chi2_red", "Reduced χ²",  "Reduced Chi-square Distribution"),
    ]:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=30, edgecolor='white', lw=0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 10–12. Metallicity and residual histograms
    # ------------------------------------------------------------------
    hist_panels = [
        ("feh_derived", "Derived [Fe/H]", "Derived Metallicity Distribution"),
        ("FEHADOP",     "SSPP [Fe/H]",    "SSPP Metallicity Distribution"),
        ("residual",    "Residual [Fe/H]","Residual Distribution"),
    ]

    for col, xlabel, title in hist_panels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=30, edgecolor='white', lw=0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
