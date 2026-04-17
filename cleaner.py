"""
cleaner.py
==========
Canonicalises star identifiers and deduplicates both the spectrum index
and the metadata catalog before the chi-square fitting step.

Produces:
  index_clean.csv   — deduplicated index of exported spectra
  catalog_clean.csv — deduplicated metadata (one row per unique star,
                      keeping the highest-SNR observation)

Why deduplication is needed
---------------------------
The SDSS CasJobs query may return multiple observations of the same star
observed on different plates or MJDs. The spectrum_export step retains all
of them. Here we keep the single best observation per star, defined as the
one with the highest snMedian_r, which is the scientifically correct
choice for maximising fit quality.

Usage
-----
    python cleaner.py --index outputs/index.csv \
                      --meta  outputs/csv/catalog_fit_stars.csv \
                      --outdir outputs/
"""

import argparse
import pandas as pd
from pathlib import Path
import yaml


def canon_id(x: str) -> str:
    """Normalise plate-mjd-fiber string to int-int-int format."""
    p, m, f = x.strip().split('-')
    return f"{int(p)}-{int(m)}-{int(float(f))}"


def parse_args() -> argparse.Namespace:
    _cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    out  = Path(_cfg["output_dir"])

    parser = argparse.ArgumentParser(description="Clean and deduplicate index + catalog.")
    parser.add_argument("--index",  type=Path, default=out / "index.csv")
    parser.add_argument("--meta",   type=Path, default=out / "csv" / "catalog_fit_stars.csv")
    parser.add_argument("--outdir", type=Path, default=out)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Clean index
    # ------------------------------------------------------------------
    index = pd.read_csv(args.index)
    index['id'] = (
        index['id']
        .astype(str)
        .str.strip()
        .str.replace('.txt', '', regex=False)
        .apply(canon_id)
    )
    index_clean = index.drop_duplicates(subset='id', keep='first')

    # ------------------------------------------------------------------
    # Clean metadata catalog
    # ------------------------------------------------------------------
    meta = pd.read_csv(args.meta)
    meta['plate'] = meta['plate'].astype(int)
    meta['mjd']   = meta['mjd'].astype(int)
    meta['fiber'] = meta['fiber'].astype(float).astype(int)
    meta['id']    = (
        meta['plate'].astype(str) + '-' +
        meta['mjd'].astype(str)   + '-' +
        meta['fiber'].astype(str)
    )

    # Keep highest-SNR observation per unique star
    meta_clean = (
        meta
        .sort_values('snMedian_r', ascending=False)
        .drop_duplicates(subset='id', keep='first')
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    common = set(index_clean['id']) & set(meta_clean['id'])
    print(f"Index unique stars : {len(index_clean)}")
    print(f"Catalog unique stars: {len(meta_clean)}")
    print(f"Stars in common    : {len(common)}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    index_clean.to_csv(args.outdir / "index_clean.csv",   index=False)
    meta_clean.to_csv( args.outdir / "catalog_clean.csv", index=False)

    print(f"\nSaved:")
    print(f"  {args.outdir / 'index_clean.csv'}")
    print(f"  {args.outdir / 'catalog_clean.csv'}")


if __name__ == "__main__":
    main()
