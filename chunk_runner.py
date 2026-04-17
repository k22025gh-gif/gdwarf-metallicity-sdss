"""
chunk_runner.py
===============
Batch runner for the SDSS G-star preprocessing pipeline.

Iterates over all *.csv chunk files in the input directory and calls
process_many() on each one, writing numbered output CSVs.

Usage
-----
    # Use paths from config.yaml:
    python chunk_runner.py

    # Override input/output directories at the command line:
    python chunk_runner.py --input data/CSV_chunks --output outputs/csv

    # Process a single chunk file:
    python chunk_runner.py --input data/CSV_chunks/chunk_01.csv --output outputs/csv
"""

import argparse
import yaml
from pathlib import Path

from preprocess_sdss import process_many


def load_config(path: Path = Path("config.yaml")) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Batch-process SDSS G-star spectral chunks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(cfg["input_dir"]),
        help="Directory containing CSV chunk files, or a single CSV file. "
             f"(default from config.yaml: {cfg['input_dir']})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for results CSVs. "
             "Overrides output_dir in config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input

    # Resolve chunk file list
    if input_path.is_dir():
        chunk_files = sorted(input_path.glob("*.csv"))
        if not chunk_files:
            raise FileNotFoundError(f"No CSV files found in {input_path}")
    elif input_path.is_file():
        chunk_files = [input_path]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Found {len(chunk_files)} chunk file(s) to process.\n")

    for i, chunk in enumerate(chunk_files, 1):
        out_name = f"results_chunk_{i:02d}.csv"
        print(f"=== Chunk {i}/{len(chunk_files)}: {chunk.name} → {out_name} ===")
        process_many(input_csv=str(chunk), output_csv=out_name)

    print("\nAll chunks complete.")


if __name__ == "__main__":
    main()
