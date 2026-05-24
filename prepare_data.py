#!/usr/bin/env python3
"""
prepare_data.py — Convert raw scraped data into the compact files the
Streamlit dashboard reads.

Inputs (defaults, override with flags):
  --combined  uo_schedules/csv/combined.csv   (from parse_schedules.py)
  --catalog   catalog.csv                      (from scrape_catalog.py)

Outputs (written to ./data):
  data/enrollment.parquet     section-level rows, only the columns the app needs
  data/catalog.csv            copied verbatim

Run this once locally before deploying the app, and again whenever you
re-scrape new data. The two files in ./data are what gets committed to
GitHub and read by Streamlit Cloud.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


# Columns to keep from combined.csv; everything else is dropped.
KEEP_COLS = [
    "Term_Code", "Subject", "Course_Number", "Course_Name",
    "CRN", "Format", "Enrollment",
]


def build_enrollment(combined_csv: Path, out_parquet: Path) -> None:
    print(f"Reading {combined_csv} ...", flush=True)
    df = pd.read_csv(
        combined_csv,
        dtype={
            "Term_Code": str,
            "Subject": str,
            "Course_Number": str,
            "Course_Name": str,
            "CRN": str,
            "Format": str,
        },
        low_memory=False,
    )
    print(f"  read {len(df):,} rows, {len(df.columns)} columns", flush=True)

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: combined.csv is missing expected columns: {missing}")

    # Enrollment to a sane numeric dtype; bad values → 0.
    df["Enrollment"] = (
        pd.to_numeric(df["Enrollment"], errors="coerce")
          .fillna(0).clip(lower=0).astype("int32")
    )

    # Normalize Format. Blank / NaN means the row was a primary
    # section with no Format label; we treat those as lectures.
    fmt = df["Format"].fillna("").str.strip()
    fmt = fmt.replace({"": "Lecture", "Dis": "Discussion"})
    df["Format_Norm"] = fmt

    # Section order within (Term_Code, Subject, Course_Number),
    # preserving the input row order. Used for "first section only" filter.
    df["section_order"] = (
        df.groupby(["Term_Code", "Subject", "Course_Number"], sort=False)
          .cumcount().astype("int16")
    )

    # Trim columns.
    out = df[KEEP_COLS + ["Format_Norm", "section_order"]].copy()

    # Categoricals shrink the parquet substantially.
    for c in ("Subject", "Format", "Format_Norm"):
        out[c] = out[c].astype("category")

    print(f"Writing {out_parquet} ...", flush=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False, compression="snappy")
    size_mb = out_parquet.stat().st_size / 1e6
    print(f"  wrote {len(out):,} rows, {size_mb:.1f} MB", flush=True)

    # Quick sanity report.
    print()
    print("Format_Norm counts:")
    print(out["Format_Norm"].value_counts().to_string())
    print()
    print(f"Terms: {out['Term_Code'].nunique()}  "
          f"(min={out['Term_Code'].min()}, max={out['Term_Code'].max()})")
    print(f"Subjects: {out['Subject'].nunique()}")
    print(f"Distinct (Subject, Course_Number): "
          f"{out.groupby(['Subject', 'Course_Number']).ngroups:,}")


def copy_catalog(catalog_csv: Path, out_csv: Path) -> None:
    print(f"\nCopying {catalog_csv} → {out_csv} ...", flush=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(catalog_csv, out_csv)
    print(f"  copied {out_csv.stat().st_size / 1e6:.2f} MB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined", type=Path,
                    default=Path("uo_schedules/csv/combined.csv"),
                    help="Path to combined.csv from parse_schedules.py")
    ap.add_argument("--catalog", type=Path, default=Path("catalog.csv"),
                    help="Path to the catalog CSV from scrape_catalog.py")
    ap.add_argument("--out-dir", type=Path, default=Path("data"),
                    help="Output directory (committed to GitHub)")
    args = ap.parse_args()

    if not args.combined.exists():
        sys.exit(f"ERROR: {args.combined} not found. "
                 f"Pass --combined /path/to/combined.csv")
    if not args.catalog.exists():
        sys.exit(f"ERROR: {args.catalog} not found. "
                 f"Pass --catalog /path/to/catalog.csv")

    build_enrollment(args.combined, args.out_dir / "enrollment.parquet")
    copy_catalog(args.catalog, args.out_dir / "catalog.csv")
    print("\nDone. Commit the files in ./data/ and push to GitHub.")


if __name__ == "__main__":
    main()
