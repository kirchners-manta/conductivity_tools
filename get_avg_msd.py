#!/usr/bin/env python3
"""
Usage:
    python average_csv.py file1.csv file2.csv [file3.csv ...]

Reads multiple semicolon-separated CSV files of the form:

# tau / ps;  MSD / pm^2;  Derivative
 0.000000; 0.000000; 2387.538107
 30.000000; 71626.143214; 1568.615499
 ...

Checks that the tau column is identical in all files,
then computes:
  - the unweighted average of the MSD column
  - the standard error of the mean MSD from the spread between files
  - the derivative recomputed from the averaged MSD

Output columns:
  1. tau
  2. averaged MSD
  3. derivative of averaged MSD
  4. uncertainty of averaged MSD (standard error)
"""

import sys
import math
from typing import List, Tuple, Optional

import numpy as np


def read_header_and_ncols(path: str) -> Tuple[Optional[List[str]], int]:
    """
    Read the first header line starting with '#' (if present) and determine
    the number of columns from the first non-empty non-comment data line.
    """
    header_parts: Optional[List[str]] = None
    ncols: Optional[int] = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.lstrip().startswith("#"):
                    if ";" in stripped and header_parts is None:
                        without_hash = stripped.lstrip()[1:].strip()
                        header_parts = [p.strip() for p in without_hash.split(";")]
                    continue

                parts = [p.strip() for p in stripped.split(";")]
                ncols = len(parts)
                break
    except OSError as e:
        raise SystemExit(f"ERROR: Cannot read file '{path}': {e}")

    if ncols is None:
        raise SystemExit(f"ERROR: No data found in file '{path}'")

    return header_parts, ncols


def read_tau_and_msd(path: str) -> Tuple[List[float], List[float]]:
    """
    Read one CSV file and return:
        tau_list,
        msd_list

    Expects at least 2 columns:
      1. tau
      2. MSD
    Any further columns are ignored.
    """
    taus: List[float] = []
    msd: List[float] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()

                if not stripped:
                    continue
                if stripped.lstrip().startswith("#"):
                    continue

                parts = [p.strip() for p in stripped.split(";")]

                if len(parts) < 2:
                    raise ValueError(
                        f"Expected at least 2 columns in {path}, got: {stripped}"
                    )

                try:
                    tau = float(parts[0])
                    msd_val = float(parts[1])
                except ValueError as e:
                    raise ValueError(
                        f"Failed to parse numeric values in {path} line: {stripped}"
                    ) from e

                taus.append(tau)
                msd.append(msd_val)

    except OSError as e:
        raise SystemExit(f"ERROR: Cannot read file '{path}': {e}")

    if not taus:
        raise SystemExit(f"ERROR: No data found in file '{path}'")

    return taus, msd


def sequences_close(
    a: List[float],
    b: List[float],
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> bool:
    """Check if two sequences of floats are (almost) identical."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


def main(argv: List[str]) -> None:
    if len(argv) < 3:
        print(
            f"Usage: {argv[0]} file1.csv file2.csv [file3.csv ...]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    file_paths = argv[1:]

    header_parts_ref, ncols_ref = read_header_and_ncols(file_paths[0])
    if ncols_ref < 2:
        print(
            f"ERROR: File '{file_paths[0]}' must have at least 2 columns.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    for path in file_paths[1:]:
        _header_parts, ncols = read_header_and_ncols(path)
        if ncols < 2:
            print(
                f"ERROR: File '{path}' must have at least 2 columns.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    tau_ref, msd0 = read_tau_and_msd(file_paths[0])
    all_msd = [msd0]

    for path in file_paths[1:]:
        tau, msd = read_tau_and_msd(path)

        if not sequences_close(tau, tau_ref):
            print(
                f"ERROR: tau values in file '{path}' do not match "
                f"the tau values in '{file_paths[0]}'.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        all_msd.append(msd)

    msd_array = np.asarray(all_msd, dtype=float)   # shape: (n_files, n_points)
    tau_array = np.asarray(tau_ref, dtype=float)

    n_files = msd_array.shape[0]

    # Unweighted mean MSD
    mean_msd = np.mean(msd_array, axis=0)

    # Uncertainty from spread between files = standard error of the mean
    if n_files > 1:
        delta_msd = np.std(msd_array, axis=0, ddof=1) / np.sqrt(n_files)
    else:
        delta_msd = np.full_like(mean_msd, np.nan)

    # Recompute derivative from averaged MSD
    if len(tau_array) >= 3:
        derivative = np.gradient(mean_msd, tau_array, edge_order=2)
    elif len(tau_array) == 2:
        derivative = np.gradient(mean_msd, tau_array, edge_order=1)
    else:
        derivative = np.full_like(mean_msd, np.nan)

    # Header
    tau_header = "tau"
    msd_header = "MSD_mean"
    deriv_header = "Derivative"
    delta_header = "delta_MSD"

    if header_parts_ref is not None and len(header_parts_ref) >= 2:
        tau_header = header_parts_ref[0]
        msd_header = header_parts_ref[1]
        delta_header = f"delta_{header_parts_ref[1]}"

    print(f"# {tau_header};  {msd_header};  {deriv_header};  {delta_header}")

    # Data rows
    for tau, msd_val, deriv_val, delta_val in zip(
        tau_array, mean_msd, derivative, delta_msd
    ):
        print(
            f" {tau:0.6f};"
            f" {msd_val:0.6f};"
            f" {deriv_val:0.6f};"
            f" {delta_val:0.6f}"
        )


if __name__ == "__main__":
    main(sys.argv)

