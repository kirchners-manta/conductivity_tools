#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def read_ccf_file(path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Read a CCF CSV file of the form:

    #Correlation Depth / fs;  CCF(...) / ...;  Integral / ...;  CCF(...) / ...;  Integral / ...; ...

    Returns:
        tau      : (n_points,) array of correlation depths
        ccf_vals : (n_points, n_ccf_cols) array of CCF values (only the CCF columns)
        header   : original header line (starting with '#'), preserved for re-use
    """
    header_line = ""
    with path.open("r", encoding="utf-8") as f:
        # Read header line (first non-empty line starting with '#')
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                header_line = line.rstrip("\n")
                break

        # Now read the rest as data lines
        data_lines: List[str] = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lstrip().startswith("#"):
                # Just in case there are more comment lines
                continue
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No data found in file {path}")

    # Parse data into arrays
    tau_list: List[float] = []
    ccf_rows: List[List[float]] = []

    for line in data_lines:
        parts = [p.strip() for p in line.split(";") if p.strip() != ""]
        # First column: tau
        tau_val = float(parts[0])
        tau_list.append(tau_val)

        # Remaining columns: alternating CCF, Integral, CCF, Integral, ...
        # We want only the CCF columns: indices 1,3,5,...
        row_ccf: List[float] = []
        # start from index 1, step 2
        for i in range(1, len(parts), 2):
            row_ccf.append(float(parts[i]))
        ccf_rows.append(row_ccf)

    tau = np.array(tau_list, dtype=float)
    ccf_vals = np.array(ccf_rows, dtype=float)  # shape (n_points, n_ccf_cols)

    return tau, ccf_vals, header_line


def check_tau_compatible(tau_ref: np.ndarray, tau: np.ndarray, path: Path) -> None:
    """Raise if tau grids are not identical (within tolerance)."""
    if tau_ref.shape != tau.shape:
        raise ValueError(
            f"Incompatible tau length between files: reference has {tau_ref.size}, "
            f"{path} has {tau.size}"
        )
    if not np.allclose(tau_ref, tau, rtol=1e-12, atol=1e-15):
        raise ValueError(
            f"Incompatible tau values between files: {path} differs from reference grid."
        )


def average_ccf_across_replicas(
    files: List[Path],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Given a list of CCF files (same type, different replicas),
    compute the unweighted average of CCF columns and return:

        tau        : (n_points,)
        ccf_avg    : (n_points, n_ccf_cols)
        header_line: header from the first file
    """
    if not files:
        raise ValueError("No files provided to average_ccf_across_replicas.")

    tau_ref, ccf_ref, header_line = read_ccf_file(files[0])
    sum_ccf = np.array(ccf_ref, dtype=float)
    count = 1

    for path in files[1:]:
        tau, ccf_vals, _ = read_ccf_file(path)
        check_tau_compatible(tau_ref, tau, path)

        if ccf_vals.shape != sum_ccf.shape:
            raise ValueError(
                f"Incompatible CCF shape in file {path}: "
                f"expected {sum_ccf.shape}, got {ccf_vals.shape}"
            )

        sum_ccf += ccf_vals
        count += 1

    ccf_avg = sum_ccf / float(count)
    return tau_ref, ccf_avg, header_line


def compute_trapezoidal_integrals(
    tau: np.ndarray, ccf: np.ndarray
) -> np.ndarray:
    """
    Given:
        tau : (n_points,)  correlation depths (fs)
        ccf : (n_points, n_ccf_cols) average CCF values

    Return:
        integrals : (n_points, n_ccf_cols) cumulative integrals using trapezoidal rule:

            I[0] = 0
            I[k] = I[k-1] + 0.5 * (f[k-1] + f[k]) * (tau[k] - tau[k-1])
    """
    n_points, n_cols = ccf.shape
    integrals = np.zeros_like(ccf, dtype=float)

    for j in range(n_cols):
        # Column j
        f = ccf[:, j]
        I = integrals[:, j]
        I[0] = 0.0
        for k in range(1, n_points):
            dt = tau[k] - tau[k - 1]
            I[k] = I[k - 1] + 0.5 * (f[k - 1] + f[k]) * dt

    return integrals


def write_averaged_file(
    out_path: Path,
    header_line: str,
    tau: np.ndarray,
    ccf_avg: np.ndarray,
    integrals: np.ndarray,
) -> None:
    """
    Write a CSV file with the same structure as the input:

    #Correlation Depth / fs;  CCF(...);  Integral / ...;  CCF(...);  Integral / ...; ...

    using:
      - 'header_line' for the header
      - 'tau' for the first column
      - 'ccf_avg' and 'integrals' interleaved as CCF, Integral, CCF, Integral, ...
    """
    n_points, n_ccf = ccf_avg.shape

    lines: List[str] = []

    # Ensure header starts with '#'
    if not header_line.lstrip().startswith("#"):
        header_line = "#" + header_line.lstrip()
    lines.append(header_line)

    # Data lines
    for i in range(n_points):
        row_vals: List[str] = []

        # Correlation depth / fs
        row_vals.append(f"{tau[i]: .8f}")

        # Interleave CCF and integral
        for j in range(n_ccf):
            row_vals.append(f"{ccf_avg[i, j]: .8f}")
            row_vals.append(f"{integrals[i, j]: .8f}")

        # Join with ";  " to roughly resemble the original style
        line = "; ".join(row_vals)
        lines.append(line)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(root: str = ".") -> None:
    root_path = Path(root).resolve()

    # Replica directories are assumed to be 1..10 under root
    replica_ids = [str(i) for i in range(1, 11)]

    # For each type of file, collect one file per replica (if present)
    def collect_files(pattern: str) -> List[Path]:
        files: List[Path] = []
        for rep in replica_ids:
            gk_dir = root_path / rep / "gk"
            if not gk_dir.is_dir():
                continue
            matches = sorted(gk_dir.glob(pattern))
            if not matches:
                print(f"[WARN] No files matching {pattern} in {gk_dir}")
                continue
            if len(matches) > 1:
                print(
                    f"[WARN] Multiple files matching {pattern} in {gk_dir}, "
                    f"using first: {matches[0].name}"
                )
            files.append(matches[0])
        return files

    # 1) ccf_decomp_self_cross*.csv
    ccf_files = collect_files("ccf_decomp_self_cross*.csv")
    if ccf_files:
        print(f"[INFO] Found {len(ccf_files)} ccf_decomp_self_cross*.csv files.")
        tau_ccf, ccf_avg, header_ccf = average_ccf_across_replicas(ccf_files)
        integrals_ccf = compute_trapezoidal_integrals(tau_ccf, ccf_avg)
        out_ccf = root_path / "ccf_decomp_self_cross_averaged.csv"
        write_averaged_file(out_ccf, header_ccf, tau_ccf, ccf_avg, integrals_ccf)
        print(f"[INFO] Wrote averaged CCF file: {out_ccf}")
    else:
        print("[INFO] No ccf_decomp_self_cross*.csv files found.")

    # 2) charge_current_decomp_self_cross*.csv
    cc_files = collect_files("charge_current_decomp_self_cross*.csv")
    if cc_files:
        print(
            f"[INFO] Found {len(cc_files)} "
            f"charge_current_decomp_self_cross*.csv files."
        )
        tau_cc, cc_avg, header_cc = average_ccf_across_replicas(cc_files)
        integrals_cc = compute_trapezoidal_integrals(tau_cc, cc_avg)
        out_cc = root_path / "charge_current_decomp_self_cross_averaged.csv"
        write_averaged_file(out_cc, header_cc, tau_cc, cc_avg, integrals_cc)
        print(f"[INFO] Wrote averaged charge-current file: {out_cc}")
    else:
        print("[INFO] No charge_current_decomp_self_cross*.csv files found.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [root_dir]", file=sys.stderr)
        sys.exit(1)

    root_dir = sys.argv[1] if len(sys.argv) == 2 else "."
    main(root_dir)

