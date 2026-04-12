#!/usr/bin/env python3
"""
Usage:
    python average_csv.py [options] file1.csv file2.csv [file3.csv ...]

Reads multiple semicolon-separated CSV files of the form:

# tau [ps];  total [ps*S/m];  variance [ps^2*S^2/m^2]
 0.000000;  0.000000;  0.000000
 30.000000;  192.166107;  23980.164852
 ...

Checks that the tau (first column) is identical in all files,
then computes either the inverse-variance-weighted average
(default) or the unweighted average of the data column(s) and the
variance of that average at each tau.

Default behavior:
    All data/variance pairs are processed:
        (2nd, 3rd), (4th, 5th), (6th, 7th), ...
    i.e. columns 2,4,6,... are treated as data columns and the
    immediately following columns 3,5,7,... as their variances.

Optional behavior:
    You can specify exactly one data/variance pair with
        --data-col N --var-col M
    where the column numbers are 1-based.

By default, chi2 scaling is enabled for weighted averages:
    With chi2 scaling, the variance of the weighted mean at each tau
    is multiplied by the reduced chi^2 computed from the input values
    and variances. This inflates the error when the observed scatter
    exceeds what the variances would predict.
"""

import sys
import math
import argparse
from typing import List, Tuple, Optional


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


def read_csv_columns(
    path: str,
    data_cols_0based: List[int],
    var_cols_0based: List[int],
) -> Tuple[List[float], List[List[float]], List[List[float]]]:
    """
    Read one CSV file and return:
        tau_list,
        data_columns_as_lists,
        variance_columns_as_lists

    data_cols_0based and var_cols_0based are 0-based column indices.
    """
    taus: List[float] = []
    vals_cols: List[List[float]] = [[] for _ in data_cols_0based]
    vars_cols: List[List[float]] = [[] for _ in var_cols_0based]

    needed_cols = max([0] + data_cols_0based + var_cols_0based) + 1

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()

                if not stripped:
                    continue
                if stripped.lstrip().startswith("#"):
                    continue

                parts = [p.strip() for p in stripped.split(";")]

                if len(parts) < needed_cols:
                    raise ValueError(
                        f"Expected at least {needed_cols} columns in {path}, got: {stripped}"
                    )

                try:
                    tau = float(parts[0])
                    row_vals = [float(parts[idx]) for idx in data_cols_0based]
                    row_vars = [float(parts[idx]) for idx in var_cols_0based]
                except ValueError as e:
                    raise ValueError(
                        f"Failed to parse numeric values in {path} line: {stripped}"
                    ) from e

                taus.append(tau)
                for k, v in enumerate(row_vals):
                    vals_cols[k].append(v)
                for k, v in enumerate(row_vars):
                    vars_cols[k].append(v)

    except OSError as e:
        raise SystemExit(f"ERROR: Cannot read file '{path}': {e}")

    if not taus:
        raise SystemExit(f"ERROR: No data found in file '{path}'")

    return taus, vals_cols, vars_cols


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


def weighted_or_unweighted_mean_and_variance(
    yj: List[float],
    vj: List[float],
    method: str,
    chi2_scale: bool,
) -> Tuple[float, float]:
    """
    Compute the mean and variance of the mean for one tau point.
    """
    if any(v < 0 for v in vj):
        raise ValueError("Negative variance encountered.")

    if method == "weighted":
        positive_indices = [i for i, v in enumerate(vj) if v > 0.0]

        if not positive_indices:
            mean_y = sum(yj) / len(yj)
            mean_v = 0.0
        else:
            weights = [1.0 / vj[i] for i in positive_indices]
            weighted_vals = [
                weights[k] * yj[positive_indices[k]]
                for k in range(len(positive_indices))
            ]
            sum_w = sum(weights)

            if sum_w == 0.0:
                mean_y = sum(yj) / len(yj)
                mean_v = 0.0
            else:
                mean_y = sum(weighted_vals) / sum_w
                mean_v = 1.0 / sum_w

                if chi2_scale and len(positive_indices) > 1:
                    chi2 = 0.0
                    for i in positive_indices:
                        resid = yj[i] - mean_y
                        chi2 += (resid * resid) / vj[i]

                    dof = len(positive_indices) - 1
                    if dof > 0:
                        chi2_red = chi2 / dof
                        if chi2_red > 0.0:
                            mean_v *= chi2_red
    else:
        n_files = len(yj)
        mean_y = sum(yj) / n_files
        mean_v = sum(vj) / (n_files * n_files)

    return mean_y, mean_v


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Average semicolon-separated CSV files over a common tau axis.\n"
            "Default: process all alternating data/variance column pairs "
            "(2,3), (4,5), (6,7), ...\n"
            "Optionally, choose one explicit pair via --data-col and --var-col.\n"
            "Weighted averaging uses chi2 scaling by default."
        )
    )
    parser.add_argument(
        "files",
        metavar="file",
        nargs="+",
        help="Input CSV files.",
    )
    parser.add_argument(
        "--data-col",
        type=int,
        help=(
            "1-based column number of the data column. "
            "Must be used together with --var-col."
        ),
    )
    parser.add_argument(
        "--var-col",
        type=int,
        help=(
            "1-based column number of the variance column. "
            "Must be used together with --data-col."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["weighted", "unweighted"],
        default="weighted",
        help=(
            "Averaging method to use: 'weighted' for inverse-variance weighting "
            "(default), or 'unweighted' for a simple mean."
        ),
    )
    parser.add_argument(
        "--no-chi2-scale",
        action="store_true",
        help=(
            "Disable reduced-chi^2 scaling of the weighted variance. "
            "By default, chi2 scaling is enabled."
        ),
    )

    args = parser.parse_args(argv[1:])

    if len(args.files) < 2:
        print("ERROR: Please supply at least two CSV files.", file=sys.stderr)
        raise SystemExit(1)

    if (args.data_col is None) != (args.var_col is None):
        print(
            "ERROR: --data-col and --var-col must either both be given or both be omitted.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    file_paths = args.files
    chi2_scale = not args.no_chi2_scale

    # Inspect first file
    header_parts_ref, ncols_ref = read_header_and_ncols(file_paths[0])

    # Check all files have same number of columns
    headers_all = [header_parts_ref]
    for path in file_paths[1:]:
        header_parts, ncols = read_header_and_ncols(path)
        headers_all.append(header_parts)
        if ncols != ncols_ref:
            print(
                f"ERROR: File '{path}' has {ncols} columns, but '{file_paths[0]}' has {ncols_ref}.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    # Determine which columns to process
    if args.data_col is not None:
        data_cols_1based = [args.data_col]
        var_cols_1based = [args.var_col]

        if args.data_col < 1 or args.data_col > ncols_ref:
            print(
                f"ERROR: --data-col must be between 1 and {ncols_ref}, got {args.data_col}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if args.var_col < 1 or args.var_col > ncols_ref:
            print(
                f"ERROR: --var-col must be between 1 and {ncols_ref}, got {args.var_col}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if args.data_col == 1 or args.var_col == 1:
            print(
                "ERROR: Column 1 is reserved for tau/time and cannot be used as data or variance.",
                file=sys.stderr,
            )
            raise SystemExit(1)
    else:
        # Default: every data column 2,4,6,... with variance column immediately after
        if ncols_ref < 3:
            print(
                f"ERROR: Need at least 3 columns, got {ncols_ref}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if (ncols_ref - 1) % 2 != 0:
            print(
                f"ERROR: Default mode expects columns as tau + alternating data/variance pairs, "
                f"but file has {ncols_ref} columns.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        data_cols_1based = list(range(2, ncols_ref, 2))
        var_cols_1based = list(range(3, ncols_ref + 1, 2))

    data_cols_0based = [c - 1 for c in data_cols_1based]
    var_cols_0based = [c - 1 for c in var_cols_1based]

    # Read first file
    tau_ref, vals0_cols, vars0_cols = read_csv_columns(
        file_paths[0],
        data_cols_0based=data_cols_0based,
        var_cols_0based=var_cols_0based,
    )

    all_vals_cols = [[vals0_cols[k]] for k in range(len(vals0_cols))]
    all_vars_cols = [[vars0_cols[k]] for k in range(len(vars0_cols))]

    # Read remaining files and check tau columns match
    for path in file_paths[1:]:
        tau, vals_cols, vars_cols = read_csv_columns(
            path,
            data_cols_0based=data_cols_0based,
            var_cols_0based=var_cols_0based,
        )

        if not sequences_close(tau, tau_ref):
            print(
                f"ERROR: tau (x data) in file '{path}' does not match "
                f"the tau values in '{file_paths[0]}'.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        for k in range(len(vals_cols)):
            all_vals_cols[k].append(vals_cols[k])
            all_vars_cols[k].append(vars_cols[k])

    n_files = len(file_paths)
    n_points = len(tau_ref)
    n_pairs = len(data_cols_0based)

    # Compute results for each pair
    mean_vals_cols: List[List[float]] = [[] for _ in range(n_pairs)]
    mean_vars_cols: List[List[float]] = [[] for _ in range(n_pairs)]

    for pair_idx in range(n_pairs):
        for j in range(n_points):
            yj = [all_vals_cols[pair_idx][i][j] for i in range(n_files)]
            vj = [all_vars_cols[pair_idx][i][j] for i in range(n_files)]

            try:
                mean_y, mean_v = weighted_or_unweighted_mean_and_variance(
                    yj=yj,
                    vj=vj,
                    method=args.method,
                    chi2_scale=chi2_scale,
                )
            except ValueError:
                print(
                    f"ERROR: Negative variance encountered at tau={tau_ref[j]} "
                    f"for data column {data_cols_1based[pair_idx]}.",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            mean_vals_cols[pair_idx].append(mean_y)
            mean_vars_cols[pair_idx].append(mean_v)

    # Build output header
    out_headers = []
    if header_parts_ref is not None and len(header_parts_ref) >= ncols_ref:
        for pair_idx, (dcol1, vcol1) in enumerate(zip(data_cols_1based, var_cols_1based)):
            data_name = header_parts_ref[dcol1 - 1]
            var_name = header_parts_ref[vcol1 - 1]
            out_headers.append(data_name)
            out_headers.append(var_name)
    else:
        for pair_idx, (dcol1, vcol1) in enumerate(zip(data_cols_1based, var_cols_1based), start=1):
            out_headers.append(f"{args.method}_mean_col{dcol1}")
            suffix = "_chi2scaled" if (args.method == "weighted" and chi2_scale) else ""
            out_headers.append(f"variance_{args.method}_col{dcol1}{suffix}")

    # Print output
    header = "# tau"
    if header_parts_ref is not None and len(header_parts_ref) >= 1:
        header = f"# {header_parts_ref[0]}"
    for h in out_headers:
        header += f";  {h}"
    print(header)

    for row_idx, tau in enumerate(tau_ref):
        row_parts = [f" {tau:0.6f}"]
        for pair_idx in range(n_pairs):
            row_parts.append(f" {mean_vals_cols[pair_idx][row_idx]:0.6f}")
            row_parts.append(f" {mean_vars_cols[pair_idx][row_idx]:0.6f}")
        print("; ".join(row_parts))


if __name__ == "__main__":
    main(sys.argv)

