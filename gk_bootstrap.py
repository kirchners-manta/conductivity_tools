#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2.0"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def format_pretty_table(df: pd.DataFrame, sep: str = ";") -> str:
    """
    Return DataFrame 'df' as a formatted string with:
      - semicolon separator
      - aligned columns:
          * floats: fixed decimals, right-aligned
          * ints: right-aligned
          * strings: left-aligned
    """
    df = df.copy()
    col_names = list(df.columns)

    col_strings = []
    col_widths = []
    col_align = []

    for col in col_names:
        s = df[col]

        if pd.api.types.is_integer_dtype(s):
            vals = [str(int(v)) for v in s]
            align = "right"
        elif pd.api.types.is_float_dtype(s):
            vals = [f"{float(v): .8f}" for v in s]
            align = "right"
        else:
            vals = [str(v) for v in s]
            align = "left"

        width = max(len(str(col)), max(len(v) for v in vals) if vals else 0)

        col_strings.append(vals)
        col_widths.append(width)
        col_align.append(align)

    lines = []
    sep_str = f"{sep} "

    header_cells = []
    for name, width in zip(col_names, col_widths):
        header_cells.append(name.ljust(width))
    lines.append(sep_str.join(header_cells))

    n_rows = len(df)
    for i in range(n_rows):
        row_cells = []
        for j, _ in enumerate(col_names):
            val_str = col_strings[j][i]
            width = col_widths[j]
            align = col_align[j]
            cell = val_str.rjust(width) if align == "right" else val_str.ljust(width)
            row_cells.append(cell)
        lines.append(sep_str.join(row_cells))

    return "\n".join(lines)


def write_machine_and_pretty(df: pd.DataFrame, path: Path, sep: str = ";") -> None:
    """
    Write:
      - machine-readable CSV to 'path'
      - pretty aligned text table to '<stem>.pretty.txt'
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, index=False, float_format="%.8f")

    pretty_path = path.with_suffix(".pretty.txt")
    pretty_path.write_text(format_pretty_table(df, sep=sep) + "\n", encoding="utf-8")

    logging.info("Wrote machine-readable table: %s", path)
    logging.info("Wrote pretty text table:      %s", pretty_path)


def write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Wrote metadata:               %s", path)


def print_progress(prefix: str, current: int, total: int) -> None:
    """Simple in-place progress indicator."""
    if total <= 0:
        return
    pct = 100.0 * current / total
    print(f"\r{prefix}: {current}/{total} ({pct:5.1f}%)", end="", flush=True)
    if current >= total:
        print()


def cumulative_trapezoid(time: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with I[0] = 0."""
    return np.cumsum(
        np.concatenate(([0.0], np.diff(time) * (y[1:] + y[:-1]) / 2.0))
    )


def compute_window_slope(time: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return signed slope and absolute slope for a window."""
    if len(time) < 2:
        return np.nan, np.nan
    dt = float(time[-1] - time[0])
    if dt <= 0.0:
        return np.nan, np.nan
    slope = float((y[-1] - y[0]) / dt)
    return slope, abs(slope)


def find_plateau_region(
    data: pd.DataFrame,
    tol: float,
    nslice: int = 10,
    incr: float = 0.01,
    min_time: float | None = None,
    min_points: int = 2,
) -> Tuple[int, int]:
    """
    Find a plateau region in an Integral(t) curve, using a strategy adapted
    from msdiff's find_linear_region.

    If min_time is given, only plateau windows whose start time is >= min_time
    are allowed. The integral itself is still assumed to have been computed
    over the full time range.

    Returns
    -------
    (firststep, laststep) : tuple[int, int]
        1-based indices of the first and last point of the plateau.
        Returns (-1, -1) if no plateau is found.
    """
    ndata = len(data.iloc[:, 0])
    if ndata < max(2, min_points):
        return -1, -1

    int_list: List[List[float]] = []
    time_all = data.iloc[:, 0].to_numpy(dtype=float)

    for n in range(2 * nslice - 1):
        linear_region = True
        t1 = ndata - int((n + 2) / 2 * ndata / nslice) + 1
        t2 = ndata - int(n / 2 * ndata / nslice) - 1

        if t1 < 1:
            t1 = 1
        if t2 > ndata:
            t2 = ndata
        if t2 <= t1:
            continue

        while linear_region:
            if min_time is not None and time_all[t1 - 1] < min_time:
                break

            region = data.iloc[t1 - 1 : t2].copy()
            if len(region) < max(2, min_points):
                break

            region.iloc[:, 0] = region.iloc[:, 0] - region.iloc[0, 0]
            region.iloc[:, 1] = region.iloc[:, 1] - region.iloc[0, 1]

            t = region.iloc[:, 0].to_numpy(dtype=float)
            y = region.iloc[:, 1].to_numpy(dtype=float)

            if len(t) < max(2, min_points):
                break
            if t[-1] <= 0.0:
                break

            slope = (y[-1] - y[0]) / (t[-1] - t[0])

            if np.isnan(slope):
                break

            if abs(slope) > tol:
                linear_region = False
            else:
                npoints = t2 - t1 + 1
                int_list.append([t1, t2, npoints, abs(slope)])

                t1_new = t1 - int(ndata * incr)
                if t1_new < 1:
                    linear_region = False
                else:
                    t1 = t1_new

    if not int_list:
        return -1, -1

    linreg_data = pd.DataFrame(
        int_list, columns=["t1", "t2", "npoints", "slope_abs"]
    )

    linreg_final = linreg_data.sort_values(
        ["npoints", "slope_abs"], ascending=[False, True]
    ).iloc[0]

    firststep = int(linreg_final["t1"])
    laststep = int(linreg_final["t2"])
    return firststep, laststep


def find_plateau_with_auto_tol(
    data: pd.DataFrame,
    tol_start: float,
    nslice: int,
    incr: float,
    min_time: float | None = None,
    min_points: int = 2,
    max_tol: float = 1e-2,
) -> Tuple[int, int, float]:
    """
    Try to find a plateau. If none is found, increase tolerance stepwise.

    Strategy:
      - Work decade by decade.
      - Within a decade [1eN, 1e(N+1)], increase in steps of 0.5eN.
      - Example:
            1e-6 -> 1.5e-6 -> 2e-6 -> ... -> 1e-5
            then 1.5e-5 -> 2e-5 -> ... -> 1e-4
            then 1.5e-4 -> 2e-4 -> ...
      - Stop when tolerance exceeds max_tol.
    """
    if tol_start <= 0:
        raise ValueError("tol_start must be > 0")
    if max_tol < tol_start:
        raise ValueError("max_tol must be >= tol_start")

    tol = tol_start

    while tol <= max_tol + 1e-15:
        firststep, laststep = find_plateau_region(
            data,
            tol,
            nslice=nslice,
            incr=incr,
            min_time=min_time,
            min_points=min_points,
        )
        if firststep != -1:
            return firststep, laststep, tol

        exponent = math.floor(math.log10(tol))
        step = 0.5 * (10.0 ** exponent)
        tol = round(tol + step, 15)

    return -1, -1, tol


def extract_plateau_stats(
    time: np.ndarray,
    avg_ccf: np.ndarray,
    integral: np.ndarray,
    firststep: int,
    laststep: int,
) -> dict:
    """Compute plateau statistics for a selected window."""
    tau_window = time[firststep - 1 : laststep]
    integral_window = integral[firststep - 1 : laststep]
    ccf_window = avg_ccf[firststep - 1 : laststep]

    slope, slope_abs = compute_window_slope(tau_window, integral_window)

    return {
        "tau_start": float(tau_window[0]),
        "tau_end": float(tau_window[-1]),
        "npoints": int(len(tau_window)),
        "integral_plateau_mean": float(np.mean(integral_window)),
        "integral_plateau_std": float(np.std(integral_window, ddof=1)) if len(integral_window) > 1 else 0.0,
        "ccf_plateau_mean": float(np.mean(ccf_window)),
        "slope": slope,
        "slope_abs": slope_abs,
    }


def make_plateau_result_row(
    contribution: str,
    col_1based: int,
    status: str,
    plateau_stats: dict | None,
    *,
    tol_start: float,
    tol_used: float,
    nslice: int,
    incr: float,
    min_tau: float | None,
    nrep: int,
) -> dict:
    base = {
        "contribution": contribution,
        "col": col_1based,
        "status": status,
        "plateau_method": "longest_window_under_slope_threshold",
        "tau_start": np.nan,
        "tau_end": np.nan,
        "npoints": -1,
        "integral_plateau_mean": np.nan,
        "integral_plateau_std": np.nan,
        "ccf_plateau_mean": np.nan,
        "slope": np.nan,
        "slope_abs": np.nan,
        "tol_start": tol_start,
        "tol_used": tol_used,
        "nslice": nslice,
        "incr": incr,
        "min_tau": min_tau if min_tau is not None else np.nan,
        "nrep": nrep,
    }
    if plateau_stats is not None:
        base.update(plateau_stats)
    return base


def make_bootstrap_result_row(
    contribution: str,
    col_1based: int,
    status: str,
    *,
    nboot: int,
    nboot_valid: int,
    integral_boot_mean: float,
    integral_boot_std: float,
    tau_start_boot_min: float,
    tau_start_boot_max: float,
    tau_end_boot_min: float,
    tau_end_boot_max: float,
    tol_start: float,
    tol_used: float,
    nslice: int,
    incr: float,
    min_tau: float | None,
    nrep: int,
    boot_ccf_mean_factor: float,
    seed: int,
) -> dict:
    return {
        "contribution": contribution,
        "col": col_1based,
        "status": status,
        "nboot": nboot,
        "nboot_valid": nboot_valid,
        "integral_boot_mean": integral_boot_mean,
        "integral_boot_std": integral_boot_std,
        "tau_start_boot_min": tau_start_boot_min,
        "tau_start_boot_max": tau_start_boot_max,
        "tau_end_boot_min": tau_end_boot_min,
        "tau_end_boot_max": tau_end_boot_max,
        "tol_start": tol_start,
        "tol_used": tol_used,
        "nslice": nslice,
        "incr": incr,
        "min_tau": min_tau if min_tau is not None else np.nan,
        "nrep": nrep,
        "boot_ccf_mean_factor": boot_ccf_mean_factor,
        "seed": seed,
    }


def analyze_single_column(
    replicas: np.ndarray,
    time: np.ndarray,
    tol_start: float,
    nslice: int,
    incr: float,
    nboot: int,
    min_tau: float | None = None,
    min_points: int = 2,
    boot_ccf_mean_factor: float = 5.0,
    rng: np.random.Generator | None = None,
    progress_prefix: str | None = None,
) -> tuple[dict, dict, list[dict]]:
    """
    Analyze one CCF column across replicas.

    min_tau refers to the actual tau/time value, not the index.
    """
    nrep = replicas.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)

    avg_ccf = replicas.mean(axis=0)
    integral = cumulative_trapezoid(time, avg_ccf)
    data = pd.DataFrame({"time": time, "integral": integral})

    firststep, laststep, tol_used = find_plateau_with_auto_tol(
        data=data,
        tol_start=tol_start,
        nslice=nslice,
        incr=incr,
        min_time=min_tau,
        min_points=min_points,
    )

    if firststep == -1:
        plateau_stats = None
        plateau_status = "no_plateau"
    else:
        plateau_stats = extract_plateau_stats(
            time=time,
            avg_ccf=avg_ccf,
            integral=integral,
            firststep=firststep,
            laststep=laststep,
        )
        plateau_status = "ok"

    plateau_result = {
        "status": plateau_status,
        "tol_used": tol_used,
        "stats": plateau_stats,
    }

    bootstrap_samples: list[dict] = []

    if plateau_status != "ok":
        bootstrap_result = {
            "status": "disabled_no_plateau_on_mean",
            "nboot_valid": 0,
            "integral_boot_mean": np.nan,
            "integral_boot_std": np.nan,
            "tau_start_boot_min": np.nan,
            "tau_start_boot_max": np.nan,
            "tau_end_boot_min": np.nan,
            "tau_end_boot_max": np.nan,
        }
        return plateau_result, bootstrap_result, bootstrap_samples

    if nrep <= 1:
        bootstrap_result = {
            "status": "disabled_insufficient_replicas",
            "nboot_valid": 0,
            "integral_boot_mean": np.nan,
            "integral_boot_std": np.nan,
            "tau_start_boot_min": np.nan,
            "tau_start_boot_max": np.nan,
            "tau_end_boot_min": np.nan,
            "tau_end_boot_max": np.nan,
        }
        return plateau_result, bootstrap_result, bootstrap_samples

    if nboot <= 1:
        bootstrap_result = {
            "status": "disabled_nboot_le_1",
            "nboot_valid": 0,
            "integral_boot_mean": np.nan,
            "integral_boot_std": np.nan,
            "tau_start_boot_min": np.nan,
            "tau_start_boot_max": np.nan,
            "tau_end_boot_min": np.nan,
            "tau_end_boot_max": np.nan,
        }
        return plateau_result, bootstrap_result, bootstrap_samples

    valid_boot_values = []
    valid_tau_start = []
    valid_tau_end = []

    boot_data = pd.DataFrame({"time": time})

    for iboot in range(nboot):
        if progress_prefix is not None and (iboot == 0 or (iboot + 1) % 100 == 0 or iboot + 1 == nboot):
            print_progress(progress_prefix, iboot + 1, nboot)

        idx = rng.integers(0, nrep, size=nrep)
        sample = replicas[idx]
        avg_ccf_boot = sample.mean(axis=0)
        integral_boot = cumulative_trapezoid(time, avg_ccf_boot)
        boot_data["integral"] = integral_boot

        b_firststep, b_laststep, b_tol_used = find_plateau_with_auto_tol(
            data=boot_data,
            tol_start=tol_start,
            nslice=nslice,
            incr=incr,
            min_time=min_tau,
            min_points=min_points,
        )

        if b_firststep == -1:
            bootstrap_samples.append(
                {
                    "bootstrap_id": iboot + 1,
                    "status": "no_plateau",
                    "integral_boot": np.nan,
                    "tau_start_boot": np.nan,
                    "tau_end_boot": np.nan,
                    "tol_used_boot": b_tol_used,
                    "ccf_plateau_mean_boot": np.nan,
                }
            )
            continue

        boot_stats = extract_plateau_stats(
            time=time,
            avg_ccf=avg_ccf_boot,
            integral=integral_boot,
            firststep=b_firststep,
            laststep=b_laststep,
        )

        ccf_mean_boot = boot_stats["ccf_plateau_mean"]

        keep = True
        if boot_ccf_mean_factor > 0:
            keep = abs(ccf_mean_boot) < b_tol_used * boot_ccf_mean_factor

        if keep:
            valid_boot_values.append(boot_stats["integral_plateau_mean"])
            valid_tau_start.append(boot_stats["tau_start"])
            valid_tau_end.append(boot_stats["tau_end"])
            sample_status = "ok"
        else:
            sample_status = "filtered_ccf_plateau_mean"

        bootstrap_samples.append(
            {
                "bootstrap_id": iboot + 1,
                "status": sample_status,
                "integral_boot": boot_stats["integral_plateau_mean"] if keep else np.nan,
                "tau_start_boot": boot_stats["tau_start"],
                "tau_end_boot": boot_stats["tau_end"],
                "tol_used_boot": b_tol_used,
                "ccf_plateau_mean_boot": ccf_mean_boot,
            }
        )

    valid_boot_values = np.asarray(valid_boot_values, dtype=float)
    valid_tau_start = np.asarray(valid_tau_start, dtype=float)
    valid_tau_end = np.asarray(valid_tau_end, dtype=float)

    if len(valid_boot_values) > 0:
        bootstrap_result = {
            "status": "ok",
            "nboot_valid": int(len(valid_boot_values)),
            "integral_boot_mean": float(np.mean(valid_boot_values)),
            "integral_boot_std": float(np.std(valid_boot_values, ddof=1)) if len(valid_boot_values) > 1 else 0.0,
            "tau_start_boot_min": float(np.min(valid_tau_start)),
            "tau_start_boot_max": float(np.max(valid_tau_start)),
            "tau_end_boot_min": float(np.min(valid_tau_end)),
            "tau_end_boot_max": float(np.max(valid_tau_end)),
        }
    else:
        bootstrap_result = {
            "status": "no_valid_bootstrap",
            "nboot_valid": 0,
            "integral_boot_mean": np.nan,
            "integral_boot_std": np.nan,
            "tau_start_boot_min": np.nan,
            "tau_start_boot_max": np.nan,
            "tau_end_boot_min": np.nan,
            "tau_end_boot_max": np.nan,
        }

    return plateau_result, bootstrap_result, bootstrap_samples


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Perform plateau analysis with optional bootstrapping for all CCF columns "
            "in one or more replica CSV files."
        )
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="Replica CSV files or glob patterns (semicolon-separated, '#' comments).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Initial tolerance for |slope| in the plateau region.",
    )
    parser.add_argument(
        "--nslice",
        type=int,
        default=10,
        help="Number of coarse slices for the search (default: 10).",
    )
    parser.add_argument(
        "--incr",
        type=float,
        default=0.01,
        help="Fraction of total points used to expand intervals (default: 0.01).",
    )
    parser.add_argument(
        "--nboot",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000).",
    )
    parser.add_argument(
        "--min-tau",
        type=float,
        default=0.0,
        help="Earliest allowed tau/time value for the start of a plateau window (default: 0).",
    )
    parser.add_argument(
        "--min-plateau-points",
        type=int,
        default=2,
        help="Minimum number of points in a valid plateau window (default: 2).",
    )
    parser.add_argument(
        "--boot-ccf-mean-factor",
        type=float,
        default=5.0,
        help=(
            "Accept a bootstrap plateau only if |mean CCF over plateau| < "
            "boot_ccf_mean_factor * tol_used_boot. Use <= 0 to disable filtering."
        ),
    )
    parser.add_argument(
        "--plateau-out",
        default="plateau_summary.csv",
        help="Machine-readable output CSV for plateau results.",
    )
    parser.add_argument(
        "--bootstrap-out",
        default="bootstrap_summary.csv",
        help="Machine-readable output CSV for bootstrap summary results.",
    )
    parser.add_argument(
        "--bootstrap-samples-out",
        default=None,
        help="Optional machine-readable output CSV for raw bootstrap samples.",
    )
    parser.add_argument(
        "--metadata-out",
        default="analysis_metadata.json",
        help="Output JSON file for run metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling (default: 42).",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    paths: List[str] = []
    for pattern in args.csv:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(matches)
        else:
            paths.append(pattern)

    if not paths:
        raise SystemExit("ERROR: No input CSV files found.")

    paths = sorted(dict.fromkeys(paths))
    first_path = Path(paths[0])

    header_line = None
    with first_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("#"):
                header_line = line.strip()
                break

    if header_line is None:
        raise SystemExit(f"ERROR: No header line starting with '#' found in {first_path}.")

    header_parts = [p.strip() for p in header_line.lstrip("#").split(";")]

    first_df = pd.read_csv(
        first_path,
        sep=";",
        comment="#",
        header=None,
        skip_blank_lines=True,
    )

    if first_df.shape[1] < 2:
        raise SystemExit("ERROR: Need at least time + one CCF column.")

    time = first_df.iloc[:, 0].astype(float).to_numpy()
    ccf_cols = list(range(1, first_df.shape[1], 2))
    if not ccf_cols:
        raise SystemExit("ERROR: No CCF columns detected.")

    replica_tables = []
    for path in paths:
        df_raw = pd.read_csv(
            path,
            sep=";",
            comment="#",
            header=None,
            skip_blank_lines=True,
        )

        if df_raw.shape != first_df.shape:
            raise SystemExit(f"ERROR: File {path} has a different shape than {first_path}.")

        time_this = df_raw.iloc[:, 0].astype(float).to_numpy()
        if not np.allclose(time_this, time, rtol=1e-12, atol=1e-15):
            raise SystemExit(f"ERROR: Time column in {path} differs from {first_path}.")

        replica_tables.append(df_raw)

    nrep = len(replica_tables)
    logging.info("Found %d replica file(s).", nrep)
    logging.info("Detected %d contribution column(s).", len(ccf_cols))

    plateau_rows = []
    bootstrap_rows = []
    bootstrap_sample_rows = []

    n_contrib = len(ccf_cols)
    for icontrib, ccf_col_idx in enumerate(ccf_cols, start=1):
        ccf_header = (
            header_parts[ccf_col_idx]
            if ccf_col_idx < len(header_parts)
            else f"col_{ccf_col_idx + 1}"
        )
        contribution = ccf_header
        if ccf_header.startswith("CCF(") and ")" in ccf_header:
            contribution = ccf_header.split("CCF(", 1)[1].split(")", 1)[0]

        logging.info("Analyzing contribution %d/%d: %s", icontrib, n_contrib, contribution)

        replicas = []
        for df_raw in replica_tables:
            ccf = df_raw.iloc[:, ccf_col_idx].astype(float).to_numpy()
            replicas.append(ccf)
        replicas = np.asarray(replicas, dtype=float)

        progress_prefix = None
        if args.nboot > 1 and replicas.shape[0] > 1:
            progress_prefix = f"[INFO] Bootstrapping {contribution}"

        plateau_result, bootstrap_result, bootstrap_samples = analyze_single_column(
            replicas=replicas,
            time=time,
            tol_start=args.tol,
            nslice=args.nslice,
            incr=args.incr,
            nboot=args.nboot,
            min_tau=args.min_tau,
            min_points=args.min_plateau_points,
            boot_ccf_mean_factor=args.boot_ccf_mean_factor,
            rng=rng,
            progress_prefix=progress_prefix,
        )

        plateau_row = make_plateau_result_row(
            contribution=contribution,
            col_1based=ccf_col_idx + 1,
            status=plateau_result["status"],
            plateau_stats=plateau_result["stats"],
            tol_start=args.tol,
            tol_used=plateau_result["tol_used"],
            nslice=args.nslice,
            incr=args.incr,
            min_tau=args.min_tau,
            nrep=nrep,
        )
        plateau_rows.append(plateau_row)

        bootstrap_row = make_bootstrap_result_row(
            contribution=contribution,
            col_1based=ccf_col_idx + 1,
            status=bootstrap_result["status"],
            nboot=args.nboot,
            nboot_valid=bootstrap_result["nboot_valid"],
            integral_boot_mean=bootstrap_result["integral_boot_mean"],
            integral_boot_std=bootstrap_result["integral_boot_std"],
            tau_start_boot_min=bootstrap_result["tau_start_boot_min"],
            tau_start_boot_max=bootstrap_result["tau_start_boot_max"],
            tau_end_boot_min=bootstrap_result["tau_end_boot_min"],
            tau_end_boot_max=bootstrap_result["tau_end_boot_max"],
            tol_start=args.tol,
            tol_used=plateau_result["tol_used"],
            nslice=args.nslice,
            incr=args.incr,
            min_tau=args.min_tau,
            nrep=nrep,
            boot_ccf_mean_factor=args.boot_ccf_mean_factor,
            seed=args.seed,
        )
        bootstrap_rows.append(bootstrap_row)

        if args.bootstrap_samples_out is not None:
            for sample in bootstrap_samples:
                bootstrap_sample_rows.append(
                    {
                        "contribution": contribution,
                        "col": ccf_col_idx + 1,
                        **sample,
                    }
                )

    plateau_df = pd.DataFrame(plateau_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    plateau_int_cols = ["col", "npoints", "nslice", "nrep"]
    for col in plateau_int_cols:
        plateau_df[col] = plateau_df[col].astype(int)

    bootstrap_int_cols = ["col", "nboot", "nboot_valid", "nslice", "nrep", "seed"]
    for col in bootstrap_int_cols:
        bootstrap_df[col] = bootstrap_df[col].astype(int)

    write_machine_and_pretty(plateau_df, Path(args.plateau_out), sep=";")
    write_machine_and_pretty(bootstrap_df, Path(args.bootstrap_out), sep=";")

    if args.bootstrap_samples_out is not None:
        bootstrap_samples_df = pd.DataFrame(bootstrap_sample_rows)
        if not bootstrap_samples_df.empty:
            for col in ["col", "bootstrap_id"]:
                bootstrap_samples_df[col] = bootstrap_samples_df[col].astype(int)
        write_machine_and_pretty(bootstrap_samples_df, Path(args.bootstrap_samples_out), sep=";")

    metadata = {
        "script_version": SCRIPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": paths,
        "nrep": nrep,
        "n_contributions": len(ccf_cols),
        "parameters": {
            "tol": args.tol,
            "nslice": args.nslice,
            "incr": args.incr,
            "nboot": args.nboot,
            "min_tau": args.min_tau,
            "min_plateau_points": args.min_plateau_points,
            "boot_ccf_mean_factor": args.boot_ccf_mean_factor,
            "seed": args.seed,
        },
        "outputs": {
            "plateau_out": args.plateau_out,
            "bootstrap_out": args.bootstrap_out,
            "bootstrap_samples_out": args.bootstrap_samples_out,
        },
    }
    write_metadata(Path(args.metadata_out), metadata)

    print("\nPlateau summary:")
    print(format_pretty_table(plateau_df, sep=";"))
    print("\nBootstrap summary:")
    print(format_pretty_table(bootstrap_df, sep=";"))

    if args.bootstrap_samples_out is not None and bootstrap_sample_rows:
        print(f"\nBootstrap samples written to {args.bootstrap_samples_out}")


if __name__ == "__main__":
    main()

