#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


def write_pretty_table(df: pd.DataFrame, path: Path, sep: str = ";") -> None:
    """
    Write DataFrame 'df' to 'path' with:
      - semicolon separator
      - aligned columns:
          * floats: fixed 8 decimal places, right-aligned
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
        for j, _name in enumerate(col_names):
            val_str = col_strings[j][i]
            width = col_widths[j]
            align = col_align[j]

            if align == "right":
                cell = val_str.rjust(width)
            else:
                cell = val_str.ljust(width)
            row_cells.append(cell)

        lines.append(sep_str.join(row_cells))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_and_se(values: np.ndarray) -> tuple[float, float]:
    """
    Return arithmetic mean and standard error from replica-to-replica spread.
    """
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))

    if len(values) > 1:
        se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    else:
        se = float("nan")

    return mean, se


def summarize_compound(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize one compound across replicas using unweighted statistics only.
    """
    D0_mean, D0_SE = mean_and_se(df["D0"].to_numpy(dtype=float))
    K_mean, K_SE = mean_and_se(df["K"].to_numpy(dtype=float))
    r2_mean, r2_SE = mean_and_se(df["r2"].to_numpy(dtype=float))
    t_start_mean, t_start_SE = mean_and_se(df["t_start_ps"].to_numpy(dtype=float))
    t_end_mean, t_end_SE = mean_and_se(df["t_end_ps"].to_numpy(dtype=float))
    n_data_mean, n_data_SE = mean_and_se(df["n_data"].to_numpy(dtype=float))

    summary = pd.DataFrame(
        [
            {
                "n_replicas": int(len(df)),
                "D0_mean": D0_mean,
                "D0_SE": D0_SE,
                "K_mean": K_mean,
                "K_SE": K_SE,
                "r2_mean": r2_mean,
                "r2_SE": r2_SE,
                "t_start_ps_mean": t_start_mean,
                "t_start_ps_SE": t_start_SE,
                "t_end_ps_mean": t_end_mean,
                "t_end_ps_SE": t_end_SE,
                "n_data_mean": n_data_mean,
                "n_data_SE": n_data_SE,
            }
        ]
    )

    summary["n_replicas"] = summary["n_replicas"].astype(int)
    return summary


def main(root: str = ".") -> None:
    root_path = Path(root).resolve()

    # Expect: <root>/<replica>/msd/<filename>
    # Example: 1/msd/msdiff_msd_C2N3_#2_out.csv
    pattern = re.compile(r"(?P<replica>\d+)/msd/(?P<filename>msdiff_.*\.csv)$")

    # Group dataframes by exact input filename
    grouped: dict[str, list[pd.DataFrame]] = {}

    for csv_path in root_path.rglob("msdiff_*.csv"):
        rel = csv_path.relative_to(root_path).as_posix()
        m = pattern.search(rel)
        if not m:
            print(f"[WARN] Skipping {csv_path}, path doesn't match '<replica>/msd/msdiff_*.csv'")
            continue

        replica = int(m.group("replica"))
        filename = m.group("filename")

        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]

        df = df.rename(
            columns={
                "D_0 / 10^-12 m^2/s": "D0",
                "delta_D": "delta_D",
                "K / 10^-12 m^2/s": "K",
                "delta_K": "delta_K",
                "r2": "r2",
                "t_start / ps": "t_start_ps",
                "t_end / ps": "t_end_ps",
                "n_data": "n_data",
            }
        )

        required = [
            "D0",
            "delta_D",
            "K",
            "delta_K",
            "r2",
            "t_start_ps",
            "t_end_ps",
            "n_data",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] Skipping {csv_path}, missing columns: {missing}")
            continue

        df = df[required].copy()
        df["replica"] = replica
        df["source_file"] = filename

        grouped.setdefault(filename, []).append(df)

    if not grouped:
        print("[INFO] No matching <replica>/msd/msdiff_*.csv files found.")
        return

    written_files = []

    for filename in sorted(grouped):
        compound_df = pd.concat(grouped[filename], ignore_index=True)

        cols_order = [
            "replica",
            "D0",
            "delta_D",
            "K",
            "delta_K",
            "r2",
            "t_start_ps",
            "t_end_ps",
            "n_data",
        ]
        compound_df = compound_df[cols_order].sort_values(["replica"]).reset_index(drop=True)

        summary = summarize_compound(compound_df)

        stem = Path(filename).stem
        all_path = root_path / f"all_replicas_{stem}.csv"
        summary_path = root_path / f"summary_{stem}.csv"

        write_pretty_table(compound_df, all_path, sep=";")
        write_pretty_table(summary, summary_path, sep=";")

        written_files.append(all_path.name)
        written_files.append(summary_path.name)

    print("[INFO] Wrote files:")
    for name in written_files:
        print(f"  {name}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [root_dir]", file=sys.stderr)
        sys.exit(1)

    root_dir = sys.argv[1] if len(sys.argv) == 2 else "."
    main(root_dir)

