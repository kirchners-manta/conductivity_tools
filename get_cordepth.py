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

    # Precompute string representations, widths, and alignment
    col_strings = []
    col_widths = []
    col_align = []  # "left" or "right"

    for col in col_names:
        s = df[col]

        # Decide type / formatting
        if pd.api.types.is_integer_dtype(s):
            # Integers: plain string, right-aligned
            vals = [str(int(v)) for v in s]
            align = "right"
        elif pd.api.types.is_float_dtype(s):
            # Floats: fixed 8 decimal places, sign-aware
            vals = [f"{float(v): .8f}" for v in s]
            align = "right"
        else:
            # Treat as string
            vals = [str(v) for v in s]
            align = "left"

        width = max(len(str(col)), max(len(v) for v in vals) if vals else 0)

        col_strings.append(vals)
        col_widths.append(width)
        col_align.append(align)

    # Build lines
    lines = []
    sep_str = f"{sep} "

    # Header line
    header_cells = []
    for name, width in zip(col_names, col_widths):
        header_cells.append(name.ljust(width))
    lines.append(sep_str.join(header_cells))

    # Data lines
    n_rows = len(df)
    for i in range(n_rows):
        row_cells = []
        for j, name in enumerate(col_names):
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


def main(root: str = ".") -> None:
    root_path = Path(root).resolve()

    # Expect: <root>/<replica>/cordepth/<cordepth>/msdiff_out.csv
    # Example: "3/cordepth/1000/msdiff_out.csv"
    pattern = re.compile(r"(?P<replica>\d+)/cordepth/solvent/(?P<cordepth>\d+)/msdiff_out\.csv$")

    all_dfs = []

    for csv_path in root_path.rglob("msdiff_out.csv"):
        rel = csv_path.relative_to(root_path)
        m = pattern.search(str(rel))
        if not m:
            print(f"[WARN] Skipping {csv_path}, path doesn't match pattern")
            continue

        replica = int(m.group("replica"))
        cordepth = int(m.group("cordepth"))

        # Read msdiff_out.csv
        # skipinitialspace=True handles the spaces after commas in your example
        df = pd.read_csv(csv_path, skipinitialspace=True)

        # Normalize column names to "cond" formalism
        df = df.rename(
            columns={
                "contribution": "contribution",
                "sigma / S*m^-1": "cond",
                "delta_sigma / S*m^-1": "delta_cond",
                "r2": "r2",
                "t_start / ps": "t_start_ps",
                "t_end / ps": "t_end_ps",
                "n_data_fit": "n_data_fit",
            }
        )

        # Add replica & cordepth as columns
        df["replica"] = replica
        df["cordepth"] = cordepth

        all_dfs.append(df)

    if not all_dfs:
        print("[INFO] No msdiff_out.csv files found.")
        return

    big_df = pd.concat(all_dfs, ignore_index=True)

    # Set column order
    cols_order = [
        "cordepth",
        "replica",
        "contribution",
        "cond",
        "delta_cond",
        "r2",
        "t_start_ps",
        "t_end_ps",
        "n_data_fit",
    ]
    big_df = big_df[cols_order]

    # Sort by correlation depth first, then replica, then contribution
    big_df = big_df.sort_values(
        ["cordepth", "replica", "contribution"]
    ).reset_index(drop=True)

    # 1) Long-format table (pretty, ';'-separated)
    write_pretty_table(big_df, root_path / "all_conductivities_long.csv", sep=";")

    # 2) Wide-format tables for cond and delta_cond
    wide_cond = (
        big_df.pivot_table(
            index=["cordepth", "replica"],
            columns="contribution",
            values="cond",
        )
        .sort_index()
        .reset_index()
    )

    wide_delta_cond = (
        big_df.pivot_table(
            index=["cordepth", "replica"],
            columns="contribution",
            values="delta_cond",
        )
        .sort_index()
        .reset_index()
    )

    write_pretty_table(
        wide_cond, root_path / "cond_wide_by_contribution.csv", sep=";"
    )
    write_pretty_table(
        wide_delta_cond, root_path / "delta_cond_wide_by_contribution.csv", sep=";"
    )

    # 3) Summary: unweighted + weighted (scaled SE) per (cordepth, contribution)
    def summarize_group(g: pd.DataFrame) -> pd.Series:
        cond = g["cond"].to_numpy(dtype=float)
        dcond = g["delta_cond"].to_numpy(dtype=float)
        N = len(g)

        # Unweighted stats (replica-level)
        cond_mean = float(np.mean(cond))
        if N > 1:
            cond_std_between = float(np.std(cond, ddof=1))
            cond_SE = float(cond_std_between / np.sqrt(N))
        else:
            cond_SE = float("nan")

        # Weighted stats using dcond as 1σ
        mask = dcond > 0.0
        if np.any(mask):
            cond_eff = cond[mask]
            dcond_eff = dcond[mask]
            w = 1.0 / (dcond_eff * dcond_eff)
            sumw = float(np.sum(w))

            if sumw > 0.0:
                cond_wmean = float(np.sum(w * cond_eff) / sumw)
                cond_w_SE = float(np.sqrt(1.0 / sumw))

                # Residuals and chi^2
                resid = cond_eff - cond_wmean
                chi2 = float(np.sum((resid * resid) / (dcond_eff * dcond_eff)))
                dof = int(len(cond_eff) - 1)
                if dof > 0:
                    chi2_red = chi2 / dof
                else:
                    chi2_red = float("nan")

                # Scaled SE: SE * sqrt(chi2_red)
                if chi2_red > 0.0:
                    cond_w_SE_scaled = float(cond_w_SE * np.sqrt(chi2_red))
                else:
                    cond_w_SE_scaled = float("nan")
            else:
                cond_wmean = float("nan")
                cond_w_SE_scaled = float("nan")
                chi2_red = float("nan")
        else:
            cond_wmean = float("nan")
            cond_w_SE_scaled = float("nan")
            chi2_red = float("nan")

        return pd.Series(
            {
                "n_replicas": N,
                # Unweighted replica stats
                "cond_mean": cond_mean,
                "cond_SE": cond_SE,
                # Weighted stats based on delta_cond
                "cond_weighted_mean": cond_wmean,
                "cond_weighted_SE_scaled": cond_w_SE_scaled,
                # Goodness of combination
                "chi2_red": chi2_red,
            }
        )

    summary = (
        big_df.groupby(["cordepth", "contribution"])
        .apply(summarize_group)
        .reset_index()
        .sort_values(["cordepth", "contribution"])
    )

    summary["n_replicas"] = summary["n_replicas"].astype(int)

    # Write summary (pretty, ';'-separated)
    write_pretty_table(
        summary, root_path / "summary_by_cordepth_and_contribution.csv", sep=";"
    )

    print("[INFO] Wrote pretty, ';'-separated tables:")
    print("  all_conductivities_long.csv")
    print("  cond_wide_by_contribution.csv")
    print("  delta_cond_wide_by_contribution.csv")
    print("  summary_by_cordepth_and_contribution.csv")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [root_dir]", file=sys.stderr)
        sys.exit(1)
    root_dir = sys.argv[1] if len(sys.argv) == 2 else "."
    main(root_dir)

