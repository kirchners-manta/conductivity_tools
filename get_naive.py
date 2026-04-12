#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


# Map species names to ion roles
SPECIES_ROLE_ALIASES = {
    "Li": "cation",
    "C6H11N2": "cation",
    "F2NO4S2": "anion",
    "C2N3": "anion",
}

CONTRIBUTION_ORDER = [
    "Einstein-Helfand",
    "Nernst-Einstein",
    "cation cross",
    "cation self",
    "cation total",
    "anion-cation",
    "anion cross",
    "anion self",
    "anion total",
]


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
        for j in range(len(col_names)):
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


def rss(*values: float) -> float:
    arr = np.asarray(values, dtype=float)
    if np.any(np.isnan(arr)):
        return np.nan
    return float(np.sqrt(np.sum(arr ** 2)))


def contribution_sort_key(name: str) -> tuple[int, str]:
    try:
        return (CONTRIBUTION_ORDER.index(name), name)
    except ValueError:
        return (len(CONTRIBUTION_ORDER), name)


def canonicalize_contribution(raw_label: str) -> str | None:
    """
    Map travis.log contribution labels to the canonical contribution names.

    Examples:
      Li                -> cation self
      C2N3              -> anion self
      Li/Li             -> cation cross
      C2N3/C2N3         -> anion cross
      Li/C2N3           -> anion-cation
      C2N3/Li           -> anion-cation
    """
    raw_label = raw_label.strip()

    if "/" not in raw_label:
        role = SPECIES_ROLE_ALIASES.get(raw_label)
        if role is None:
            return None
        return f"{role} self"

    left, right = [x.strip() for x in raw_label.split("/", 1)]
    left_role = SPECIES_ROLE_ALIASES.get(left)
    right_role = SPECIES_ROLE_ALIASES.get(right)

    if left_role is None or right_role is None:
        return None

    if left == right:
        return f"{left_role} cross"

    if {left_role, right_role} == {"anion", "cation"}:
        return "anion-cation"

    return None


INTERVAL_RE = re.compile(
    r"Performing linear regression on interval\s+([0-9.+-Ee]+)\s*-\s*([0-9.+-Ee]+)\s*ps"
)

COND_RE = re.compile(
    r"conductivity\s*=\s*([+-]?[0-9.]+(?:[Ee][+-]?[0-9]+)?)\s*\+/-\s*([+-]?[0-9.]+(?:[Ee][+-]?[0-9]+)?)"
)

CONTRIB_RE = re.compile(
    r"^\s*([^:]+?)\s*:\s*([+-]?[0-9.]+(?:[Ee][+-]?[0-9]+)?)\s*\+/-\s*([+-]?[0-9.]+(?:[Ee][+-]?[0-9]+)?)\s*S/m"
)


def parse_travis_log(path: Path) -> pd.DataFrame:
    """
    Parse one travis.log and return a long-format DataFrame with the same
    contribution labels used elsewhere in the conductivity workflow.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    current = None
    last_complete = None

    in_contrib_section = False

    for line in lines:
        m = INTERVAL_RE.search(line)
        if m:
            # start a new block; keep only the last complete one found
            current = {
                "t_start_ps": float(m.group(1)),
                "t_end_ps": float(m.group(2)),
                "eh_cond": None,
                "eh_delta": None,
                "contribs": {},
            }
            in_contrib_section = False
            continue

        if current is None:
            continue

        if current["eh_cond"] is None:
            m = COND_RE.search(line)
            if m:
                current["eh_cond"] = float(m.group(1))
                current["eh_delta"] = float(m.group(2))
                continue

        if "Contributions to the conductivity:" in line:
            in_contrib_section = True
            continue

        if in_contrib_section:
            if not line.strip():
                in_contrib_section = False
                continue

            m = CONTRIB_RE.match(line)
            if m:
                raw_label = m.group(1).strip()
                cond = float(m.group(2))
                delta = float(m.group(3))

                canonical = canonicalize_contribution(raw_label)
                if canonical is None:
                    print(f"[WARN] Skipping unknown contribution label '{raw_label}' in {path}")
                    continue

                current["contribs"][canonical] = {
                    "cond": cond,
                    "delta_cond": delta,
                }
                continue

        # keep the last block that has enough information
        required_keys = {"cation self", "anion self", "cation cross", "anion cross", "anion-cation"}
        if (
            current["eh_cond"] is not None
            and required_keys.issubset(current["contribs"].keys())
        ):
            last_complete = current.copy()
            last_complete["contribs"] = dict(current["contribs"])

    if last_complete is None:
        raise ValueError(f"Could not find a complete regression/contribution block in {path}")

    block = last_complete

    c_self = block["contribs"]["cation self"]["cond"]
    c_self_err = block["contribs"]["cation self"]["delta_cond"]

    a_self = block["contribs"]["anion self"]["cond"]
    a_self_err = block["contribs"]["anion self"]["delta_cond"]

    c_cross = block["contribs"]["cation cross"]["cond"]
    c_cross_err = block["contribs"]["cation cross"]["delta_cond"]

    a_cross = block["contribs"]["anion cross"]["cond"]
    a_cross_err = block["contribs"]["anion cross"]["delta_cond"]

    pair = block["contribs"]["anion-cation"]["cond"]
    pair_err = block["contribs"]["anion-cation"]["delta_cond"]

    rows = [
        {
            "contribution": "Einstein-Helfand",
            "cond": block["eh_cond"],
            "delta_cond": block["eh_delta"],
        },
        {
            "contribution": "Nernst-Einstein",
            "cond": c_self + a_self,
            "delta_cond": rss(c_self_err, a_self_err),
        },
        {
            "contribution": "cation cross",
            "cond": c_cross,
            "delta_cond": c_cross_err,
        },
        {
            "contribution": "cation self",
            "cond": c_self,
            "delta_cond": c_self_err,
        },
        {
            "contribution": "cation total",
            "cond": c_self + c_cross,
            "delta_cond": rss(c_self_err, c_cross_err),
        },
        {
            "contribution": "anion-cation",
            "cond": pair,
            "delta_cond": pair_err,
        },
        {
            "contribution": "anion cross",
            "cond": a_cross,
            "delta_cond": a_cross_err,
        },
        {
            "contribution": "anion self",
            "cond": a_self,
            "delta_cond": a_self_err,
        },
        {
            "contribution": "anion total",
            "cond": a_self + a_cross,
            "delta_cond": rss(a_self_err, a_cross_err),
        },
    ]

    df = pd.DataFrame(rows)
    df["r2"] = np.nan
    df["t_start_ps"] = block["t_start_ps"]
    df["t_end_ps"] = block["t_end_ps"]
    df["n_data_fit"] = np.nan

    return df[
        [
            "contribution",
            "cond",
            "delta_cond",
            "r2",
            "t_start_ps",
            "t_end_ps",
            "n_data_fit",
        ]
    ]


def main(root: str = ".") -> None:
    root_path = Path(root).resolve()

    # Expect: <root>/<replica>/cordepth/solvent/<cordepth>/travis.log
    pattern = re.compile(r"(?P<replica>\d+)/cordepth/mass/(?P<cordepth>\d+)/travis\.log$")

    all_dfs = []

    for log_path in root_path.rglob("travis.log"):
        rel = log_path.relative_to(root_path)
        m = pattern.search(str(rel))
        if not m:
            #print(f"[WARN] Skipping {log_path}, path doesn't match pattern")
            continue

        replica = int(m.group("replica"))
        cordepth = int(m.group("cordepth"))

        try:
            df = parse_travis_log(log_path)
        except Exception as e:
            print(f"[WARN] Failed to parse {log_path}: {e}")
            continue

        df["replica"] = replica
        df["cordepth"] = cordepth

        all_dfs.append(df)

    if not all_dfs:
        print("[INFO] No travis.log files found.")
        return

    big_df = pd.concat(all_dfs, ignore_index=True)

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

    # Sort by correlation depth, replica, canonical contribution order
    big_df = big_df.sort_values(
        by=["cordepth", "replica", "contribution"],
        key=lambda s: s.map(contribution_sort_key) if s.name == "contribution" else s,
    ).reset_index(drop=True)

    # 1) Long-format table
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

    # Reorder wide columns if present
    cond_cols = ["cordepth", "replica"] + [c for c in CONTRIBUTION_ORDER if c in wide_cond.columns]
    delta_cols = ["cordepth", "replica"] + [c for c in CONTRIBUTION_ORDER if c in wide_delta_cond.columns]

    wide_cond = wide_cond[cond_cols]
    wide_delta_cond = wide_delta_cond[delta_cols]

    write_pretty_table(wide_cond, root_path / "cond_wide_by_contribution.csv", sep=";")
    write_pretty_table(wide_delta_cond, root_path / "delta_cond_wide_by_contribution.csv", sep=";")

    # 3) Summary: unweighted + weighted (scaled SE) per (cordepth, contribution)
    def summarize_group(g: pd.DataFrame) -> pd.Series:
        cond = g["cond"].to_numpy(dtype=float)
        dcond = g["delta_cond"].to_numpy(dtype=float)
        N = len(g)

        cond_mean = float(np.mean(cond))
        if N > 1:
            cond_std_between = float(np.std(cond, ddof=1))
            cond_SE = float(cond_std_between / np.sqrt(N))
        else:
            cond_SE = float("nan")

        mask = dcond > 0.0
        if np.any(mask):
            cond_eff = cond[mask]
            dcond_eff = dcond[mask]
            w = 1.0 / (dcond_eff * dcond_eff)
            sumw = float(np.sum(w))

            if sumw > 0.0:
                cond_wmean = float(np.sum(w * cond_eff) / sumw)
                cond_w_SE = float(np.sqrt(1.0 / sumw))

                resid = cond_eff - cond_wmean
                chi2 = float(np.sum((resid * resid) / (dcond_eff * dcond_eff)))
                dof = int(len(cond_eff) - 1)
                chi2_red = chi2 / dof if dof > 0 else float("nan")

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
                "cond_mean": cond_mean,
                "cond_SE": cond_SE,
                "cond_weighted_mean": cond_wmean,
                "cond_weighted_SE_scaled": cond_w_SE_scaled,
                "chi2_red": chi2_red,
            }
        )

    summary = (
        big_df.groupby(["cordepth", "contribution"])
        .apply(summarize_group)
        .reset_index()
    )

    summary["n_replicas"] = summary["n_replicas"].astype(int)

    summary = summary.sort_values(
        by=["cordepth", "contribution"],
        key=lambda s: s.map(contribution_sort_key) if s.name == "contribution" else s,
    ).reset_index(drop=True)

    write_pretty_table(
        summary, root_path / "summary_by_cordepth_and_contribution.csv", sep=";"
    )

    print("[INFO] Wrote pretty, ';'-separated tables:")
    print("  all_conductivities_long.csv")
    print("  cond_wide_by_contribution.csv")
    print("  delta_cond_wide_by_contribution.csv")
    print("  summary_by_cordepth_and_contribution.csv")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [root_dir]", file=sys.stderr)
        sys.exit(1)

    root_dir = sys.argv[1] if len(sys.argv) == 2 else "."
    main(root_dir)

