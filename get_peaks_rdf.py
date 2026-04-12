#!/usr/bin/env python3
"""
Find robust maxima/minima in radial distribution functions g(r).

Strategy:
- Parse semicolon-separated data (supports comment header lines starting with '#')
- Smooth g(r) with Savitzky-Golay (or optional Gaussian)
- Use peak prominence + minimum distance/width to reject noise
- Find peaks in g(r) and minima by finding peaks in -g(r)
- Report first meaningful peak and first meaningful minimum after that peak (common RDF use-case)
- Optionally refine positions on raw data by quadratic fit around each extremum
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import find_peaks, savgol_filter
except ImportError as e:
    raise SystemExit(
        "This script needs SciPy. Install with: pip install scipy"
    ) from e


@dataclass
class Extremum:
    kind: str          # "max" or "min"
    idx: int
    r: float
    g: float
    prominence: float
    width: float       # in samples


def read_rdf_file(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Reads a file with lines like:
    # Distance / pm;  g(r);  Integral
    0.5000000000;  0.000000000;  0.000000000

    Returns r, g, integral (integral may be None if missing).
    """
    r_list = []
    g_list = []
    i_list = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(";")]
            if len(parts) < 2:
                continue
            try:
                r_val = float(parts[0].replace(",", "."))
                g_val = float(parts[1].replace(",", "."))
                r_list.append(r_val)
                g_list.append(g_val)
                if len(parts) >= 3 and parts[2] != "":
                    i_list.append(float(parts[2].replace(",", ".")))
            except ValueError:
                continue

    r = np.asarray(r_list, dtype=float)
    g = np.asarray(g_list, dtype=float)

    integral = None
    if len(i_list) == len(r_list) and len(i_list) > 0:
        integral = np.asarray(i_list, dtype=float)

    if r.size < 5:
        raise ValueError(f"{path}: not enough data points parsed.")

    # Ensure sorted by r
    order = np.argsort(r)
    r, g = r[order], g[order]
    if integral is not None:
        integral = integral[order]

    return r, g, integral


def choose_savgol_window(n: int, approx_frac: float = 0.03, min_win: int = 7) -> int:
    """
    Pick a reasonable odd window length for Savitzky–Golay based on data length.
    approx_frac=0.03 means ~3% of the signal length.
    """
    win = max(min_win, int(math.ceil(n * approx_frac)))
    if win % 2 == 0:
        win += 1
    # must be < n and >= polyorder+2
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < min_win:
        win = min_win if min_win < n else (n if n % 2 == 1 else n - 1)
    return max(5, win)


def smooth_signal(g: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    window = int(window)
    polyorder = int(polyorder)
    if window % 2 == 0:
        window += 1
    if window >= g.size:
        window = g.size - 1 if (g.size - 1) % 2 == 1 else g.size - 2
    polyorder = min(polyorder, window - 2)
    return savgol_filter(g, window_length=window, polyorder=polyorder, mode="interp")


def robust_scale(x: np.ndarray) -> float:
    """
    Robust noise estimate using MAD (median absolute deviation).
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else float(np.std(x))


def quadratic_refine(r: np.ndarray, g: np.ndarray, idx: int) -> Tuple[float, float]:
    """
    Refine extremum position using quadratic fit to 3 points around idx on RAW signal.
    Returns (r_refined, g_refined). If refinement isn't possible, returns the original point.
    """
    if idx <= 0 or idx >= len(r) - 1:
        return r[idx], g[idx]

    x = r[idx - 1: idx + 2]
    y = g[idx - 1: idx + 2]
    # Fit y = ax^2 + bx + c
    coeff = np.polyfit(x, y, 2)
    a, b, c = coeff
    if a == 0:
        return r[idx], g[idx]
    x0 = -b / (2 * a)
    # constrain to local interval to avoid crazy jumps
    if x0 < x[0] or x0 > x[-1]:
        return r[idx], g[idx]
    y0 = a * x0**2 + b * x0 + c
    return float(x0), float(y0)


def find_extrema(
    r: np.ndarray,
    g_raw: np.ndarray,
    g_smooth: np.ndarray,
    *,
    min_distance_in_r: float,
    prominence_sigma: float,
    width_min_samples: int,
    refine: bool = True,
) -> Tuple[List[Extremum], List[Extremum]]:
    """
    Returns (maxima, minima) lists, sorted by r.
    - min_distance_in_r converted to samples using median dr
    - prominence threshold computed from robust noise scale of (raw - smooth)
    """
    dr = np.median(np.diff(r))
    if not np.isfinite(dr) or dr <= 0:
        dr = 1.0

    distance_samples = max(1, int(round(min_distance_in_r / dr)))

    noise = g_raw - g_smooth
    sigma = robust_scale(noise)
    prom = max(0.0, float(prominence_sigma) * sigma)

    # Maxima on smoothed g
    p_idx, p_props = find_peaks(
        g_smooth,
        distance=distance_samples,
        prominence=prom,
        width=width_min_samples,
    )

    maxima: List[Extremum] = []
    for k, idx in enumerate(p_idx):
        rr, gg = (r[idx], g_raw[idx])
        if refine:
            rr, gg = quadratic_refine(r, g_raw, int(idx))
        maxima.append(
            Extremum(
                kind="max",
                idx=int(idx),
                r=float(rr),
                g=float(gg),
                prominence=float(p_props["prominences"][k]),
                width=float(p_props["widths"][k]) if "widths" in p_props else float("nan"),
            )
        )

    # Minima: peaks of -g_smooth
    m_idx, m_props = find_peaks(
        -g_smooth,
        distance=distance_samples,
        prominence=prom,
        width=width_min_samples,
    )

    minima: List[Extremum] = []
    for k, idx in enumerate(m_idx):
        rr, gg = (r[idx], g_raw[idx])
        if refine:
            rr, gg = quadratic_refine(r, g_raw, int(idx))
        minima.append(
            Extremum(
                kind="min",
                idx=int(idx),
                r=float(rr),
                g=float(gg),
                prominence=float(m_props["prominences"][k]),
                width=float(m_props["widths"][k]) if "widths" in m_props else float("nan"),
            )
        )

    maxima.sort(key=lambda e: e.r)
    minima.sort(key=lambda e: e.r)
    return maxima, minima


def pick_first_peak_and_following_minimum(
    maxima: List[Extremum],
    minima: List[Extremum],
    *,
    r_min: float = -float("inf"),
) -> Tuple[Optional[Extremum], Optional[Extremum]]:
    """
    Common RDF use-case:
    - first significant peak beyond r_min
    - first significant minimum after that peak
    """
    first_peak = next((m for m in maxima if m.r > r_min), None)
    if first_peak is None:
        return None, None
    first_min_after = next((m for m in minima if m.r > first_peak.r), None)
    return first_peak, first_min_after


def main() -> None:
    ap = argparse.ArgumentParser(description="Find robust maxima/minima in noisy RDF CSV-like files.")
    ap.add_argument("inputs", nargs="+", help="Input files (supports glob patterns).")
    ap.add_argument("--window", type=int, default=0,
                    help="Savitzky-Golay window length (odd). 0=auto based on file length.")
    ap.add_argument("--polyorder", type=int, default=3, help="Savitzky-Golay polynomial order.")
    ap.add_argument("--min-distance", type=float, default=2.0,
                    help="Minimum separation between extrema in r-units (same as your Distance column).")
    ap.add_argument("--prom-sigma", type=float, default=3.0,
                    help="Minimum prominence as N * robust_noise_sigma (higher rejects more noise).")
    ap.add_argument("--min-width", type=int, default=3,
                    help="Minimum peak width in samples (reject single-bin wiggles).")
    ap.add_argument("--r-min", type=float, default=-1.0,
                    help="Ignore extrema with r <= this (useful to drop contact-region artifacts).")
    ap.add_argument("--top", type=int, default=0,
                    help="If >0, print top-N maxima/minima by prominence (in addition to first peak/min).")
    ap.add_argument("--no-refine", action="store_true", help="Disable quadratic refinement on raw data.")
    args = ap.parse_args()

    # Expand globs
    paths: List[Path] = []
    for pat in args.inputs:
        matches = list(Path().glob(pat))
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pat))

    for path in paths:
        r, g, _ = read_rdf_file(path)
        window = args.window or choose_savgol_window(len(g), approx_frac=0.03, min_win=7)
        g_sm = smooth_signal(g, window=window, polyorder=args.polyorder)

        maxima, minima = find_extrema(
            r, g_raw=g, g_smooth=g_sm,
            min_distance_in_r=args.min_distance,
            prominence_sigma=args.prom_sigma,
            width_min_samples=args.min_width,
            refine=not args.no_refine,
        )

        # apply r-min cutoff
        maxima_f = [e for e in maxima if e.r > args.r_min]
        minima_f = [e for e in minima if e.r > args.r_min]

        first_peak, first_min = pick_first_peak_and_following_minimum(maxima_f, minima_f, r_min=args.r_min)

        print(f"\n=== {path} ===")
        print(f"Smoothing: Savitzky–Golay window={window} polyorder={args.polyorder}")
        print(f"Filters: min_distance={args.min_distance}  prom_sigma={args.prom_sigma}  min_width={args.min_width}  r_min={args.r_min}")

        if first_peak:
            print(f"First peak:    r={first_peak.r:.6g}  g={first_peak.g:.6g}  prom={first_peak.prominence:.6g}")
        else:
            print("First peak:    not found")

        if first_min:
            print(f"First minimum: r={first_min.r:.6g}  g={first_min.g:.6g}  prom={first_min.prominence:.6g}")
        else:
            print("First minimum: not found (after first peak)")

        if args.top and args.top > 0:
            # Top-N by prominence
            max_top = sorted(maxima_f, key=lambda e: e.prominence, reverse=True)[: args.top]
            min_top = sorted(minima_f, key=lambda e: e.prominence, reverse=True)[: args.top]

            print("\nTop maxima by prominence:")
            for e in max_top:
                print(f"  r={e.r:.6g}  g={e.g:.6g}  prom={e.prominence:.6g}  width={e.width:.3g}")

            print("\nTop minima by prominence:")
            for e in min_top:
                print(f"  r={e.r:.6g}  g={e.g:.6g}  prom={e.prominence:.6g}  width={e.width:.3g}")


if __name__ == "__main__":
    main()

