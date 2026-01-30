#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module-level luminescence-to-I–V pipeline (public version).

This script demonstrates the end-to-end workflow:
1) Load one luminescence image (EL/PL) and convert to grayscale.
2) Segment the image into a (numRows × numCols) cell grid.
3) For each cell patch, call a public `calculate_J0_Rs(...)` stub to produce a reconstructed cell J–V curve.
4) Aggregate cell J–V curves into sub-strings (series), then submodules (parallel), then the full module (series).
5) Report module-level I–V metrics (Pmpp, Voc, etc.).

Important
---------
The proprietary iterative inversion algorithm is intentionally omitted and replaced by a public stub
(`calculate_J0_Rs.py`). The purpose is to expose the workflow and software structure without
disclosing patent-protected details.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

from calculate_J0_Rs import calculate_J0_Rs


# -------------------------
# Series / parallel helpers
# -------------------------
def calculate_series_string(cells: List[Dict[str, np.ndarray]], n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine a list of cell (or substring) J–V curves in series.

    Input elements must be dicts containing:
        {"V": np.ndarray, "J": np.ndarray}  (J in mA/cm^2)
    """
    valid = [c for c in cells if isinstance(c, dict) and ("V" in c) and ("J" in c) and len(c["V"]) and len(c["J"])]
    if not valid:
        return np.array([]), np.array([])

    min_J = min(np.min(c["J"]) for c in valid)
    max_J = min(np.max(c["J"]) for c in valid)
    if not (min_J < max_J):
        return np.array([]), np.array([])

    J_samples = np.linspace(min_J, max_J, n_samples)
    V_string = np.zeros_like(J_samples, dtype=float)

    for c in valid:
        J = np.asarray(c["J"], dtype=float)
        V = np.asarray(c["V"], dtype=float)
        order = np.argsort(J)
        J, V = J[order], V[order]
        V_interp = np.interp(J_samples, J, V, left=V[0], right=V[-1])
        V_string += V_interp

    return V_string, J_samples


def calculate_parallel_strings(
    V1: np.ndarray, J1: np.ndarray, V2: np.ndarray, J2: np.ndarray, n_samples: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine two J–V curves in parallel.

    We interpolate currents onto a common voltage grid and add them.
    """
    if len(V1) == 0 or len(V2) == 0:
        return np.array([]), np.array([])

    min_V = max(np.min(V1), np.min(V2))
    max_V = min(np.max(V1), np.max(V2))
    if not (min_V < max_V):
        return np.array([]), np.array([])

    V_samples = np.linspace(min_V, max_V, n_samples)

    o1 = np.argsort(V1); o2 = np.argsort(V2)
    V1s, J1s = V1[o1], J1[o1]
    V2s, J2s = V2[o2], J2[o2]

    J1i = np.interp(V_samples, V1s, J1s)
    J2i = np.interp(V_samples, V2s, J2s)

    return V_samples, (J1i + J2i)


# -------------------------
# Image segmentation helper
# -------------------------
@dataclass(frozen=True)
class GridSpec:
    num_rows: int = 6
    num_cols: int = 24
    deta: float = 0.15      # spacing factor used in the original dataset
    split_col: int = 12     # left/right split for the two parallel sub-strings in each submodule
    submodule_rows: int = 2 # number of rows per submodule (6 rows -> 3 submodules)


def _load_grayscale(img_path: Path) -> np.ndarray:
    img = Image.open(str(img_path))
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.float64)


def _segment_cells(img_np: np.ndarray, grid: GridSpec) -> List[Tuple[int, int, np.ndarray]]:
    """
    Return a list of (row, col, patch) for each cell, using the same geometric rule as the original code.
    """
    H, W = img_np.shape
    cell_h = H / grid.num_rows
    cell_w = W / (grid.num_cols + grid.deta)
    gap_w = cell_w * 0.15

    cells: List[Tuple[int, int, np.ndarray]] = []
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if c < grid.split_col:
                x = (c + 1) * cell_w - cell_w
            else:
                x = grid.split_col * cell_w + gap_w + (c - (grid.split_col - 1)) * cell_w - cell_w
            y = (r + 1) * cell_h - cell_h

            x0 = max(0, min(int(round(x)), W - 1))
            y0 = max(0, min(int(round(y)), H - 1))
            w = max(1, int(round(cell_w)))
            h = max(1, int(round(cell_h)))
            x1 = max(x0 + 1, min(x0 + w, W))
            y1 = max(y0 + 1, min(y0 + h, H))
            patch = img_np[y0:y1, x0:x1].copy()
            if patch.size == 0:
                continue
            cells.append((r, c, patch))
    return cells


# -------------------------
# Public workflow entrypoint
# -------------------------
def process_one_image(
    img_path: Path,
    *,
    Jph: float = 42.49,
    total_area: float = 165.62,  # cm^2 used to convert J (mA/cm^2) to I (A)
    grid: GridSpec = GridSpec(),
    f1: Optional[float] = None,
    f2: Optional[float] = None,
    downsample: int = 1,
    use_parallel: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, float]:
    """
    Process a single image and return module-level metrics.

    Returns
    -------
    dict with keys: Pmpp, Voc, Vmpp, Impp, FF
    """
    img_np = _load_grayscale(img_path)
    cell_items = _segment_cells(img_np, grid)

    # Optional downsample for speed (used in calibration)
    if isinstance(downsample, int) and downsample > 1:
        cell_items = [(r, c, p[::downsample, ::downsample]) for (r, c, p) in cell_items]

    # Per-cell reconstruction (public stub)
    JV_curves: List[List[Optional[Dict[str, np.ndarray]]]] = [
        [None for _ in range(grid.num_cols)] for _ in range(grid.num_rows)
    ]

    def _run_one(item):
        r, c, patch = item
        ret = calculate_J0_Rs(patch, total_area, Jph, f1=f1, f2=f2)
        total_JV = ret[10]
        return r, c, total_JV

    if use_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_run_one, it) for it in cell_items]
            for fut in as_completed(futs):
                r, c, total_JV = fut.result()
                if isinstance(total_JV, dict) and ("V" in total_JV) and ("J" in total_JV):
                    JV_curves[r][c] = total_JV
    else:
        for it in cell_items:
            r, c, total_JV = _run_one(it)
            if isinstance(total_JV, dict) and ("V" in total_JV) and ("J" in total_JV):
                JV_curves[r][c] = total_JV

    # Group into 3 submodules × 2 parallel sub-strings
    n_submodules = max(1, grid.num_rows // grid.submodule_rows)
    subs: List[List[List[Dict[str, np.ndarray]]]] = [
        [[], []] for _ in range(n_submodules)
    ]

    for r in range(grid.num_rows):
        sm = min(n_submodules - 1, r // grid.submodule_rows)
        for c in range(grid.num_cols):
            JV = JV_curves[r][c]
            if JV is None:
                continue
            s = 0 if c < grid.split_col else 1
            subs[sm][s].append(JV)

    # Series inside each substring
    VJ_substrings: List[Tuple[np.ndarray, np.ndarray]] = []
    for sm in range(n_submodules):
        for s in range(2):
            V_str, J_str = calculate_series_string(subs[sm][s])
            VJ_substrings.append((V_str, J_str))

    # Parallel within each submodule
    V_submods: List[Tuple[np.ndarray, np.ndarray]] = []
    for sm in range(n_submodules):
        (V_a, J_a) = VJ_substrings[2 * sm + 0]
        (V_b, J_b) = VJ_substrings[2 * sm + 1]
        V_sm, J_sm = calculate_parallel_strings(V_a, J_a, V_b, J_b)
        V_submods.append((V_sm, J_sm))

    if any(len(V) == 0 or len(J) == 0 for (V, J) in V_submods):
        raise RuntimeError("Insufficient submodule data to compute the full module I–V curve.")

    # Series across submodules -> module JV
    sub_dicts = [{"V": V, "J": J} for (V, J) in V_submods]
    V_mod, J_mod = calculate_series_string(sub_dicts)

    if len(V_mod) == 0 or len(J_mod) == 0:
        raise RuntimeError("Failed to compute module I–V curve.")

    # Convert J (mA/cm^2) to I (A) using a reference area
    I_mod = J_mod * float(total_area) / 1000.0
    P_mod = V_mod * I_mod

    idx = int(np.argmax(P_mod))
    Pmpp = float(P_mod[idx])
    Vmpp = float(V_mod[idx])
    Impp = float(I_mod[idx])

    Voc = float(np.max(V_mod))
    Isc = float(np.max(I_mod))
    FF = float(Pmpp / (Voc * Isc)) if (Voc > 0 and Isc > 0) else 0.0

    return {"Pmpp": Pmpp, "Voc": Voc, "Vmpp": Vmpp, "Impp": Impp, "FF": FF}


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Public one-shot luminescence workflow (module-level).")
    p.add_argument("--image", type=str, required=True, help="Path to a luminescence image (EL/PL).")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=24)
    p.add_argument("--deta", type=float, default=0.15)
    p.add_argument("--split-col", type=int, default=12)
    p.add_argument("--submodule-rows", type=int, default=2)

    p.add_argument("--Jph", type=float, default=42.49)
    p.add_argument("--area", type=float, default=165.62)
    p.add_argument("--f1", type=float, default=None)
    p.add_argument("--f2", type=float, default=None)

    p.add_argument("--downsample", type=int, default=4)
    p.add_argument("--parallel", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    grid = GridSpec(
        num_rows=args.rows,
        num_cols=args.cols,
        deta=args.deta,
        split_col=args.split_col,
        submodule_rows=args.submodule_rows,
    )
    metrics = process_one_image(
        Path(args.image),
        Jph=args.Jph,
        total_area=args.area,
        grid=grid,
        f1=args.f1,
        f2=args.f2,
        downsample=max(1, int(args.downsample)),
        use_parallel=bool(args.parallel),
        max_workers=args.max_workers,
    )
    print("Module metrics (public stub):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
