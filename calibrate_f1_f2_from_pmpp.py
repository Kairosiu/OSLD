#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate scaling factors (f1, f2) using a reference module's measured electrical output.

Public release note
-------------------
This repository exposes the calibration workflow and the software interfaces, while omitting the
patent-protected iterative inversion. The forward model used here is the public stub in
`Iteration_module_workflow.py` + `calculate_J0_Rs.py`.

You can still run this script end-to-end to validate the calibration/control flow. Replacing the
stub with the private inversion module will yield the full performance.

Typical use
-----------
- Provide at least Pmpp of a reference module. Providing Voc (or Vmpp) is strongly recommended to
  constrain the 2D fit (f1, f2).

Example
-------
python calibrate_f1_f2_from_pmpp.py --image "ref_module_EL.png" --pmpp 530.2 --voc 41.6 --downsample 4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares

from Iteration_module_workflow import process_one_image


DEFAULTS = {
    "image": "ref_module_EL.png",
    "pmpp": 530.2,
    "voc": 49.6,           # optional but recommended for 2D (f1,f2)
    "vmpp": None,          # optional alternative to Voc
    "Jph": 42.49,          # mA/cm^2
    "total_area": 165.62,  # cm^2
    "rows": 6,
    "cols": 24,
    "deta": 0.15,
    "split_col": 12,
    "submodule_rows": 2,
    "downsample": 4,
    "use_parallel": 0,
    "max_workers": None,
    "f1_init": 2.961928e-19,
    "f2_init": 2.048644e-11,
    "prior_weight": 0.15,
    "max_nfev": 20,
    "verbose": 2,
}


def _positive(x: Optional[float], name: str) -> float:
    if x is None or not (x > 0):
        raise ValueError(f"{name} must be a positive number, got {x!r}")
    return float(x)


def predict_metrics(
    img_path: Path,
    f1: float,
    f2: float,
    *,
    Jph: float,
    total_area: float,
    rows: int,
    cols: int,
    deta: float,
    split_col: int,
    submodule_rows: int,
    downsample: int,
    use_parallel: bool,
    max_workers: Optional[int],
) -> Dict[str, float]:
    from Iteration_module_workflow import GridSpec  # local import to keep module lightweight
    grid = GridSpec(
        num_rows=rows,
        num_cols=cols,
        deta=deta,
        split_col=split_col,
        submodule_rows=submodule_rows,
    )
    return process_one_image(
        img_path,
        Jph=Jph,
        total_area=total_area,
        grid=grid,
        f1=f1,
        f2=f2,
        downsample=downsample,
        use_parallel=use_parallel,
        max_workers=max_workers,
    )


def calibrate_f1_f2(
    img_path: Path,
    pmpp_meas: float,
    *,
    voc_meas: Optional[float] = None,
    vmpp_meas: Optional[float] = None,
    Jph: float,
    total_area: float,
    rows: int,
    cols: int,
    deta: float,
    split_col: int,
    submodule_rows: int,
    downsample: int,
    use_parallel: bool,
    max_workers: Optional[int],
    f1_init: float,
    f2_init: float,
    prior_weight: float,
    max_nfev: int,
    verbose: int,
) -> Tuple[float, float, Dict[str, float]]:
    pmpp_meas = _positive(pmpp_meas, "pmpp_meas")
    if voc_meas is not None:
        voc_meas = _positive(voc_meas, "voc_meas")
    if vmpp_meas is not None:
        vmpp_meas = _positive(vmpp_meas, "vmpp_meas")

    f1_init = _positive(f1_init, "f1_init")
    f2_init = _positive(f2_init, "f2_init")

    # Broad but strictly positive bounds in log-space
    f1_min, f1_max = 1e-25, 1e-15
    f2_min, f2_max = 1e-15, 1e-5

    x0 = np.array([math.log(f1_init), math.log(f2_init)], dtype=float)
    lb = np.array([math.log(f1_min), math.log(f2_min)], dtype=float)
    ub = np.array([math.log(f1_max), math.log(f2_max)], dtype=float)

    use_voc = voc_meas is not None
    use_vmpp = (vmpp_meas is not None) and (not use_voc)

    def residuals(x: np.ndarray) -> np.ndarray:
        f1 = float(math.exp(x[0]))
        f2 = float(math.exp(x[1]))

        pred = predict_metrics(
            img_path,
            f1=f1,
            f2=f2,
            Jph=Jph,
            total_area=total_area,
            rows=rows,
            cols=cols,
            deta=deta,
            split_col=split_col,
            submodule_rows=submodule_rows,
            downsample=downsample,
            use_parallel=use_parallel,
            max_workers=max_workers,
        )

        r = [(pred["Pmpp"] - pmpp_meas) / pmpp_meas]
        if use_voc:
            r.append((pred["Voc"] - voc_meas) / voc_meas)
        elif use_vmpp:
            r.append((pred["Vmpp"] - vmpp_meas) / vmpp_meas)
        else:
            # Under-determined: keep (f1,f2) near initial guesses to pick a stable solution.
            r.append(prior_weight * (math.log(f1 / f1_init)))
            r.append(prior_weight * (math.log(f2 / f2_init)))
        return np.array(r, dtype=float)

    res = least_squares(
        residuals,
        x0=x0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max(1, int(max_nfev)),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=int(verbose),
    )

    f1_hat = float(math.exp(res.x[0]))
    f2_hat = float(math.exp(res.x[1]))
    pred = predict_metrics(
        img_path,
        f1=f1_hat,
        f2=f2_hat,
        Jph=Jph,
        total_area=total_area,
        rows=rows,
        cols=cols,
        deta=deta,
        split_col=split_col,
        submodule_rows=submodule_rows,
        downsample=downsample,
        use_parallel=use_parallel,
        max_workers=max_workers,
    )
    return f1_hat, f2_hat, pred


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate f1,f2 from measured module output (public workflow).")
    p.add_argument("--image", type=str, default=DEFAULTS["image"])
    p.add_argument("--pmpp", type=float, default=DEFAULTS["pmpp"])
    p.add_argument("--voc", type=float, default=DEFAULTS["voc"])
    p.add_argument("--vmpp", type=float, default=DEFAULTS["vmpp"])

    p.add_argument("--Jph", type=float, default=DEFAULTS["Jph"])
    p.add_argument("--total-area", type=float, default=DEFAULTS["total_area"])
    p.add_argument("--rows", type=int, default=DEFAULTS["rows"])
    p.add_argument("--cols", type=int, default=DEFAULTS["cols"])
    p.add_argument("--deta", type=float, default=DEFAULTS["deta"])
    p.add_argument("--split-col", type=int, default=DEFAULTS["split_col"])
    p.add_argument("--submodule-rows", type=int, default=DEFAULTS["submodule_rows"])

    p.add_argument("--downsample", type=int, default=DEFAULTS["downsample"])
    p.add_argument("--use-parallel", type=int, default=DEFAULTS["use_parallel"])
    p.add_argument("--max-workers", type=int, default=DEFAULTS["max_workers"])

    p.add_argument("--f1-init", type=float, default=DEFAULTS["f1_init"])
    p.add_argument("--f2-init", type=float, default=DEFAULTS["f2_init"])
    p.add_argument("--prior-weight", type=float, default=DEFAULTS["prior_weight"])
    p.add_argument("--max-nfev", type=int, default=DEFAULTS["max_nfev"])
    p.add_argument("--verbose", type=int, default=DEFAULTS["verbose"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    img_path = Path(args.image)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    f1_hat, f2_hat, pred = calibrate_f1_f2(
        img_path,
        pmpp_meas=args.pmpp,
        voc_meas=args.voc,
        vmpp_meas=args.vmpp,
        Jph=args.Jph,
        total_area=args.total_area,
        rows=args.rows,
        cols=args.cols,
        deta=args.deta,
        split_col=args.split_col,
        submodule_rows=args.submodule_rows,
        downsample=max(1, int(args.downsample)),
        use_parallel=bool(args.use_parallel),
        max_workers=args.max_workers,
        f1_init=args.f1_init,
        f2_init=args.f2_init,
        prior_weight=args.prior_weight,
        max_nfev=args.max_nfev,
        verbose=args.verbose,
    )

    print("\nCalibration result (public workflow)")
    print(f"  f1 = {f1_hat:.6e}")
    print(f"  f2 = {f2_hat:.6e}")
    print("\nForward-model check (at calibrated f1,f2)")
    print(f"  Pmpp_pred = {pred['Pmpp']:.3f} W")
    print(f"  Voc_pred  = {pred['Voc']:.3f} V")
    print(f"  Vmpp_pred = {pred['Vmpp']:.3f} V")
    print(f"  Impp_pred = {pred['Impp']:.3f} A")
    print(f"  FF_pred   = {pred['FF']:.4f}")


if __name__ == "__main__":
    main()
