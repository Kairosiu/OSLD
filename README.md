# One-shot luminescence diagnostics (public workflow)

This repository is a **public / level-2** release intended for peer review, reproducibility of the **workflow**, and
integration into downstream tooling (data IO, plotting, automation). 
## What is included
- A runnable **end-to-end pipeline** from a single luminescence image to a reconstructed module I–V curve,
  using a **public stub** in place of the proprietary solver.
- A **calibration script** that solves for `(f1, f2)` from a reference module's measured `Pmpp` (and optionally `Voc`).

## What is intentionally omitted
- The proprietary iterative solver that recovers `J01_xy`, `J02_xy`, and `Rs_xy` by matching measured and reconstructed
  luminescence fields under a fixed injection condition. That solver is the core of a pending patent.

## Files
- `calculate_J0_Rs_public_stub.py` — public drop-in replacement exposing the same function signature and return structure.
- `Iteration_module_workflow_public.py` — module-level workflow and series/parallel aggregation.
- `calibrate_f1_f2_from_pmpp_public.py` — 2D calibration of `f1` and `f2` from measured electrical output.

## Install
```bash
pip install -r requirements.txt
```

## Run the public workflow
```bash
python Iteration_module_workflow_public.py --image "ref_module_EL.png" --rows 6 --cols 24 --downsample 4
```

## Calibrate (f1, f2)
Providing `Voc` (or `Vmpp`) is recommended to constrain the 2D fit:
```bash
python calibrate_f1_f2_from_pmpp_public.py --image "ref_module_EL.png" --pmpp 530.2 --voc 41.6 --downsample 4
```

The full implementation will be made publicly available in a future release, subject to patent and publication processes.