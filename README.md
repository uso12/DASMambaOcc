# DASMambaOcc

`DASMambaOcc` is a DAOcc-based occupancy project that adds three high-ROI upgrades:

- fixed detection-guidance extraction from DAOcc `CenterHead` outputs
- ALOcc-inspired adaptive depth-lifting in view transform (`AdaptiveLiftingBEVTransformV2`)
- OccMamba-inspired refinement sub-head on top of coarse occupancy logits (`MambaRefinementSubHead`)

Primary target:
- keep DAOcc detection-assisted stability while improving geometry quality and long-range occupancy consistency.

## Base and Additions

Base runtime/backbone:
- DAOcc-style `BEVFusion` training/eval flow via `DAOCC_ROOT`

Added modules:
- `HybridBEVFusionPlus`
- `HybridBEVOCCHead2DRefine`
- `AdaptiveLiftingBEVTransformV2`
- `MambaRefinementSubHead`

## Layout

```text
DASMambaOcc/
├── configs/
├── docs/
├── env/
├── src/dasmambaocc/
├── tools/
├── data/
└── work_dirs/
```

## Quick Start

1. Install env: `docs/install.md`
2. Link data symlinks: `docs/data.md`
3. Train/eval: `docs/run.md`
4. Run smoke-start checks: `bash tools/smoke_start.sh`

## Notes

- Existing repos are treated as read-only references.
- All new code is created under `DASMambaOcc/` only.
- `tools/create_conda_env.sh` defaults to copying `daocc` into `dasmambaocc`.
