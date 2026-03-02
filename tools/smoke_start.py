#!/usr/bin/env python3
import argparse
import compileall
import contextlib
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from mmcv import Config
from torchpack.utils.config import configs

from bootstrap_paths import bootstrap_paths


DEFAULT_CONFIGS = [
    "configs/nuscenes/occ3d/dasmambaocc_occ3d_nus_w_mask.yaml",
    "configs/nuscenes/occ3d/dasmambaocc_occ3d_nus_w_mask_1gpu_long.yaml",
    "configs/nuscenes/occ3d/ablations/dasmambaocc_occ3d_nus_ablation_baseline.yaml",
    "configs/nuscenes/occ3d/ablations/dasmambaocc_occ3d_nus_ablation_lift_only.yaml",
    "configs/nuscenes/occ3d/ablations/dasmambaocc_occ3d_nus_ablation_full.yaml",
]


def _echo(msg: str) -> None:
    print(f"[SMOKE] {msg}")


def _check_compile(project_root: Path) -> None:
    _echo("Compiling python sources")
    targets = [project_root / "src" / "dasmambaocc", project_root / "tools"]
    for target in targets:
        if not compileall.compile_dir(str(target), quiet=1):
            raise RuntimeError(f"Python compile failed under {target}")


def _load_cfg(cfg_path: Path) -> Config:
    from mmdet3d.utils import recursive_eval

    configs.clear()
    configs.load(str(cfg_path), recursive=True)
    return Config(recursive_eval(configs), filename=str(cfg_path))


def _check_config_and_model(cfg_paths: List[Path], skip_model_build: bool, require_mamba: bool) -> None:
    from mmdet3d.models import build_model

    logging.getLogger("mmcv").setLevel(logging.WARNING)

    for cfg_path in cfg_paths:
        _echo(f"Loading config: {cfg_path}")
        cfg = _load_cfg(cfg_path)

        required_keys = ("model", "data", "runner")
        for key in required_keys:
            if key not in cfg:
                raise RuntimeError(f"Config missing key '{key}': {cfg_path}")

        occ_cfg = cfg.model.get("heads", {}).get("occ", {})
        use_temporal_memory = bool(occ_cfg.get("use_temporal_memory", False))
        train_type = cfg.data.get("train", {}).get("type", "")
        use_cbgs = bool(cfg.get("use_cbgs", train_type == "CBGSDataset"))
        if use_temporal_memory and use_cbgs:
            raise RuntimeError(
                f"Temporal memory is enabled but CBGS is active in {cfg_path}. "
                "Set use_cbgs=false or disable temporal memory."
            )

        if skip_model_build:
            continue

        # Keep smoke checks offline-friendly: skip pretrained backbone downloads.
        camera_cfg = cfg.model.get("encoders", {}).get("camera", None)
        if camera_cfg is not None:
            backbone_cfg = camera_cfg.get("backbone", None)
            if backbone_cfg is not None:
                backbone_cfg.pop("init_cfg", None)
                backbone_cfg["pretrained"] = None

        _echo(f"Building model: {cfg_path.name}")
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                model = build_model(cfg.model)
        if model.__class__.__name__ != "HybridBEVFusionPlus":
            raise RuntimeError(f"Unexpected model type for {cfg_path}: {model.__class__.__name__}")
        if "occ" not in model.heads:
            raise RuntimeError(f"Occ head missing for {cfg_path}")
        if model.heads["occ"].__class__.__name__ != "HybridBEVOCCHead2DRefine":
            raise RuntimeError(f"Unexpected occ head type for {cfg_path}: {model.heads['occ'].__class__.__name__}")
        occ_head = model.heads["occ"]
        refine_subhead = getattr(occ_head, "refine_subhead", None)
        if refine_subhead is not None:
            requested_mamba = bool(getattr(refine_subhead, "requested_mamba", False))
            active_mamba = bool(getattr(refine_subhead, "use_mamba", False))
            if requested_mamba and not active_mamba:
                msg = f"Mamba was requested but unavailable in {cfg_path.name}; fallback mixer is active."
                if require_mamba:
                    raise RuntimeError(msg)
                _echo(f"WARNING: {msg}")


def _check_guidance_and_temporal_modules() -> None:
    from dasmambaocc.models.fusion_models.hybrid_bevfusion_plus import HybridBEVFusionPlus
    from dasmambaocc.models.modules.detection_guidance import DetectionGuidanceProjector
    from dasmambaocc.models.vtransforms.adaptive_lift_vtransform import AdaptiveLiftingBEVTransformV2
    from dasmambaocc.models.modules.temporal_memory import FeatureMemoryBank

    _echo("Checking detection guidance extraction")
    nested_pred = (
        [{"heatmap": torch.full((1, 3, 2, 2), -5.0)}],
        {"misc": ({"heatmap": torch.full((1, 2, 2, 2), -4.0)},)},
    )
    guidance_logits = HybridBEVFusionPlus._extract_detection_guidance(nested_pred)
    if guidance_logits is None:
        raise RuntimeError("Detection guidance extraction returned None for nested detector outputs")
    if guidance_logits.shape != (1, 1, 2, 2):
        raise RuntimeError(f"Unexpected detection guidance shape: {tuple(guidance_logits.shape)}")
    if not bool((guidance_logits < 0).all()):
        raise RuntimeError("Detection guidance extraction should keep raw negative logits before projector sigmoid")

    projector = DetectionGuidanceProjector(blur_kernel=3)
    guidance_prob = projector(guidance_logits, target_hw=(2, 2))
    if guidance_prob is None:
        raise RuntimeError("Guidance projector returned None for valid logits")
    if float(guidance_prob.max()) > 1.0 or float(guidance_prob.min()) < 0.0:
        raise RuntimeError("Guidance projector output is out of [0, 1]")
    nearest_projector = DetectionGuidanceProjector(blur_kernel=1, interpolate_mode="nearest")
    coarse = torch.tensor([[[[10.0, -10.0], [-10.0, -10.0]]]])
    upsampled = nearest_projector(coarse, target_hw=(4, 4))
    if not bool((upsampled[:, :, :2, :2] > 0.9).all() and (upsampled[:, :, 2:, 2:] < 0.1).all()):
        raise RuntimeError("Nearest guidance resize check failed")

    range_projector = DetectionGuidanceProjector(
        blur_kernel=1,
        interpolate_mode="nearest",
        source_x_range=[-54.0, 54.0],
        source_y_range=[-54.0, 54.0],
        target_x_range=[-40.0, 40.0],
        target_y_range=[-40.0, 40.0],
    )
    src_logits = torch.full((1, 1, 180, 180), -20.0)
    src_logits[0, 0, 90, 90] = 20.0     # in-range center peak
    src_logits[0, 0, 90, 179] = 20.0    # out-of-range edge peak
    cropped = range_projector(src_logits, target_hw=(200, 200))
    if cropped is None:
        raise RuntimeError("Range-aware guidance projector returned None")
    if float(cropped[0, 0, 100, 100]) < 0.9:
        raise RuntimeError("Range-aware guidance projector lost center in-range peak")
    if float(cropped[0, 0, :, 199].max()) > 0.1:
        raise RuntimeError("Range-aware guidance projector failed to crop out-of-range edge peak")

    intr = torch.zeros(1, 1, 4, 4)
    intr[..., 0, 0] = 1000.0
    intr[..., 1, 1] = 1000.0
    intr[..., 0, 2] = 500.0
    intr[..., 1, 2] = 500.0
    intr[..., 2, 2] = 1.0

    aug = torch.eye(4).view(1, 1, 4, 4).clone()
    aug[..., 0, 0] = 1.2
    aug[..., 1, 1] = 0.8
    aug[..., 0, 3] = 10.0
    aug[..., 1, 3] = -5.0

    cam_vec = AdaptiveLiftingBEVTransformV2._camera_condition_vector(intr, aug)
    if cam_vec is None or cam_vec.shape[-1] != 8:
        raise RuntimeError("Camera condition vector check failed: invalid output shape")
    aug_mag = float(cam_vec[..., 4:].abs().mean())
    if aug_mag < 0.1:
        raise RuntimeError("Camera condition vector check failed: augmentation terms were over-suppressed")

    _echo("Checking temporal memory gradient flow")
    bank = FeatureMemoryBank(momentum=0.9, blend=0.25, max_entries=16)
    feats = torch.randn(2, 8, 4, 4, requires_grad=True)
    metas = [{"scene_token": "scene_a"}, {"scene_token": "scene_b"}]
    fused = bank(feats, metas=metas)
    if fused.shape != feats.shape:
        raise RuntimeError("Temporal memory output shape mismatch")
    fused.sum().backward()
    if feats.grad is None or not torch.isfinite(feats.grad).all():
        raise RuntimeError("Temporal memory blocked or corrupted gradients")
    if float(feats.grad.abs().sum()) <= 0:
        raise RuntimeError("Temporal memory gradient norm is zero")


def _check_refine_and_hnm() -> None:
    from dasmambaocc.models.modules.hard_negative_mining import hard_negative_suppression_loss
    from dasmambaocc.models.modules.mamba_refine_subhead import MambaRefinementSubHead

    _echo("Checking refinement sub-head and hard-negative loss")
    refine = MambaRefinementSubHead(
        bev_channels=16,
        num_classes=18,
        dz=4,
        hidden_dim=32,
        num_layers=1,
        use_mamba=False,
        scan_orders=("xy", "yx"),
    )
    bev_feat = torch.randn(2, 16, 6, 5, requires_grad=True)
    occ_logits = torch.randn(2, 5, 6, 4, 18, requires_grad=True)
    occ_refined = refine(bev_feat, occ_logits)
    if occ_refined.shape != occ_logits.shape:
        raise RuntimeError("Refinement sub-head changed occupancy tensor shape")
    occ_refined.mean().backward()
    if bev_feat.grad is None or occ_logits.grad is None:
        raise RuntimeError("Refinement sub-head gradient flow is broken")

    det_guidance_xy = torch.zeros(2, 5, 6, 1)
    mask_camera = torch.ones(2, 5, 6, 4, dtype=torch.bool)
    hnm = hard_negative_suppression_loss(
        occ_pred=occ_refined.detach(),
        det_guidance_xy=det_guidance_xy,
        mask_camera=mask_camera,
        empty_class_idx=17,
        guidance_threshold=0.15,
        loss_weight=0.2,
    )
    if not torch.isfinite(hnm):
        raise RuntimeError("Hard-negative suppression loss returned non-finite value")


def _check_pipeline_contracts(daocc_root: Path) -> None:
    from dasmambaocc.datasets.pipelines.image_normalize_safe import ImageNormalizeSafe

    _echo("Checking pipeline contracts (occ_aug_matrix, flip sync, RGB normalize)")

    # 1) Ensure LoadOccGTFromFile emits occ_aug_matrix in DAOcc pipeline source.
    loading_src = (daocc_root / "mmdet3d" / "datasets" / "pipelines" / "loading.py").read_text()
    if 'results["occ_aug_matrix"] = np.eye(4).astype(np.float32)' not in loading_src:
        raise RuntimeError("DAOcc LoadOccGTFromFile contract changed: occ_aug_matrix assignment missing")

    # 2) Ensure RandomFlip3D flips voxel_semantics and masks in DAOcc source.
    flip_src = (daocc_root / "mmdet3d" / "datasets" / "pipelines" / "transforms_3d.py").read_text()
    required_flip_snippets = [
        "data['voxel_semantics'] = data['voxel_semantics'][:, ::-1, :].copy()",
        "data['mask_lidar'] = data['mask_lidar'][:, ::-1, :].copy()",
        "data['mask_camera'] = data['mask_camera'][:, ::-1, :].copy()",
        "data['voxel_semantics'] = data['voxel_semantics'][::-1, :, :].copy()",
        "data['mask_lidar'] = data['mask_lidar'][::-1, :, :].copy()",
        "data['mask_camera'] = data['mask_camera'][::-1, :, :].copy()",
    ]
    for snippet in required_flip_snippets:
        if snippet not in flip_src:
            raise RuntimeError(f"DAOcc RandomFlip3D contract changed: missing '{snippet}'")

    # 3) BGR ndarray should be converted to RGB when to_rgb=True.
    normalize = ImageNormalizeSafe(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # red channel in BGR storage
    normalized = normalize({"img": [bgr]})["img"][0]
    if float(normalized[2].mean()) > 0.1:
        raise RuntimeError("ImageNormalizeSafe to_rgb conversion failed (blue channel remained active)")
    if float(normalized[0].mean()) < 0.9:
        raise RuntimeError("ImageNormalizeSafe to_rgb conversion failed (red channel not activated)")

    # 4) Ensure map-output packaging in eval mode handles gt_masks_bev=None.
    fusion_src = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "dasmambaocc"
        / "models"
        / "fusion_models"
        / "hybrid_bevfusion_plus.py"
    ).read_text()
    expected_guard = '"gt_masks_bev": gt_masks_bev[k].cpu() if gt_masks_bev is not None else None'
    if expected_guard not in fusion_src:
        raise RuntimeError("HybridBEVFusionPlus map eval contract changed: missing gt_masks_bev None-guard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DASMambaOcc startup smoke checks")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config paths to validate/build",
    )
    parser.add_argument(
        "--skip-model-build",
        action="store_true",
        help="Skip model construction checks (config + module checks still run)",
    )
    parser.add_argument(
        "--require-mamba",
        action="store_true",
        help="Fail if a config requests Mamba refinement but runtime falls back to non-Mamba mixer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root, daocc_root = bootstrap_paths()
    project_root = Path(project_root).resolve()
    daocc_root = Path(daocc_root).resolve()

    import dasmambaocc  # noqa: F401

    cfg_paths = []
    for cfg in args.configs:
        cfg_path = Path(cfg).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (project_root / cfg_path).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg_paths.append(cfg_path)

    _check_compile(project_root)
    _check_config_and_model(
        cfg_paths,
        skip_model_build=args.skip_model_build,
        require_mamba=args.require_mamba,
    )
    _check_pipeline_contracts(daocc_root)
    _check_guidance_and_temporal_modules()
    _check_refine_and_hnm()
    _echo("All smoke checks passed")


if __name__ == "__main__":
    main()
