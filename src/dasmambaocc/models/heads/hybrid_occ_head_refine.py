from typing import Optional, Sequence

import einops
import torch
import torch.nn.functional as F

from mmdet3d.models.builder import HEADS
from mmdet3d.models.heads.occ.bev_occ_head import BEVOCCHead2D

from ..modules.detection_guidance import DetectionGuidanceProjector
from ..modules.hard_negative_mining import hard_negative_suppression_loss
from ..modules.mamba_refine_subhead import MambaRefinementSubHead
from ..modules.temporal_memory import FeatureMemoryBank


@HEADS.register_module()
class HybridBEVOCCHead2DRefine(BEVOCCHead2D):
    """Detection-guided occupancy head with temporal memory and OccMamba-style refinement."""

    def __init__(
        self,
        *args,
        guidance_gain: float = 1.5,
        guidance_threshold: float = 0.2,
        guidance_blur_kernel: int = 3,
        guidance_resize_mode: str = "nearest",
        guidance_source_x_range: Optional[tuple] = None,
        guidance_source_y_range: Optional[tuple] = None,
        guidance_target_x_range: Optional[tuple] = None,
        guidance_target_y_range: Optional[tuple] = None,
        use_temporal_memory: bool = True,
        temporal_momentum: float = 0.9,
        temporal_blend: float = 0.25,
        max_memory_entries: int = 2048,
        temporal_max_timestamp_gap: int = 2_000_000,
        temporal_enable_pose_warp: bool = True,
        temporal_x_range: tuple = (-40.0, 40.0),
        temporal_y_range: tuple = (-40.0, 40.0),
        hard_negative_weight: float = 0.2,
        hard_negative_threshold: float = 0.15,
        empty_class_idx: Optional[int] = None,
        hard_negative_foreground_classes: Optional[Sequence[int]] = None,
        enable_refine_subhead: bool = True,
        refine_subhead: Optional[dict] = None,
        coarse_aux_weight: float = 0.2,
        refine_consistency_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.guidance_gain = guidance_gain
        self.guidance_threshold = guidance_threshold
        self.guidance_resize_mode = guidance_resize_mode
        self.guidance_projector = DetectionGuidanceProjector(
            blur_kernel=guidance_blur_kernel,
            interpolate_mode=guidance_resize_mode,
            source_x_range=guidance_source_x_range,
            source_y_range=guidance_source_y_range,
            target_x_range=guidance_target_x_range,
            target_y_range=guidance_target_y_range,
        )

        self.use_temporal_memory = use_temporal_memory
        self.temporal_memory = FeatureMemoryBank(
            momentum=temporal_momentum,
            blend=temporal_blend,
            max_entries=max_memory_entries,
            max_timestamp_gap=temporal_max_timestamp_gap,
            enable_pose_warp=temporal_enable_pose_warp,
            x_range=temporal_x_range,
            y_range=temporal_y_range,
        )

        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_threshold = hard_negative_threshold
        if empty_class_idx is None:
            empty_class_idx = self.num_classes - 1
        if not 0 <= int(empty_class_idx) < self.num_classes:
            raise ValueError(
                f"empty_class_idx must be in [0, {self.num_classes - 1}], got {empty_class_idx}"
            )
        self.empty_class_idx = int(empty_class_idx)
        if hard_negative_foreground_classes is None:
            self.hard_negative_foreground_classes = None
        else:
            self.hard_negative_foreground_classes = [int(x) for x in hard_negative_foreground_classes]

        refine_subhead = dict(refine_subhead or {})
        self.enable_refine_subhead = enable_refine_subhead
        if self.enable_refine_subhead:
            self.refine_subhead = MambaRefinementSubHead(
                bev_channels=self.in_dim,
                num_classes=self.num_classes,
                dz=self.Dz,
                **refine_subhead,
            )
        else:
            self.refine_subhead = None

        self.coarse_aux_weight = coarse_aux_weight
        self.refine_consistency_weight = refine_consistency_weight

        self._cached_guidance_xy: Optional[torch.Tensor] = None
        self._cached_occ_coarse: Optional[torch.Tensor] = None

    @staticmethod
    def _safe_inverse(mat: torch.Tensor) -> torch.Tensor:
        try:
            return torch.linalg.inv(mat)
        except RuntimeError:
            return torch.linalg.pinv(mat)

    def _warp_guidance_to_occ_space(
        self,
        guidance: Optional[torch.Tensor],
        target_hw: Sequence[int],
        lidar_aug_matrix: Optional[torch.Tensor],
        lidar2ego: Optional[torch.Tensor],
        occ_aug_matrix: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if guidance is None:
            return None
        if guidance.dim() != 4 or guidance.shape[1] != 1:
            return None
        if (
            lidar_aug_matrix is None
            or lidar2ego is None
            or occ_aug_matrix is None
            or not torch.is_tensor(lidar_aug_matrix)
            or not torch.is_tensor(lidar2ego)
            or not torch.is_tensor(occ_aug_matrix)
        ):
            return None

        b, _, src_h, src_w = guidance.shape
        tgt_h, tgt_w = int(target_hw[0]), int(target_hw[1])
        device = guidance.device
        math_dtype = torch.float32

        src_x_range = self.guidance_projector.source_x_range
        src_y_range = self.guidance_projector.source_y_range
        tgt_x_range = self.guidance_projector.target_x_range
        tgt_y_range = self.guidance_projector.target_y_range
        if src_x_range is None or src_y_range is None or tgt_x_range is None or tgt_y_range is None:
            return None

        x_step = (float(tgt_x_range[1]) - float(tgt_x_range[0])) / max(tgt_w, 1)
        y_step = (float(tgt_y_range[1]) - float(tgt_y_range[0])) / max(tgt_h, 1)
        x_coords = torch.arange(tgt_w, device=device, dtype=math_dtype)
        y_coords = torch.arange(tgt_h, device=device, dtype=math_dtype)
        x_coords = float(tgt_x_range[0]) + (x_coords + 0.5) * x_step
        y_coords = float(tgt_y_range[0]) + (y_coords + 0.5) * y_step
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        zeros = torch.zeros_like(xx)
        pts_occ = torch.stack([xx, yy, zeros], dim=-1).view(1, -1, 3).expand(b, -1, -1)

        occ = occ_aug_matrix.to(device=device, dtype=math_dtype)
        l2e = lidar2ego.to(device=device, dtype=math_dtype)
        laug = lidar_aug_matrix.to(device=device, dtype=math_dtype)
        if occ.shape[0] != b:
            occ = occ.expand(b, -1, -1)
        if l2e.shape[0] != b:
            l2e = l2e.expand(b, -1, -1)
        if laug.shape[0] != b:
            laug = laug.expand(b, -1, -1)

        occ_rot = occ[:, :3, :3]
        occ_trans = occ[:, :3, 3]
        occ_inv = self._safe_inverse(occ_rot)
        pts_ego = torch.bmm((pts_occ - occ_trans.unsqueeze(1)), occ_inv.transpose(1, 2))

        l2e_rot = l2e[:, :3, :3]
        l2e_trans = l2e[:, :3, 3]
        l2e_inv = self._safe_inverse(l2e_rot)
        pts_lidar = torch.bmm((pts_ego - l2e_trans.unsqueeze(1)), l2e_inv.transpose(1, 2))

        laug_rot = laug[:, :3, :3]
        laug_trans = laug[:, :3, 3]
        pts_src = torch.bmm(pts_lidar, laug_rot.transpose(1, 2)) + laug_trans.unsqueeze(1)

        xs = pts_src[..., 0].view(b, tgt_h, tgt_w)
        ys = pts_src[..., 1].view(b, tgt_h, tgt_w)
        src_x_denom = max(float(src_x_range[1] - src_x_range[0]), 1e-6)
        src_y_denom = max(float(src_y_range[1] - src_y_range[0]), 1e-6)
        xs = (xs - float(src_x_range[0])) / src_x_denom
        ys = (ys - float(src_y_range[0])) / src_y_denom
        xs = xs * 2.0 - 1.0
        ys = ys * 2.0 - 1.0
        grid = torch.stack([xs, ys], dim=-1)

        mode = "nearest" if self.guidance_resize_mode == "nearest" else "bilinear"
        warped = F.grid_sample(guidance, grid, mode=mode, padding_mode="zeros", align_corners=False)
        warped = torch.nan_to_num(warped, nan=0.0, posinf=1.0, neginf=0.0)
        if warped.shape[-2:] != (tgt_h, tgt_w):
            return None
        return warped

    def forward(
        self,
        img_feats,
        lidar_aug_matrix=None,
        lidar2ego=None,
        occ_aug_matrix=None,
        det_guidance_logits: Optional[torch.Tensor] = None,
        metas=None,
        camera_ego2global: Optional[torch.Tensor] = None,
        ego2global: Optional[torch.Tensor] = None,
        mask_camera: Optional[torch.Tensor] = None,
    ):
        if isinstance(img_feats, list):
            assert len(img_feats) == 1
            img_feats = img_feats[0]

        img_feats = einops.rearrange(img_feats, "bs c w h -> bs c h w")
        img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=1e4, neginf=-1e4)

        # Temporal memory runs before occupancy-grid coordinate transform.
        # Input features are still lidar-augmented from upstream BEV lifting/fusion;
        # memory bank handles per-frame lidar_aug_matrix when pose-warping history.
        if self.use_temporal_memory:
            if ego2global is None:
                ego2global = camera_ego2global
            img_feats = self.temporal_memory(
                img_feats,
                metas=metas,
                ego2global=ego2global,
                lidar_aug_matrix=lidar_aug_matrix,
            )
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.coordinate_transform is not None:
            img_feats = self.coordinate_transform(img_feats, lidar_aug_matrix, lidar2ego, occ_aug_matrix)
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=1e4, neginf=-1e4)

        guidance = None
        if det_guidance_logits is not None and det_guidance_logits.dim() >= 3:
            src_hw = tuple(det_guidance_logits.shape[-2:])
            guidance_src = self.guidance_projector(det_guidance_logits, src_hw)
            guidance = self._warp_guidance_to_occ_space(
                guidance=guidance_src,
                target_hw=img_feats.shape[-2:],
                lidar_aug_matrix=lidar_aug_matrix,
                lidar2ego=lidar2ego,
                occ_aug_matrix=occ_aug_matrix,
            )
        if guidance is None:
            guidance = self.guidance_projector(det_guidance_logits, img_feats.shape[-2:])

        if guidance is not None:
            guidance = torch.nan_to_num(guidance, nan=0.0, posinf=1.0, neginf=0.0)
            hard_gate = (guidance >= self.guidance_threshold).to(img_feats.dtype)
            soft_gate = 0.5 * guidance
            img_feats = img_feats * (1.0 + self.guidance_gain * hard_gate * guidance + soft_gate)
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=1e4, neginf=-1e4)
            self._cached_guidance_xy = guidance.squeeze(1).permute(0, 2, 1).unsqueeze(-1).detach()
        else:
            self._cached_guidance_xy = None

        occ_coarse = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, dx, dy = occ_coarse.shape[:3]
        if self.use_predicter:
            occ_coarse = self.predicter(occ_coarse)
            occ_coarse = occ_coarse.view(bs, dx, dy, self.Dz, self.num_classes)

        occ_coarse = torch.nan_to_num(occ_coarse, nan=0.0, posinf=1e4, neginf=-1e4)
        self._cached_occ_coarse = occ_coarse

        if self.enable_refine_subhead and self.refine_subhead is not None:
            occ_pred = self.refine_subhead(
                img_feats,
                occ_coarse,
                occ_aug_matrix=occ_aug_matrix,
                visibility_mask=mask_camera,
            )
        else:
            occ_pred = occ_coarse

        return torch.nan_to_num(occ_pred, nan=0.0, posinf=1e4, neginf=-1e4)

    def _resize_cached_guidance(self, occ_pred: torch.Tensor) -> Optional[torch.Tensor]:
        if self._cached_guidance_xy is None:
            return None

        if self._cached_guidance_xy.shape[:3] == occ_pred.shape[:3]:
            return self._cached_guidance_xy

        guidance = self._cached_guidance_xy.squeeze(-1).permute(0, 2, 1).unsqueeze(1)
        if self.guidance_resize_mode == "nearest":
            guidance = F.interpolate(
                guidance,
                size=(occ_pred.shape[2], occ_pred.shape[1]),
                mode="nearest",
            )
        else:
            guidance = F.interpolate(
                guidance,
                size=(occ_pred.shape[2], occ_pred.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
        return guidance.squeeze(1).permute(0, 2, 1).unsqueeze(-1)

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        occ_pred = torch.nan_to_num(occ_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_dict = super().loss(occ_pred, voxel_semantics, mask_camera)

        if (
            self.coarse_aux_weight > 0
            and self._cached_occ_coarse is not None
            and self._cached_occ_coarse.shape == occ_pred.shape
            and self._cached_occ_coarse.data_ptr() != occ_pred.data_ptr()
        ):
            coarse_loss = super().loss(self._cached_occ_coarse, voxel_semantics, mask_camera)
            loss_dict["loss_occ_coarse_aux"] = coarse_loss["loss_occ"] * self.coarse_aux_weight

        if (
            self.refine_consistency_weight > 0
            and self._cached_occ_coarse is not None
            and self._cached_occ_coarse.shape == occ_pred.shape
            and self._cached_occ_coarse.data_ptr() != occ_pred.data_ptr()
        ):
            # Keep coarse branch as teacher by default to avoid destabilizing
            # pretrained coarse logits with online student errors.
            log_p = F.log_softmax(occ_pred, dim=-1)
            q = F.softmax(self._cached_occ_coarse.detach(), dim=-1)
            kl = F.kl_div(log_p, q, reduction="none").sum(dim=-1)
            if mask_camera is not None:
                valid_mask = mask_camera.to(dtype=torch.bool)
                if valid_mask.shape == kl.shape and valid_mask.any():
                    kl = kl[valid_mask].mean()
                else:
                    kl = kl.mean()
            else:
                kl = kl.mean()
            loss_dict["loss_occ_refine_consistency"] = kl * self.refine_consistency_weight

        guidance = self._resize_cached_guidance(occ_pred)
        if self.hard_negative_weight > 0 and guidance is not None:
            guidance = torch.nan_to_num(guidance, nan=0.0, posinf=1.0, neginf=0.0)
            loss_dict["loss_occ_hnm"] = hard_negative_suppression_loss(
                occ_pred=occ_pred,
                det_guidance_xy=guidance,
                mask_camera=mask_camera,
                voxel_semantics=voxel_semantics,
                empty_class_idx=self.empty_class_idx,
                guidance_threshold=self.hard_negative_threshold,
                loss_weight=self.hard_negative_weight,
                foreground_class_indices=self.hard_negative_foreground_classes,
            )

        if self.enable_refine_subhead and self.refine_subhead is not None:
            loss_dict["refine_use_mamba"] = occ_pred.new_tensor(float(self.refine_subhead.use_mamba))

        return loss_dict
