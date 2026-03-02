from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import VTRANSFORMS
from mmdet3d.models.vtransforms.bev_ss import BEVTransformV2


@VTRANSFORMS.register_module()
class AdaptiveLiftingBEVTransformV2(BEVTransformV2):
    """DAOcc BEVTransformV2 with ALOcc-inspired adaptive lifting improvements."""

    def __init__(
        self,
        *args,
        use_camera_condition: bool = True,
        camera_gate_gain: float = 0.12,
        use_adaptive_view_weight: bool = True,
        use_geometry_denoise: bool = True,
        denoise_alpha: float = 0.2,
        denoise_kernel: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if denoise_kernel < 1 or denoise_kernel % 2 == 0:
            raise ValueError("denoise_kernel must be a positive odd integer")

        in_channels = int(kwargs.get("in_channels", 256))
        self.use_camera_condition = use_camera_condition
        self.camera_gate_gain = camera_gate_gain
        self.use_adaptive_view_weight = use_adaptive_view_weight
        self.use_geometry_denoise = use_geometry_denoise
        self.denoise_alpha = denoise_alpha
        self.denoise_kernel = denoise_kernel

        if self.use_camera_condition:
            self.camera_mlp = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, in_channels),
                nn.Tanh(),
            )
        else:
            self.camera_mlp = None

    @staticmethod
    def _camera_condition_vector(
        camera_intrinsics: Optional[torch.Tensor],
        img_aug_matrix: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if camera_intrinsics is None:
            return None

        intr = camera_intrinsics[..., :3, :3]
        fx = intr[..., 0, 0]
        fy = intr[..., 1, 1]
        cx = intr[..., 0, 2]
        cy = intr[..., 1, 2]

        if img_aug_matrix is None:
            sx = torch.ones_like(fx)
            sy = torch.ones_like(fy)
            tx = torch.zeros_like(cx)
            ty = torch.zeros_like(cy)
        else:
            sx = img_aug_matrix[..., 0, 0]
            sy = img_aug_matrix[..., 1, 1]
            tx = img_aug_matrix[..., 0, 3]
            ty = img_aug_matrix[..., 1, 3]

        intr_vec = torch.stack([fx, fy, cx, cy], dim=-1)
        scale_vec = torch.stack([sx, sy], dim=-1)
        trans_vec = torch.stack([tx, ty], dim=-1)

        intr_norm = intr_vec.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        scale_norm = scale_vec.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
        trans_norm = trans_vec.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)

        intr_vec = intr_vec / intr_norm
        scale_vec = scale_vec / scale_norm
        trans_vec = trans_vec / trans_norm
        return torch.cat([intr_vec, scale_vec, trans_vec], dim=-1)

    def _apply_adaptive_lifting(
        self,
        x: torch.Tensor,
        camera_intrinsics: Optional[torch.Tensor],
        img_aug_matrix: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # x: [B, N, C, H, W]
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.use_camera_condition and self.camera_mlp is not None:
            cam_vec = self._camera_condition_vector(camera_intrinsics, img_aug_matrix)
            if cam_vec is not None:
                gate = self.camera_mlp(cam_vec.float()).to(x.dtype)
                gate = gate.unsqueeze(-1).unsqueeze(-1)
                x = x * (1.0 + self.camera_gate_gain * gate)

        if self.use_adaptive_view_weight:
            # ALOcc-like adaptive per-view weighting based on feature confidence.
            score = x.abs().mean(dim=(2, 3, 4), keepdim=True)
            score = torch.softmax(score, dim=1)
            x = x * (1.0 + score)

        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    def _geometry_denoise(self, bev_feats: torch.Tensor) -> torch.Tensor:
        if not self.use_geometry_denoise or bev_feats.dim() != 4:
            return bev_feats

        pad = self.denoise_kernel // 2
        smooth = F.avg_pool2d(bev_feats, kernel_size=self.denoise_kernel, stride=1, padding=pad)
        edge = (bev_feats - smooth).abs().mean(dim=1, keepdim=True)
        blend = torch.sigmoid(-edge)
        alpha = self.denoise_alpha
        return bev_feats * (1.0 - alpha * blend) + smooth * (alpha * blend)

    def forward(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        x = self._apply_adaptive_lifting(x, camera_intrinsics, img_aug_matrix)
        bev = super().forward(
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            **kwargs,
        )
        bev = torch.nan_to_num(bev, nan=0.0, posinf=1e4, neginf=-1e4)
        bev = self._geometry_denoise(bev)
        return torch.nan_to_num(bev, nan=0.0, posinf=1e4, neginf=-1e4)
