from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionGuidanceProjector(nn.Module):
    """Projects detector logits into a smooth single-channel spatial prior."""

    def __init__(
        self,
        blur_kernel: int = 3,
        interpolate_mode: str = "nearest",
        source_x_range: Optional[Sequence[float]] = None,
        source_y_range: Optional[Sequence[float]] = None,
        target_x_range: Optional[Sequence[float]] = None,
        target_y_range: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if interpolate_mode not in ("nearest", "bilinear"):
            raise ValueError("interpolate_mode must be 'nearest' or 'bilinear'")
        self.blur_kernel = blur_kernel
        self.interpolate_mode = interpolate_mode
        self.source_x_range = self._validate_range(source_x_range, "source_x_range")
        self.source_y_range = self._validate_range(source_y_range, "source_y_range")
        self.target_x_range = self._validate_range(target_x_range, "target_x_range")
        self.target_y_range = self._validate_range(target_y_range, "target_y_range")

    @staticmethod
    def _validate_range(r: Optional[Sequence[float]], name: str) -> Optional[tuple]:
        if r is None:
            return None
        if len(r) != 2:
            raise ValueError(f"{name} must be a 2-element sequence [min, max]")
        r0 = float(r[0])
        r1 = float(r[1])
        if not r0 < r1:
            raise ValueError(f"{name} min must be smaller than max")
        return (r0, r1)

    def _has_physical_ranges(self) -> bool:
        return (
            self.source_x_range is not None
            and self.source_y_range is not None
            and self.target_x_range is not None
            and self.target_y_range is not None
        )

    def _resample_with_physical_ranges(self, x: torch.Tensor, target_hw: Sequence[int]) -> torch.Tensor:
        # x: [B, 1, H_src, W_src]
        b, _, _, _ = x.shape
        target_h = int(target_hw[0])
        target_w = int(target_hw[1])
        device = x.device
        dtype = x.dtype

        src_x_min, src_x_max = self.source_x_range
        src_y_min, src_y_max = self.source_y_range
        tgt_x_min, tgt_x_max = self.target_x_range
        tgt_y_min, tgt_y_max = self.target_y_range

        x_step = (tgt_x_max - tgt_x_min) / max(target_w, 1)
        y_step = (tgt_y_max - tgt_y_min) / max(target_h, 1)

        x_coords = torch.arange(target_w, device=device, dtype=dtype)
        y_coords = torch.arange(target_h, device=device, dtype=dtype)
        x_coords = tgt_x_min + (x_coords + 0.5) * x_step
        y_coords = tgt_y_min + (y_coords + 0.5) * y_step

        x_norm = (x_coords - src_x_min) / max(src_x_max - src_x_min, 1e-6)
        y_norm = (y_coords - src_y_min) / max(src_y_max - src_y_min, 1e-6)
        x_norm = x_norm * 2.0 - 1.0
        y_norm = y_norm * 2.0 - 1.0

        yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

        return F.grid_sample(
            x,
            grid,
            mode=self.interpolate_mode,
            padding_mode="zeros",
            align_corners=False,
        )

    def forward(
        self,
        guidance_logits: Optional[torch.Tensor],
        target_hw: Sequence[int],
    ) -> Optional[torch.Tensor]:
        if guidance_logits is None:
            return None

        x = torch.nan_to_num(guidance_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            return None

        if x.size(1) > 1:
            x = x.max(dim=1, keepdim=True).values

        x = x.clamp(-20.0, 20.0).sigmoid()
        if tuple(x.shape[-2:]) != tuple(target_hw):
            if self._has_physical_ranges():
                x = self._resample_with_physical_ranges(x, target_hw)
            else:
                if self.interpolate_mode == "nearest":
                    x = F.interpolate(x, size=target_hw, mode="nearest")
                else:
                    x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

        if self.blur_kernel > 1:
            pad = self.blur_kernel // 2
            # Use max pooling as morphological dilation to preserve sparse peaks
            # from small objects while still expanding support.
            x = F.max_pool2d(x, kernel_size=self.blur_kernel, stride=1, padding=pad)

        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x.clamp_(0.0, 1.0)
