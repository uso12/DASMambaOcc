from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionGuidanceProjector(nn.Module):
    """Projects detector logits into a smooth single-channel spatial prior."""

    def __init__(self, blur_kernel: int = 3, interpolate_mode: str = "nearest") -> None:
        super().__init__()
        if blur_kernel < 1 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if interpolate_mode not in ("nearest", "bilinear"):
            raise ValueError("interpolate_mode must be 'nearest' or 'bilinear'")
        self.blur_kernel = blur_kernel
        self.interpolate_mode = interpolate_mode

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
            if self.interpolate_mode == "nearest":
                x = F.interpolate(x, size=target_hw, mode="nearest")
            else:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)

        if self.blur_kernel > 1:
            pad = self.blur_kernel // 2
            x = F.avg_pool2d(x, kernel_size=self.blur_kernel, stride=1, padding=pad)

        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x.clamp_(0.0, 1.0)
