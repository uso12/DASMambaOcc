from collections import OrderedDict
from typing import Any, Dict, Iterable
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMemoryBank(nn.Module):
    """Sequence-keyed EMA feature memory for lightweight temporal stabilization."""

    def __init__(
        self,
        momentum: float = 0.9,
        blend: float = 0.25,
        max_entries: int = 2048,
        max_timestamp_gap: int = 2_000_000,
        enable_pose_warp: bool = True,
        x_range: tuple = (-40.0, 40.0),
        y_range: tuple = (-40.0, 40.0),
    ) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if max_timestamp_gap < 0:
            raise ValueError("max_timestamp_gap must be >= 0")
        if len(x_range) != 2 or len(y_range) != 2:
            raise ValueError("x_range and y_range must be (min, max) tuples")
        if not (float(x_range[0]) < float(x_range[1]) and float(y_range[0]) < float(y_range[1])):
            raise ValueError("x_range/y_range min must be smaller than max")
        self.momentum = momentum
        self.blend = blend
        self.max_entries = max_entries
        self.max_timestamp_gap = int(max_timestamp_gap)
        self.enable_pose_warp = bool(enable_pose_warp)
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self._memory: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._last_timestamp: "OrderedDict[str, int]" = OrderedDict()
        self._pose_memory: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._warned_non_temporal_key = False
        self._warned_missing_pose = False

    @staticmethod
    def _key_from_meta(meta: Dict[str, Any], sample_idx: int) -> str:
        for key in ("sequence_group_idx", "scene_token", "scene_name"):
            if key in meta:
                return str(meta[key])

        lidar_path = meta.get("lidar_path", None)
        if isinstance(lidar_path, str) and lidar_path:
            file_name = lidar_path.rsplit("/", 1)[-1]
            if "__" in file_name:
                # NuScenes lidar file format keeps a stable route prefix before "__".
                return f"log_{file_name.split('__', 1)[0]}"
            return f"lidar_{file_name}"

        for key in ("sample_idx", "token"):
            if key in meta:
                return str(meta[key])

        return f"sample_{sample_idx}"

    @staticmethod
    def _timestamp_from_meta(meta: Dict[str, Any]) -> int:
        ts = meta.get("timestamp", None)
        if ts is None:
            return -1
        try:
            return int(ts)
        except (TypeError, ValueError):
            return -1

    def _prune(self) -> None:
        while len(self._memory) > self.max_entries:
            key, _ = self._memory.popitem(last=False)
            self._last_timestamp.pop(key, None)
            self._pose_memory.pop(key, None)

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    def _can_blend(self, key: str, timestamp: int) -> bool:
        if key not in self._memory:
            return False
        if timestamp < 0:
            return False

        prev = self._last_timestamp.get(key, -1)
        if prev < 0:
            return False

        delta = timestamp - prev
        return 0 < delta <= self.max_timestamp_gap

    @staticmethod
    def _to_pose_tensor(pose_like, device: torch.device) -> torch.Tensor:
        pose = torch.as_tensor(pose_like, device=device, dtype=torch.float32)
        if pose.dim() == 3:
            pose = pose[0]
        if pose.shape != (4, 4):
            return None
        return pose

    def _extract_current_pose(self, ego2global, sample_idx: int, device: torch.device) -> torch.Tensor:
        if ego2global is None:
            return None
        try:
            pose_like = ego2global[sample_idx]
        except Exception:
            return None
        return self._to_pose_tensor(pose_like, device=device)

    @staticmethod
    def _safe_inverse(mat: torch.Tensor) -> torch.Tensor:
        try:
            return torch.linalg.inv(mat)
        except RuntimeError:
            return torch.linalg.pinv(mat)

    def _theta_from_relative_pose(self, rel_pose: torch.Tensor, device: torch.device) -> torch.Tensor:
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        sx = max((x_max - x_min) * 0.5, 1e-6)
        sy = max((y_max - y_min) * 0.5, 1e-6)
        bx = (x_max + x_min) * 0.5
        by = (y_max + y_min) * 0.5

        r11 = rel_pose[0, 0]
        r12 = rel_pose[0, 1]
        tx = rel_pose[0, 3]
        r21 = rel_pose[1, 0]
        r22 = rel_pose[1, 1]
        ty = rel_pose[1, 3]

        theta = torch.zeros((1, 2, 3), device=device, dtype=torch.float32)
        theta[0, 0, 0] = r11
        theta[0, 0, 1] = r12 * (sy / sx)
        theta[0, 0, 2] = (r11 * bx + r12 * by + tx - bx) / sx
        theta[0, 1, 0] = r21 * (sx / sy)
        theta[0, 1, 1] = r22
        theta[0, 1, 2] = (r21 * bx + r22 * by + ty - by) / sy
        return theta

    def _warp_history_to_current(
        self,
        hist: torch.Tensor,
        prev_pose: torch.Tensor,
        curr_pose: torch.Tensor,
    ) -> torch.Tensor:
        # rel pose maps points from current ego frame to previous ego frame.
        rel_pose = self._safe_inverse(prev_pose) @ curr_pose
        theta = self._theta_from_relative_pose(rel_pose, device=hist.device)

        hist_batch = hist.unsqueeze(0).float()
        grid = F.affine_grid(theta=theta, size=hist_batch.shape, align_corners=False)
        warped = F.grid_sample(
            hist_batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped.squeeze(0).to(dtype=hist.dtype)

    def forward(
        self,
        feats: torch.Tensor,
        metas: Iterable[Dict[str, Any]] = None,
        ego2global: torch.Tensor = None,
    ) -> torch.Tensor:
        if metas is None:
            return feats

        metas = list(metas)
        if len(metas) != feats.shape[0]:
            return feats

        feats = self._sanitize(feats)
        fused_list = []
        for i, meta in enumerate(metas):
            key = self._key_from_meta(meta, i)
            timestamp = self._timestamp_from_meta(meta)
            can_blend = self._can_blend(key, timestamp)
            hist = self._memory.get(key) if can_blend else None
            curr_pose = self._extract_current_pose(ego2global, i, device=feats.device)
            prev_pose = self._pose_memory.get(key) if can_blend else None

            if hist is not None:
                hist = self._sanitize(hist.to(device=feats.device, dtype=feats.dtype))
                if self.enable_pose_warp:
                    if curr_pose is not None and prev_pose is not None:
                        prev_pose = prev_pose.to(device=feats.device, dtype=torch.float32)
                        hist = self._warp_history_to_current(hist, prev_pose=prev_pose, curr_pose=curr_pose)
                    else:
                        hist = None
                        if not self._warned_missing_pose:
                            warnings.warn(
                                "FeatureMemoryBank pose-warp is enabled but ego2global pose is unavailable; "
                                "temporal blending is skipped for safety.",
                                stacklevel=2,
                            )
                            self._warned_missing_pose = True

            if hist is not None:
                fused_i = (1.0 - self.blend) * feats[i] + self.blend * hist
                fused_i = self._sanitize(fused_i)
            else:
                fused_i = feats[i]

            fused_list.append(fused_i)

            with torch.no_grad():
                if hist is not None:
                    updated = self.momentum * hist + (1.0 - self.momentum) * feats[i].detach()
                else:
                    updated = feats[i].detach()

                updated = self._sanitize(updated)
                if torch.isfinite(updated).all():
                    self._memory[key] = updated.cpu()
                    self._last_timestamp[key] = timestamp
                    if curr_pose is not None and torch.isfinite(curr_pose).all():
                        self._pose_memory[key] = curr_pose.detach().cpu()
                    elif key in self._pose_memory:
                        self._pose_memory.pop(key, None)
                elif key in self._memory:
                    self._memory.pop(key, None)
                    self._last_timestamp.pop(key, None)
                    self._pose_memory.pop(key, None)

            if (
                not self._warned_non_temporal_key
                and timestamp < 0
                and key.startswith("sample_")
            ):
                warnings.warn(
                    "FeatureMemoryBank metadata has no stable temporal key/timestamp; "
                    "temporal blending is effectively disabled for safety.",
                    stacklevel=2,
                )
                self._warned_non_temporal_key = True

        self._prune()
        return torch.stack(fused_list, dim=0)
