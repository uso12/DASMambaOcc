from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def hard_negative_suppression_loss(
    occ_pred: torch.Tensor,
    det_guidance_xy: Optional[torch.Tensor],
    mask_camera: Optional[torch.Tensor],
    voxel_semantics: Optional[torch.Tensor],
    empty_class_idx: int,
    guidance_threshold: float,
    loss_weight: float,
    foreground_class_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Penalize non-empty occupancy outside detector-supported regions."""
    if loss_weight <= 0 or det_guidance_xy is None:
        return occ_pred.new_tensor(0.0)

    occ_pred = torch.nan_to_num(occ_pred, nan=0.0, posinf=1e4, neginf=-1e4)
    det_guidance_xy = torch.nan_to_num(det_guidance_xy, nan=0.0, posinf=1.0, neginf=0.0)
    probs = occ_pred.softmax(dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

    num_classes = int(probs.shape[-1])
    if foreground_class_indices is not None:
        fg_idx = torch.as_tensor(foreground_class_indices, device=probs.device, dtype=torch.long).view(-1)
        valid = (fg_idx >= 0) & (fg_idx < num_classes) & (fg_idx != int(empty_class_idx))
        fg_idx = fg_idx[valid].unique(sorted=True)
        if fg_idx.numel() == 0:
            return occ_pred.new_tensor(0.0)
        nonempty_prob = probs.index_select(dim=-1, index=fg_idx).sum(dim=-1)
    else:
        nonempty_prob = 1.0 - probs[..., empty_class_idx]
    nonempty_prob = torch.nan_to_num(nonempty_prob, nan=0.0, posinf=1.0, neginf=0.0)

    if det_guidance_xy.dim() == 5:
        det_mask_xy = det_guidance_xy[..., 0, 0] >= guidance_threshold
    elif det_guidance_xy.dim() == 4:
        det_mask_xy = det_guidance_xy[..., 0] >= guidance_threshold
    else:
        raise ValueError(f"det_guidance_xy must be 4D/5D, got shape {tuple(det_guidance_xy.shape)}")
    det_mask = det_mask_xy.unsqueeze(-1).expand_as(nonempty_prob)

    if mask_camera is None:
        valid_mask = torch.ones_like(nonempty_prob, dtype=torch.bool)
    else:
        valid_mask = mask_camera.to(torch.bool)

    if voxel_semantics is None:
        empty_gt_mask = torch.ones_like(nonempty_prob, dtype=torch.bool)
    else:
        gt = voxel_semantics.to(device=occ_pred.device)
        if gt.shape != nonempty_prob.shape:
            raise ValueError(
                f"voxel_semantics shape {tuple(gt.shape)} must match occupancy grid "
                f"shape {tuple(nonempty_prob.shape)}"
            )
        empty_gt_mask = gt.to(torch.long) == int(empty_class_idx)

    # Suppress only where detector is silent and GT confirms empty voxels.
    neg_mask = valid_mask & (~det_mask) & empty_gt_mask

    if int(neg_mask.sum()) == 0:
        return occ_pred.new_tensor(0.0)

    pred_neg = nonempty_prob[neg_mask]
    pred_neg = torch.nan_to_num(pred_neg, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if pred_neg.numel() == 0:
        return occ_pred.new_tensor(0.0)
    target = torch.zeros_like(pred_neg)
    return F.binary_cross_entropy(pred_neg, target) * loss_weight
