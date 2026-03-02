from typing import Optional

import torch
import torch.nn.functional as F


def hard_negative_suppression_loss(
    occ_pred: torch.Tensor,
    det_guidance_xy: Optional[torch.Tensor],
    mask_camera: torch.Tensor,
    empty_class_idx: int,
    guidance_threshold: float,
    loss_weight: float,
) -> torch.Tensor:
    """Penalize non-empty occupancy outside detector-supported regions."""
    if loss_weight <= 0 or det_guidance_xy is None:
        return occ_pred.new_tensor(0.0)

    occ_pred = torch.nan_to_num(occ_pred, nan=0.0, posinf=1e4, neginf=-1e4)
    det_guidance_xy = torch.nan_to_num(det_guidance_xy, nan=0.0, posinf=1.0, neginf=0.0)
    probs = occ_pred.softmax(dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    nonempty_prob = 1.0 - probs[..., empty_class_idx]
    nonempty_prob = torch.nan_to_num(nonempty_prob, nan=0.0, posinf=1.0, neginf=0.0)

    if det_guidance_xy.dim() == 5:
        det_mask_xy = det_guidance_xy[..., 0, 0] >= guidance_threshold
    elif det_guidance_xy.dim() == 4:
        det_mask_xy = det_guidance_xy[..., 0] >= guidance_threshold
    else:
        raise ValueError(f"det_guidance_xy must be 4D/5D, got shape {tuple(det_guidance_xy.shape)}")
    det_mask = det_mask_xy.unsqueeze(-1).expand_as(nonempty_prob)

    valid_mask = mask_camera.to(torch.bool)
    neg_mask = valid_mask & (~det_mask)

    if int(neg_mask.sum()) == 0:
        return occ_pred.new_tensor(0.0)

    pred_neg = nonempty_prob[neg_mask]
    pred_neg = torch.nan_to_num(pred_neg, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if pred_neg.numel() == 0:
        return occ_pred.new_tensor(0.0)
    target = torch.zeros_like(pred_neg)
    return F.binary_cross_entropy(pred_neg, target) * loss_weight
