import inspect
import warnings
from typing import Iterable, Optional

import torch

from mmcv.runner import auto_fp16
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion


@FUSIONMODELS.register_module()
class HybridBEVFusionPlus(BEVFusion):
    """BEVFusion variant with robust detection-prior extraction for occupancy."""

    def __init__(
        self,
        use_detection_guidance: bool = True,
        detach_detection_guidance: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.use_detection_guidance = use_detection_guidance
        self.detach_detection_guidance = detach_detection_guidance
        self._warned_missing_occ_aug_matrix = False

    @staticmethod
    def _iter_prediction_dicts(pred) -> Iterable[dict]:
        if pred is None:
            return
        if isinstance(pred, dict):
            yield pred
            for value in pred.values():
                if isinstance(value, (dict, list, tuple)):
                    yield from HybridBEVFusionPlus._iter_prediction_dicts(value)
            return
        if isinstance(pred, (list, tuple)):
            for item in pred:
                yield from HybridBEVFusionPlus._iter_prediction_dicts(item)

    @staticmethod
    def _extract_detection_guidance(pred_dict) -> Optional[torch.Tensor]:
        if pred_dict is None:
            return None

        hm_list = []
        for item in HybridBEVFusionPlus._iter_prediction_dicts(pred_dict):
            heatmap = item.get("heatmap", None)
            if heatmap is None or not torch.is_tensor(heatmap):
                continue
            if heatmap.dim() != 4:
                continue
            # Keep detector outputs in logit space here; projector applies sigmoid once.
            heatmap = torch.nan_to_num(heatmap, nan=0.0, posinf=20.0, neginf=-20.0)
            hm_list.append(heatmap.amax(dim=1, keepdim=True))

        if not hm_list:
            return None
        return torch.cat(hm_list, dim=1).amax(dim=1, keepdim=True).clamp_(-20.0, 20.0)

    @staticmethod
    def _occ_head_extra_kwargs(occ_head, det_guidance, metas, context_kwargs):
        try:
            params = inspect.signature(occ_head.forward).parameters
        except (TypeError, ValueError):
            return {}

        supports_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        extra_kwargs = {}
        if "det_guidance_logits" in params or supports_var_kwargs:
            extra_kwargs["det_guidance_logits"] = det_guidance
        if "metas" in params or supports_var_kwargs:
            extra_kwargs["metas"] = metas
        for key in ("camera_ego2global", "ego2global", "mask_camera"):
            if key in context_kwargs and (key in params or supports_var_kwargs):
                extra_kwargs[key] = context_kwargs[key]
        return extra_kwargs

    @staticmethod
    def _default_occ_aug_matrix(occ_aug_matrix, lidar_aug_matrix):
        if occ_aug_matrix is not None:
            return occ_aug_matrix
        if torch.is_tensor(lidar_aug_matrix):
            batch_size = int(lidar_aug_matrix.shape[0])
            eye = torch.eye(4, device=lidar_aug_matrix.device, dtype=lidar_aug_matrix.dtype)
            return eye.unsqueeze(0).repeat(batch_size, 1, 1)
        return None

    def _forward_occ_head(
        self,
        occ_head,
        x,
        lidar_aug_matrix,
        lidar2ego,
        occ_aug_matrix,
        det_guidance,
        metas,
        context_kwargs,
    ):
        extra_kwargs = self._occ_head_extra_kwargs(occ_head, det_guidance, metas, context_kwargs)
        return occ_head(x, lidar_aug_matrix, lidar2ego, occ_aug_matrix, **extra_kwargs)

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
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
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        return self.forward_single(
            img,
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
            gt_masks_bev,
            gt_bboxes_3d,
            gt_labels_3d,
            **kwargs,
        )

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
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
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
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
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        if self.decoder is not None:
            x = self.decoder["backbone"](x)
            x = self.decoder["neck"](x)

        batch_size = x.shape[0]

        object_pred = None
        if "object" in self.heads:
            object_pred = self.heads["object"](x, metas)

        det_guidance = None
        if self.use_detection_guidance:
            det_guidance = self._extract_detection_guidance(object_pred)
            if det_guidance is not None and self.detach_detection_guidance:
                det_guidance = det_guidance.detach()

        if self.training:
            outputs = {}

            if "object" in self.heads:
                obj_losses = self.heads["object"].loss(gt_bboxes_3d, gt_labels_3d, object_pred)
                for name, val in obj_losses.items():
                    key = f"loss/object/{name}" if val.requires_grad else f"stats/object/{name}"
                    outputs[key] = val * self.loss_scale["object"] if val.requires_grad else val

            if "map" in self.heads:
                map_losses = self.heads["map"](x, gt_masks_bev)
                for name, val in map_losses.items():
                    key = f"loss/map/{name}" if val.requires_grad else f"stats/map/{name}"
                    outputs[key] = val * self.loss_scale["map"] if val.requires_grad else val

            if "occ" in self.heads:
                occ_head = self.heads["occ"]
                occ_aug_matrix = self._default_occ_aug_matrix(kwargs.get("occ_aug_matrix", None), lidar_aug_matrix)
                if occ_aug_matrix is None and not self._warned_missing_occ_aug_matrix:
                    warnings.warn(
                        "occ_aug_matrix is missing and could not be inferred; passing None to occ head.",
                        stacklevel=2,
                    )
                    self._warned_missing_occ_aug_matrix = True
                occ_pred = self._forward_occ_head(
                    occ_head,
                    x,
                    lidar_aug_matrix,
                    lidar2ego,
                    occ_aug_matrix,
                    det_guidance,
                    metas,
                    kwargs,
                )

                occ_losses = occ_head.loss(occ_pred, kwargs["voxel_semantics"], kwargs["mask_camera"])
                for name, val in occ_losses.items():
                    key = f"loss/occ/{name}" if val.requires_grad else f"stats/occ/{name}"
                    outputs[key] = val * self.loss_scale["occ"] if val.requires_grad else val

            if det_guidance is not None:
                outputs["stats/occ/det_guidance_mean"] = det_guidance.detach().mean()
                outputs["stats/occ/det_guidance_max"] = det_guidance.detach().max()
            else:
                outputs["stats/occ/det_guidance_mean"] = x.detach().new_tensor(0.0)
                outputs["stats/occ/det_guidance_max"] = x.detach().new_tensor(0.0)

            return outputs

        outputs = [{} for _ in range(batch_size)]

        if "object" in self.heads:
            bboxes = self.heads["object"].get_bboxes(object_pred, metas)
            for k, (boxes, scores, labels) in enumerate(bboxes):
                outputs[k].update(
                    {
                        "boxes_3d": boxes.to("cpu"),
                        "scores_3d": scores.cpu(),
                        "labels_3d": labels.cpu(),
                    }
                )

        if "map" in self.heads:
            logits = self.heads["map"](x)
            for k in range(batch_size):
                outputs[k].update(
                    {
                        "masks_bev": logits[k].cpu(),
                        "gt_masks_bev": gt_masks_bev[k].cpu() if gt_masks_bev is not None else None,
                    }
                )

        if "occ" in self.heads:
            occ_head = self.heads["occ"]
            occ_aug_matrix = self._default_occ_aug_matrix(kwargs.get("occ_aug_matrix", None), lidar_aug_matrix)
            if occ_aug_matrix is None and not self._warned_missing_occ_aug_matrix:
                warnings.warn(
                    "occ_aug_matrix is missing and could not be inferred; passing None to occ head.",
                    stacklevel=2,
                )
                self._warned_missing_occ_aug_matrix = True
            occ_pred = self._forward_occ_head(
                occ_head,
                x,
                lidar_aug_matrix,
                lidar2ego,
                occ_aug_matrix,
                det_guidance,
                metas,
                kwargs,
            )
            occ_pred = occ_head.get_occ(occ_pred)
            for k in range(batch_size):
                outputs[k].update({"occ_pred": occ_pred[k]})

        return outputs
