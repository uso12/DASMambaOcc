from typing import Any, Dict

import numpy as np
import torchvision
from PIL import Image
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ImageNormalizeSafe:
    """Image normalization with optional BGR->RGB conversion for ndarray inputs."""

    def __init__(self, mean, std, to_rgb: bool = True):
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def _prepare_image(self, img: Any):
        if isinstance(img, Image.Image):
            if self.to_rgb and img.mode != "RGB":
                return img.convert("RGB")
            return img

        arr = np.asarray(img)
        if self.to_rgb and arr.ndim == 3 and arr.shape[2] >= 3:
            # NumPy images are commonly BGR when loaded via OpenCV.
            arr = arr[..., :3][:, :, ::-1].copy()
        return arr

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["img"] = [self.compose(self._prepare_image(img)) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return data
