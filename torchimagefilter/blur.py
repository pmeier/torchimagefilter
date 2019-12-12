from typing import Optional
import torch
from .filter import BoxFilter, GaussFilter

__all__ = ["box_blur", "gaussian_blur"]


def box_blur(image: torch.Tensor, radius: int, normalize: bool = True, **kwargs):
    image_filter = BoxFilter(radius, normalize=normalize, **kwargs).to(image.device)
    return image_filter(image)


def gaussian_blur(
    image: torch.Tensor,
    std: Optional[float] = None,
    radius: Optional[int] = None,
    normalize: bool = True,
    **kwargs
):
    image_filter = GaussFilter(
        std=std, radius=radius, normalize=normalize, **kwargs
    ).to(image.device)
    return image_filter(image)
