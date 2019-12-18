import numpy as np
import torch

__all__ = ["radius_to_size", "gauss_kernel", "box_kernel"]


def radius_to_size(radius: int) -> int:
    return 2 * radius + 1


def _normalize_kernel(kernel: torch.Tensor) -> torch.Tensor:
    return kernel / torch.sum(kernel)


def box_kernel(radius: int, normalize: bool = True) -> torch.Tensor:
    size = radius_to_size(radius)
    kernel = torch.ones((size, size))
    if normalize:
        kernel = _normalize_kernel(kernel)
    return kernel


def gauss_kernel(std: float, radius: int, normalize: bool = True) -> torch.Tensor:
    var = std ** 2.0
    factor = np.sqrt(2.0 * np.pi * var)
    exponent = torch.arange(-radius, radius + 1, dtype=torch.float)
    exponent = -(exponent ** 2.0) / (2.0 * var)
    kernel = (factor * torch.exp(exponent)).unsqueeze(1)
    kernel = torch.mm(kernel, kernel.t())
    if normalize:
        kernel = _normalize_kernel(kernel)
    return kernel
