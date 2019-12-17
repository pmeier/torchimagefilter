from typing import Optional, Tuple, List
from copy import copy
import numpy as np
import torch
from torch import nn
from torch.nn.functional import conv2d, pad
from .kernel import *


class ImageFilter(nn.Module):
    def __init__(
        self,
        kernel: torch.Tensor,
        output_shape: str = "same",
        padding_mode: str = "constant",
        constant_padding_value: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("kernel", kernel)
        if output_shape not in ("valid", "same", "full"):
            raise ValueError(
                "'output_shape' should be one of " "{'valid', 'same', 'full'}"
            )
        self.output_shape = output_shape
        self.padding_size = self._calculate_padding_size()
        if output_shape not in ("valid", "same", "full"):
            raise ValueError(
                "'padding_mode' should be one of "
                "{'constant', 'reflect', 'replicate', 'circular'}"
            )
        self.padding_mode = padding_mode
        self.constant_padding_value = constant_padding_value

    @property
    def kernel_size(self) -> Tuple[int, int]:
        return self.kernel.size()

    def _calculate_padding_size(self) -> List[int]:
        kernel_height, kernel_width = self.kernel_size
        if self.output_shape == "valid":
            pad_top = pad_bot = pad_left = pad_right = 0
        elif self.output_shape == "same":
            pad_vert, pad_horz = kernel_height - 1, kernel_width - 1
            pad_top, pad_left = pad_vert // 2, pad_horz // 2
            pad_bot, pad_right = pad_vert - pad_top, pad_horz - pad_left
        else:  # self.output_shape == "full":
            pad_top = pad_bot = kernel_height - 1
            pad_left = pad_right = kernel_width - 1
        return [pad_left, pad_right, pad_top, pad_bot]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        is_batched = image.dim() == 4
        if not is_batched:
            image = image.unsqueeze(0)

        image = self._pad_image(image)
        image = self._filter_image(image)

        if not is_batched:
            image = image.squeeze(0)

        return image

    def _filter_image(self, image: torch.Tensor) -> torch.Tensor:
        num_channels = image.size()[1]
        weight = self.kernel.view(1, 1, *self.kernel.size())
        weight = weight.repeat(num_channels, 1, 1, 1)
        return conv2d(
            image,
            weight,
            bias=None,
            groups=num_channels,
            stride=1,
            padding=0,
            dilation=1,
        )

    def _pad_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.output_shape == "valid":
            return image
        return pad(
            image,
            self.padding_size,
            mode=self.padding_mode,
            value=self.constant_padding_value,
        )

    def extra_repr(self) -> str:
        extra_repr = self.image_filter_extra_repr()
        if extra_repr:
            extra_repr += ", "
        extra_repr += "output_shape={output_shape}"
        if self.output_shape != "valid":
            extra_repr += ", padding_mode={padding_mode}"
            if self.padding_mode == "constant":
                extra_repr += ", constant_padding_value={constant_padding_value}"
        return extra_repr.format(**self.__dict__)

    def image_filter_extra_repr(self) -> str:
        return ""


class BoxFilter(ImageFilter):
    def __init__(self, radius: int, normalize: bool = True, **kwargs):
        self.radius = radius

        kernel = box_kernel(radius, normalize=normalize)
        super().__init__(kernel, **kwargs)

    def image_filter_extra_repr(self) -> str:
        return "radius={radius}".format(**self.__dict__)


class GaussFilter(ImageFilter):
    def __init__(
        self,
        std: Optional[float] = None,
        radius: Optional[int] = None,
        normalize: bool = True,
        **kwargs
    ):
        if std is None and radius is None:
            msg = (
                "One argument of 'std' and 'radius' has to have a value different "
                "from None. The respective other is subsequently calculated."
            )
            raise ValueError(msg)

        if std is None:
            std = radius / 3.0
        self.std = std

        if radius is None:
            radius = int(np.ceil(3.0 * std))
        self.radius = radius

        kernel = gauss_kernel(std, radius, normalize=normalize)
        super().__init__(kernel, **kwargs)

    def image_filter_extra_repr(self) -> str:
        dct = copy(self.__dict__)
        dct["std"] = "{:.2g}".format(self.std)
        return "std={std}, radius={radius}".format(**dct)
