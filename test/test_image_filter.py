from typing import Type
from os import path
import unittest
import pyimagetest
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms.functional import to_tensor
import torchimagefilter as tif


class TorchvisionImageBackend(pyimagetest.ImageBackend):
    @property
    def native_image_type(self) -> Type[torch.FloatTensor]:
        return torch.FloatTensor

    def import_image(self, file: str) -> np.ndarray:
        return to_tensor(Image.open(file))

    def export_image(self, image: torch.FloatTensor) -> np.ndarray:
        return image.detach().cpu().permute((1, 2, 0)).numpy()


class Tester(pyimagetest.ImageTestcase, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_image_backend("torchvision", TorchvisionImageBackend())

    @property
    def default_test_image_file(self) -> str:
        # The test image was downloaded from
        # http://www.r0k.us/graphics/kodak/kodim15.html
        # and is cleared for unrestricted usage
        return path.join(path.dirname(__file__), "test_image.png")

    def test_box_filter(self):
        for radius in range(6):
            pil_image = self.load_image(self.backends["PIL"])
            pil_image = pil_image.filter(ImageFilter.BoxBlur(radius))

            torchvision_image = self.load_image(self.backends["torchvision"])
            torchvision_image = tif.box_blur(torchvision_image, radius)

            self.assertImagesAlmostEqual(pil_image, torchvision_image)

    def test_gauss_filter(self):
        pass
        # FIXME: implement with the corresponding PIL parameters
        # for radius in range(6):
        #     pil_image = self.load_image(self.backends["PIL"])
        #     pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius))
        #
        #     torchvision_image = self.load_image(self.backends["torchvision"])
        #     torchvision_image = tif.gaussian_blur(torchvision_image, radius)
        #
        #     self.assertImagesAlmostEqual(pil_image, torchvision_image)


if __name__ == "__main__":
    unittest.main()
