import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.utils import save_image


class ImageProcessor:
    @staticmethod
    def normalize_tensor(tensor: Tensor) -> Tensor:
        """
        Normalize tensor values -> https://stats.stackexchange.com/a/178629
        min x = 0, max x = 255
        [ 0, 1] -> x = (x - min x) / (max x - min x)
        [-1, 1] -> x = 2 * (x - min x) / (max x - min x) - 1
        """
        return 2 * tensor / 255 - 1

    @staticmethod
    def denormalize_tensor(tensor: Tensor) -> Tensor:
        """
        Denormalize tensor values
        Reverse function of normalize_tensor()
        """
        return (tensor + 1) * 255 / 2

    @classmethod
    def img_to_tensor(cls, path: str) -> Tensor:
        """Given image path, return normalized tensor (C, H, W)"""
        transform = transforms.Compose([transforms.PILToTensor()])
        with Image.open(path) as image:
            return cls.normalize_tensor(transform(image).type(torch.float32))

    @staticmethod
    def save_tensor_as_image(tensor: Tensor, image_path: str) -> None:
        """Save given tensor as image"""
        save_image(tensor, image_path)

    @classmethod
    def padding_center(cls, tensor: Tensor, max_h: int, max_w: int) -> Tensor:
        """Add padding to the 2d square tensor and place it on the center"""
        norm_zero = float(cls.normalize_tensor(0))
        quotient_h, remainder_h = divmod(max_h - tensor.size(1), 2)
        quotient_w, remainder_w = divmod(max_w - tensor.size(2), 2)
        return F.pad(
            tensor,
            [quotient_w + remainder_w, quotient_w, quotient_h + remainder_h, quotient_h],
            value=norm_zero,
        )
