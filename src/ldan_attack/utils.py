"""Data, reproducibility, and image-space utilities."""

import os
import random
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_imagefolder(path: str, image_size: int, batch_size: int, shuffle: bool,
                     num_workers: int = 4) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def to_diffusion_range(images: Tensor) -> Tensor:
    return images.mul(2.0).sub(1.0)


def from_diffusion_range(images: Tensor) -> Tensor:
    return images.add(1.0).div(2.0)


def clip_image(images: Tensor) -> Tensor:
    return images.clamp(0.0, 1.0)


def save_tensor_image(images: Tensor, path: str):
    from torchvision.utils import save_image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(images.detach().cpu().clamp(0.0, 1.0), path, nrow=1)

