"""Small deterministic adapter for shape and pipeline smoke tests only."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .diffusion import DiffusionLatentAdapter


class ToyLatentAdapter(DiffusionLatentAdapter):
    """A VAE-like adapter; it is not used for reported experiments."""

    latent_channels = 4

    def encode(self, images: Tensor, tau_star: int) -> Tensor:
        pooled = F.avg_pool2d(images, kernel_size=8, stride=8)
        return torch.cat([pooled, pooled[:, :1]], dim=1)

    def decode(self, latents: Tensor, tau_star: int, output_size) -> Tensor:
        rgb = latents[:, :3]
        return F.interpolate(rgb, size=output_size, mode="bilinear", align_corners=False)

