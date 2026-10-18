"""Learnable attention neighborhoods and attention-to-latent masks."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AttentionNeighborhood(nn.Module):
    """Sample projected attention supports and map them to latent masks."""

    def __init__(self, attention_channels: int, latent_channels: int, radius: float = 0.10,
                 projection_eps: float = 1.0e-8):
        super().__init__()
        self.radius = float(radius)
        self.projection_eps = float(projection_eps)
        self.scale = nn.Sequential(
            nn.Conv2d(attention_channels, attention_channels, kernel_size=1),
            nn.Softplus(),
        )
        self.mask_mapper = nn.Conv2d(attention_channels, latent_channels, kernel_size=1)

    def project(self, increment: Tensor) -> Tensor:
        norm = increment.flatten(1).norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(self.radius / (norm + self.projection_eps), max=1.0)
        return increment * factor.view(-1, 1, 1, 1)

    def sample(self, attention: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(attention)
        if noise.shape != attention.shape:
            raise ValueError("neighborhood noise must have the same shape as attention")
        increment = self.scale(attention) * noise
        return attention + self.project(increment)

    def to_mask(self, support: Tensor, latent_size) -> Tensor:
        resized = F.interpolate(support, size=latent_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.mask_mapper(resized))

