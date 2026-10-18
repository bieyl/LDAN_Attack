"""Latent residual generator used by the attack."""

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.silu(x + self.block(x))


class LatentResidualGenerator(nn.Module):
    """Generate a latent residual conditioned on a sampled latent code."""

    def __init__(self, latent_channels: int = 4, noise_dim: int = 16, base_channels: int = 128):
        super().__init__()
        self.noise_dim = noise_dim
        self.input = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8 if base_channels % 8 == 0 else 1, base_channels),
            nn.SiLU(inplace=True),
        )
        self.noise_projection = nn.Sequential(
            nn.Linear(noise_dim, base_channels),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels, base_channels),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
            ResidualBlock(base_channels),
        )
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, latent_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, latent: Tensor, noise: Tensor) -> Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError("noise must have shape B x {}".format(self.noise_dim))
        hidden = self.input(latent)
        condition = self.noise_projection(noise).unsqueeze(-1).unsqueeze(-1)
        hidden = hidden + condition
        return self.output(self.blocks(hidden))

