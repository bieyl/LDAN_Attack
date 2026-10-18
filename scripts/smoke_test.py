#!/usr/bin/env python
"""Run a dependency-light shape test for the trainable attack modules."""

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ldan_attack.attention import GradCAMExtractor
from ldan_attack.config import AttackConfig
from ldan_attack.diffusion import DiffusionLatentAdapter
from ldan_attack.pipeline import LatentAttentionAttack
from ldan_attack.toy import ToyLatentAdapter


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def main():
    config = AttackConfig(
        image_size=32,
        latent_dim=4,
        batch_size=2,
        neighborhood_samples=2,
        target_layer="features.0",
        device="cpu",
    )
    model = TinyClassifier()
    adapter = ToyLatentAdapter()
    attack = LatentAttentionAttack(model, adapter, config)
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    attack.initialize(images, labels)
    values = attack.training_step(images, labels)
    values["loss"].backward()
    generated = attack.generate(images, labels, samples=2)
    assert generated.shape == (2, 2, 3, 32, 32)
    assert torch.isfinite(values["loss"])
    print("smoke test passed", tuple(generated.shape))


if __name__ == "__main__":
    main()

