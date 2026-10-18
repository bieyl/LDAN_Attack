"""Latent diffusion-based attention-guided neighborhood attack."""

from .config import AttackConfig

__all__ = ["AttackConfig", "LatentAttentionAttack"]


def __getattr__(name):
    if name == "LatentAttentionAttack":
        from .pipeline import LatentAttentionAttack

        return LatentAttentionAttack
    raise AttributeError(name)
