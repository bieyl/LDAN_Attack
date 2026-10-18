"""Losses used by the latent attention attack."""

import torch
from torch import Tensor
import torch.nn.functional as F


def attack_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Untargeted surrogate loss maximized by the attack."""
    return F.cross_entropy(logits, labels)


def attention_alignment_loss(adversarial_attention: Tensor, support: Tensor) -> Tensor:
    """Positive Frobenius distance to the sampled attention support."""
    if adversarial_attention.shape != support.shape:
        raise ValueError("attention tensors must have identical shapes")
    return (adversarial_attention - support).flatten(1).norm(p=2, dim=1).mean()


def attention_diversity_loss(attention_a: Tensor, attention_b: Tensor,
                             noise_a: Tensor, noise_b: Tensor,
                             eps: float = 1.0e-8) -> Tensor:
    """Cosine diversity normalized by latent-code separation."""
    a = attention_a.flatten(1)
    b = attention_b.flatten(1)
    cosine = F.cosine_similarity(a, b, dim=1, eps=eps)
    distance = (noise_a - noise_b).flatten(1).norm(p=2, dim=1)
    return ((1.0 - cosine) / (distance + eps)).mean()


def clip_linf(images: Tensor, clean: Tensor, epsilon: float) -> Tensor:
    """Project normalized images into the pixel-space L-infinity box."""
    return torch.maximum(torch.minimum(images, clean + epsilon), clean - epsilon)

