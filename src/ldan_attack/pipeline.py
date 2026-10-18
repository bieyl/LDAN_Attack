"""End-to-end latent diffusion attention-neighborhood attack."""

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .attention import GradCAMExtractor, freeze, unwrap_logits
from .config import AttackConfig
from .generator import LatentResidualGenerator
from .losses import (
    attention_alignment_loss,
    attention_diversity_loss,
    attack_loss,
    clip_linf,
)
from .models import preprocess
from .neighborhood import AttentionNeighborhood
from .utils import clip_image, from_diffusion_range, to_diffusion_range


class LatentAttentionAttack(nn.Module):
    """Trainable attack module for one frozen surrogate and diffusion model."""

    def __init__(self, surrogate: nn.Module, diffusion, config: AttackConfig):
        super().__init__()
        self.config = config
        self.surrogate = surrogate
        self.diffusion = diffusion
        freeze(self.surrogate)
        self.attention = GradCAMExtractor(
            self.surrogate, config.target_layer, eps=config.normalization_eps
        )
        self.generator: Optional[LatentResidualGenerator] = None
        self.neighborhood: Optional[AttentionNeighborhood] = None

    @property
    def initialized(self) -> bool:
        return self.generator is not None and self.neighborhood is not None

    def train(self, mode: bool = True):
        """Train only the attack modules; keep source and diffusion models in eval mode."""
        super().train(mode)
        self.surrogate.eval()
        self.diffusion.eval()
        return self

    def initialize(self, images: Tensor, labels: Tensor):
        """Create shape-dependent modules before constructing the optimizer."""
        if self.initialized:
            return
        with torch.enable_grad():
            clean_attention = self.attention(
                preprocess(images), labels, create_graph=False
            ).detach()
        with torch.no_grad():
            latent = self.diffusion.encode(to_diffusion_range(images), self.config.tau_star)
        self.neighborhood = AttentionNeighborhood(
            attention_channels=clean_attention.shape[1],
            latent_channels=latent.shape[1],
            radius=self.config.radius,
            projection_eps=self.config.projection_eps,
        ).to(images.device)
        self.generator = LatentResidualGenerator(
            latent_channels=latent.shape[1],
            noise_dim=self.config.latent_dim,
        ).to(images.device)

    def _decode_clipped(self, latent: Tensor, clean: Tensor) -> Tensor:
        decoded = self.diffusion.decode(
            latent, self.config.tau_star, output_size=clean.shape[-2:]
        )
        decoded = from_diffusion_range(decoded)
        return clip_linf(clip_image(decoded), clean, self.config.epsilon)

    def _forward_pair(self, latent: Tensor, clean: Tensor, support: Tensor,
                      eta_a: Tensor, eta_b: Tensor):
        mask = self.neighborhood.to_mask(support, latent.shape[-2:])
        residual_a = self.generator(latent, eta_a) * mask
        residual_b = self.generator(latent, eta_b) * mask
        adversarial_a = self._decode_clipped(latent + residual_a, clean)
        adversarial_b = self._decode_clipped(latent + residual_b, clean)
        input_a = preprocess(adversarial_a)
        input_b = preprocess(adversarial_b)
        logits_a = unwrap_logits(self.surrogate(input_a))
        logits_b = unwrap_logits(self.surrogate(input_b))
        attention_a = self.attention(input_a, self._labels, create_graph=True)
        attention_b = self.attention(input_b, self._labels, create_graph=True)
        return adversarial_a, adversarial_b, logits_a, logits_b, attention_a, attention_b

    def training_step(self, images: Tensor, labels: Tensor) -> Dict[str, Tensor]:
        """Compute the paper objective for one mini-batch."""
        if not self.initialized:
            self.initialize(images, labels)
        self._labels = labels
        with torch.no_grad():
            latent = self.diffusion.encode(to_diffusion_range(images), self.config.tau_star)
        with torch.enable_grad():
            clean_attention = self.attention(
                preprocess(images), labels, create_graph=False
            ).detach()
        losses = {"adv": [], "attn": [], "div": []}
        for _ in range(self.config.neighborhood_samples):
            support = self.neighborhood.sample(clean_attention)
            eta_a = torch.randn(
                images.shape[0], self.config.latent_dim, device=images.device
            )
            eta_b = torch.randn_like(eta_a)
            _, _, logits_a, logits_b, attention_a, attention_b = self._forward_pair(
                latent, images, support, eta_a, eta_b
            )
            losses["adv"].append(
                0.5 * (attack_loss(logits_a, labels) + attack_loss(logits_b, labels))
            )
            losses["attn"].append(
                0.5 * (
                    attention_alignment_loss(attention_a, support)
                    + attention_alignment_loss(attention_b, support)
                )
            )
            losses["div"].append(
                attention_diversity_loss(
                    attention_a, attention_b, eta_a, eta_b, self.config.diversity_eps
                )
            )
        l_adv = torch.stack(losses["adv"]).mean()
        l_attn = torch.stack(losses["attn"]).mean()
        l_div = torch.stack(losses["div"]).mean()
        objective = -l_adv + self.config.lambda_attn * l_attn - self.config.lambda_div * l_div
        return {
            "loss": objective,
            "l_adv": l_adv.detach(),
            "l_attn": l_attn.detach(),
            "l_div": l_div.detach(),
        }

    @torch.no_grad()
    def generate(self, images: Tensor, labels: Tensor, samples: int = 1) -> Tensor:
        """Generate one or more clipped adversarial samples per input."""
        if not self.initialized:
            self.initialize(images, labels)
        with torch.enable_grad():
            clean_attention = self.attention(
                preprocess(images), labels, create_graph=False
            ).detach()
        with torch.no_grad():
            latent = self.diffusion.encode(to_diffusion_range(images), self.config.tau_star)
            outputs = []
            for _ in range(samples):
                support = self.neighborhood.sample(clean_attention)
                mask = self.neighborhood.to_mask(support, latent.shape[-2:])
                eta = torch.randn(
                    images.shape[0], self.config.latent_dim, device=images.device
                )
                residual = self.generator(latent, eta) * mask
                outputs.append(self._decode_clipped(latent + residual, images))
            return torch.stack(outputs, dim=1)

    def close(self):
        self.attention.close()

