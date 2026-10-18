"""Differentiable Grad-CAM attention extraction.

The extractor keeps the channel-wise Grad-CAM tensor used by the method rather
than collapsing it to a single heat map.  ``create_graph=True`` is required
when the attention alignment/diversity losses are back-propagated through an
adversarial image to the perturbation generator.
"""

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def unwrap_logits(output) -> Tensor:
    """Return logits from torchvision, timm, or tuple-style model outputs."""
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("Unsupported classifier output type: {}".format(type(output)))


def normalize_attention(attention: Tensor, eps: float = 1.0e-6) -> Tensor:
    """Apply channel-wise L2 normalization to a ``B x C x H x W`` tensor."""
    if attention.ndim != 4:
        raise ValueError("attention must have shape B x C x H x W")
    norm = attention.flatten(2).norm(p=2, dim=2, keepdim=True)
    return attention / (norm.unsqueeze(-1) + eps)


class GradCAMExtractor:
    """Extract class-conditioned channel-wise Grad-CAM tensors."""

    def __init__(self, model: nn.Module, target_layer: str, eps: float = 1.0e-6):
        self.model = model
        self.target_layer = target_layer
        self.eps = eps
        modules: Dict[str, nn.Module] = dict(model.named_modules())
        if target_layer not in modules:
            available = ", ".join(list(modules)[:20])
            raise ValueError(
                "Unknown target layer '{}'. Example available layers: {}".format(
                    target_layer, available
                )
            )
        self._activations: Optional[Tensor] = None
        self._handle = modules[target_layer].register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self._activations = output

    def __call__(self, images: Tensor, labels: Tensor, create_graph: bool = True) -> Tensor:
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        images = images.requires_grad_(True)
        self._activations = None
        logits = unwrap_logits(self.model(images))
        if self._activations is None:
            raise RuntimeError("The target layer did not produce an activation")
        scores = logits.gather(1, labels[:, None]).sum()
        gradients = torch.autograd.grad(
            scores,
            self._activations,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=False,
        )[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        attention = weights * self._activations
        return normalize_attention(attention, self.eps)

    def close(self):
        self._handle.remove()


def freeze(module: nn.Module):
    """Freeze parameters while retaining gradients with respect to inputs."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()

