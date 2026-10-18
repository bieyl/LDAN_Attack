"""Configuration objects for the latent attention attack."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AttackConfig:
    """The defaults reported for the paper's ImageNet transfer protocol."""

    image_size: int = 224
    epsilon: float = 16.0 / 255.0
    diffusion_steps: int = 50
    tau_star: int = 25
    radius: float = 0.10
    neighborhood_samples: int = 2
    latent_dim: int = 16
    lambda_attn: float = 10.0
    lambda_div: float = 1000.0
    diversity_eps: float = 1.0e-8
    projection_eps: float = 1.0e-8
    normalization_eps: float = 1.0e-6
    generator_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    num_workers: int = 4
    seed: int = 2024
    surrogate: str = "inception_v3"
    target_layer: str = "Mixed_7c"
    diffusion_model: str = "runwayml/stable-diffusion-v1-5"
    train_dir: str = "./data/imagenet/train"
    eval_dir: str = "./data/nips2017"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda"

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "AttackConfig":
        """Build a config while rejecting misspelled fields."""
        valid = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError("Unknown configuration fields: {}".format(", ".join(unknown)))
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> "AttackConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without optional deps
            raise RuntimeError("PyYAML is required to read configuration files") from exc
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls.from_dict(values)

    def resolve_device(self) -> str:
        """Return CUDA when requested and available, otherwise CPU."""
        if self.device != "cuda":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

