"""Adapters for the frozen latent-diffusion backbone.

The repository does not ship a diffusion checkpoint.  ``DiffusersLatentAdapter``
loads the checkpoint supplied by the user and keeps all diffusion parameters
frozen.  The VAE encode/decode path is available for quick checks; setting
``use_ddim=True`` enables the DDIM inverse/denoise path when the installed
Diffusers version provides the required scheduler classes.
"""

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DiffusionLatentAdapter(nn.Module):
    """Interface consumed by :class:`LatentAttentionAttack`."""

    latent_channels: int = 4

    def __init__(self):
        super().__init__()

    def encode(self, images: Tensor, tau_star: int) -> Tensor:
        raise NotImplementedError

    def decode(self, latents: Tensor, tau_star: int, output_size) -> Tensor:
        raise NotImplementedError


class DiffusersLatentAdapter(DiffusionLatentAdapter):
    """Stable-Diffusion VAE adapter with an optional DDIM trajectory."""

    def __init__(self, model_id: str, device: torch.device, diffusion_steps: int = 50,
                 use_ddim: bool = False, dtype: Optional[torch.dtype] = None):
        super().__init__()
        try:
            from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        except ImportError as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "Diffusers is required for the latent diffusion adapter. "
                "Install requirements.txt before running the attack."
            ) from exc

        self.device = device
        self.use_ddim = use_ddim
        self.diffusion_steps = diffusion_steps
        # The training objective differentiates through the frozen decoder.
        # Float32 is the safe default; callers may opt into fp16 explicitly.
        dtype = dtype or torch.float32
        self.compute_dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        self.vae.to(device).eval()
        self.unet = None
        self.scheduler = None
        self.inverse_scheduler = None
        self.text_encoder = None
        self.tokenizer = None
        if use_ddim:
            try:
                from diffusers import DDIMInverseScheduler

                self.unet = UNet2DConditionModel.from_pretrained(
                    model_id, subfolder="unet", torch_dtype=dtype
                ).to(device).eval()
                self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
                self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
                from transformers import CLIPTextModel, CLIPTokenizer

                self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_id, subfolder="text_encoder", torch_dtype=dtype
                ).to(device).eval()
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "use_ddim=True requires a complete Stable-Diffusion checkpoint and "
                    "a Diffusers version containing DDIMInverseScheduler."
                ) from exc
        for module in (self.vae, self.unet, self.text_encoder):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))

    @torch.no_grad()
    def _empty_prompt_embeddings(self, batch_size: int):
        if self.tokenizer is None or self.text_encoder is None:
            return None
        tokens = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(tokens)[0]

    def _vae_encode(self, images: Tensor) -> Tensor:
        # VAE expects image coordinates in [-1, 1].
        posterior = self.vae.encode(images.to(self.compute_dtype)).latent_dist
        return posterior.mean * self.scaling_factor

    def encode(self, images: Tensor, tau_star: int) -> Tensor:
        latents = self._vae_encode(images)
        if not self.use_ddim:
            return latents.float()
        self.inverse_scheduler.set_timesteps(self.diffusion_steps, device=images.device)
        embeddings = self._empty_prompt_embeddings(images.shape[0])
        # DDIM inversion moves from the clean latent toward the requested
        # operating point.  The exact trajectory is fixed by the checkpoint.
        for timestep in self.inverse_scheduler.timesteps[: max(1, tau_star)]:
            prediction = self.unet(
                latents, timestep, encoder_hidden_states=embeddings
            ).sample
            latents = self.inverse_scheduler.step(prediction, timestep, latents).prev_sample
        return latents.float()

    def decode(self, latents: Tensor, tau_star: int, output_size) -> Tensor:
        latents = latents.to(self.compute_dtype)
        if not self.use_ddim:
            decoded = self.vae.decode(latents / self.scaling_factor).sample
            decoded = decoded.float()
            return decoded if decoded.shape[-2:] == tuple(output_size) else F.interpolate(
                decoded, size=output_size, mode="bilinear", align_corners=False
            )
        self.scheduler.set_timesteps(self.diffusion_steps, device=latents.device)
        embeddings = self._empty_prompt_embeddings(latents.shape[0])
        # The scheduler stores descending denoising timesteps.  Starting at
        # tau_star reproduces the paper's fixed injection point convention.
        timesteps = self.scheduler.timesteps
        start = min(max(int(tau_star), 0), len(timesteps) - 1)
        sample = latents
        for timestep in timesteps[start:]:
            prediction = self.unet(
                sample, timestep, encoder_hidden_states=embeddings
            ).sample
            sample = self.scheduler.step(prediction, timestep, sample).prev_sample
        decoded = self.vae.decode(sample / self.scaling_factor).sample
        decoded = decoded.float()
        return decoded if decoded.shape[-2:] == tuple(output_size) else F.interpolate(
            decoded, size=output_size, mode="bilinear", align_corners=False
        )

