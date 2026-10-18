#!/usr/bin/env python
"""Train the latent perturbation generator for one surrogate model."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ldan_attack.config import AttackConfig
from ldan_attack.diffusion import DiffusersLatentAdapter
from ldan_attack.models import build_classifier, default_target_layer
from ldan_attack.pipeline import LatentAttentionAttack
from ldan_attack.toy import ToyLatentAdapter
from ldan_attack.utils import make_imagefolder, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--surrogate", default=None)
    parser.add_argument("--target-layer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--toy-diffusion", action="store_true",
                        help="Use the shape-test adapter; never use it for paper results.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = AttackConfig.from_yaml(args.config)
    if args.surrogate:
        config.surrogate = args.surrogate
        if not args.target_layer:
            config.target_layer = default_target_layer(config.surrogate)
    if args.target_layer:
        config.target_layer = args.target_layer
    elif not config.target_layer:
        config.target_layer = default_target_layer(config.surrogate)
    if args.device:
        config.device = args.device
    device = torch.device(config.resolve_device())
    diffusion_dtype = torch.float16 if config.diffusion_dtype.lower() in {"float16", "fp16"} else torch.float32
    seed_everything(config.seed)

    loader = make_imagefolder(
        config.train_dir, config.image_size, config.batch_size, shuffle=True,
        num_workers=config.num_workers, manifest=config.train_manifest,
    )
    surrogate = build_classifier(config.surrogate, pretrained=True).to(device).eval()
    if config.target_layer not in dict(surrogate.named_modules()):
        if args.target_layer:
            raise ValueError(
                "Target layer '{}' is not present in the {} surrogate".format(
                    config.target_layer, config.surrogate
                )
            )
        config.target_layer = default_target_layer(config.surrogate, surrogate)
    if args.toy_diffusion:
        diffusion = ToyLatentAdapter().to(device)
    else:
        diffusion = DiffusersLatentAdapter(
            config.diffusion_model,
            device=device,
            diffusion_steps=config.diffusion_steps,
            use_ddim=args.use_ddim,
            dtype=diffusion_dtype,
        )
    attack = LatentAttentionAttack(surrogate, diffusion, config).to(device)

    first_images, first_labels = next(iter(loader))
    attack.initialize(first_images.to(device), first_labels.to(device))
    optimizer = torch.optim.Adam(
        list(attack.generator.parameters()) + list(attack.neighborhood.parameters()),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.generator_epochs):
        attack.train()
        meters = {"loss": 0.0, "l_adv": 0.0, "l_attn": 0.0, "l_div": 0.0}
        count = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            values = attack.training_step(images, labels)
            values["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(attack.generator.parameters()) + list(attack.neighborhood.parameters()), 1.0
            )
            optimizer.step()
            batch_size = images.shape[0]
            count += batch_size
            for key in meters:
                meters[key] += float(values[key].detach()) * batch_size
        report = {key: value / max(count, 1) for key, value in meters.items()}
        print("epoch {:03d}: {}".format(epoch + 1, json.dumps(report, sort_keys=True)))

    output_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "generator": attack.generator.state_dict(),
        "neighborhood": attack.neighborhood.state_dict(),
        "config": config.__dict__,
        "surrogate": config.surrogate,
        "target_layer": config.target_layer,
    }
    path = output_dir / (config.surrogate + "_latent_attention.pt")
    torch.save(checkpoint, path)
    print("saved checkpoint to {}".format(path))
    attack.close()


if __name__ == "__main__":
    main()

