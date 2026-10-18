#!/usr/bin/env python
"""Generate clipped adversarial images with a trained attack module."""

import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ldan_attack.config import AttackConfig
from ldan_attack.diffusion import DiffusersLatentAdapter
from ldan_attack.models import build_classifier, default_target_layer
from ldan_attack.pipeline import LatentAttentionAttack
from ldan_attack.utils import make_imagefolder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="./outputs/adversarial")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-ddim", action="store_true")
    args = parser.parse_args()
    config = AttackConfig.from_yaml(args.config)
    if args.device:
        config.device = args.device
    device = torch.device(config.resolve_device())
    target_layer = config.target_layer or default_target_layer(config.surrogate)
    loader = make_imagefolder(config.eval_dir, config.image_size, config.batch_size, False,
                              config.num_workers)
    surrogate = build_classifier(config.surrogate, pretrained=True).to(device).eval()
    diffusion = DiffusersLatentAdapter(
        config.diffusion_model, device=device, diffusion_steps=config.diffusion_steps,
        use_ddim=args.use_ddim,
    )
    attack = LatentAttentionAttack(surrogate, diffusion, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        attack.initialize(images, labels)
        attack.generator.load_state_dict(checkpoint["generator"])
        attack.neighborhood.load_state_dict(checkpoint["neighborhood"])
        generated = attack.generate(images, labels, samples=1)[:, 0]
        for offset, image in enumerate(generated):
            save_image(image.detach().cpu(), output_dir / "{:06d}.png".format(index * loader.batch_size + offset))
    attack.close()
    print("saved adversarial images to {}".format(output_dir))


if __name__ == "__main__":
    main()

