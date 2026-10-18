#!/usr/bin/env python
"""Evaluate ASR on target models and the source-excluded ensemble."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ldan_attack.config import AttackConfig
from ldan_attack.diffusion import DiffusersLatentAdapter
from ldan_attack.models import build_classifier, preprocess
from ldan_attack.pipeline import LatentAttentionAttack
from ldan_attack.utils import make_imagefolder


def output_logits(model, images):
    output = model(images)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets", nargs="+",
                        default=["inception_v3", "inception_v4", "inception_resnet_v2",
                                 "resnet_v2", "vgg16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--output", default="./outputs/evaluation.json")
    args = parser.parse_args()
    config = AttackConfig.from_yaml(args.config)
    if args.device:
        config.device = args.device
    device = torch.device(config.resolve_device())
    loader = make_imagefolder(config.eval_dir, config.image_size, config.batch_size, False,
                              config.num_workers)
    surrogate = build_classifier(config.surrogate, pretrained=True).to(device).eval()
    diffusion = DiffusersLatentAdapter(
        config.diffusion_model, device=device, diffusion_steps=config.diffusion_steps,
        use_ddim=args.use_ddim,
    )
    attack = LatentAttentionAttack(surrogate, diffusion, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    totals = {name: 0 for name in args.targets}
    total = 0
    ensemble_success = 0
    models = {name: build_classifier(name, pretrained=True).to(device).eval()
              for name in args.targets}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        attack.initialize(images, labels)
        attack.generator.load_state_dict(checkpoint["generator"])
        attack.neighborhood.load_state_dict(checkpoint["neighborhood"])
        adversarial = attack.generate(images, labels, samples=1)[:, 0]
        ensemble_logits = None
        for name, model in models.items():
            logits = output_logits(model, preprocess(adversarial))
            predictions = logits.argmax(dim=1)
            totals[name] += int((predictions != labels).sum())
            if name != config.surrogate:
                ensemble_logits = logits if ensemble_logits is None else ensemble_logits + logits
        ensemble_success += int((ensemble_logits.argmax(dim=1) != labels).sum())
        total += labels.numel()
    report = {name: 100.0 * value / max(total, 1) for name, value in totals.items()}
    report["ensemble_excluding_surrogate"] = 100.0 * ensemble_success / max(total, 1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    attack.close()


if __name__ == "__main__":
    main()

