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
from ldan_attack.utils import make_imagefolder, seed_everything


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
                                 "resnet_v2", "vgg16", "vit_b_16", "swin_b"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-ddim", dest="use_ddim", action="store_true", default=None,
                        help="Use the DDIM inversion/denoising path.")
    parser.add_argument("--no-ddim", dest="use_ddim", action="store_false",
                        help="Use the VAE-only adapter for debugging; not a paper run.")
    parser.add_argument("--output", default="./outputs/evaluation.json")
    args = parser.parse_args()
    config = AttackConfig.from_yaml(args.config)
    if args.device:
        config.device = args.device
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config.apply_checkpoint_settings(checkpoint.get("config", {}))
    if checkpoint.get("surrogate"):
        config.surrogate = checkpoint["surrogate"]
    if checkpoint.get("target_layer"):
        config.target_layer = checkpoint["target_layer"]
    use_ddim = checkpoint.get("use_ddim", True) if args.use_ddim is None else args.use_ddim
    seed_everything(config.seed)
    device = torch.device(config.resolve_device())
    diffusion_dtype = torch.float16 if config.diffusion_dtype.lower() in {"float16", "fp16"} else torch.float32
    loader = make_imagefolder(config.eval_dir, config.image_size, config.batch_size, False,
                              config.num_workers, manifest=config.eval_manifest,
                              max_samples=config.eval_images, seed=config.seed)
    surrogate = build_classifier(config.surrogate, pretrained=True).to(device).eval()
    diffusion = DiffusersLatentAdapter(
        config.diffusion_model, device=device, diffusion_steps=config.diffusion_steps,
        use_ddim=use_ddim, dtype=diffusion_dtype,
    )
    attack = LatentAttentionAttack(surrogate, diffusion, config).to(device)
    checkpoint = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in checkpoint.items()
    }
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
        ensemble_probs = None
        with torch.no_grad():
            for name, model in models.items():
                logits = output_logits(model, preprocess(adversarial))
                predictions = logits.argmax(dim=1)
                totals[name] += int((predictions != labels).sum())
                if name != config.surrogate:
                    probs = torch.softmax(logits, dim=1)
                    ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs
        if ensemble_probs is None:
            raise ValueError(
                "The ensemble requires at least one target model other than the surrogate."
            )
        non_source_count = sum(name != config.surrogate for name in models)
        ensemble_probs = ensemble_probs / non_source_count
        ensemble_success += int((ensemble_probs.argmax(dim=1) != labels).sum())
        total += labels.numel()
    report = {name: 100.0 * value / max(total, 1) for name, value in totals.items()}
    report["ensemble_excluding_surrogate"] = 100.0 * ensemble_success / max(total, 1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    attack.close()


if __name__ == "__main__":
    main()

