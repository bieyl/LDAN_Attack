# Latent diffusion-based attention-guided neighborhood perturbation

This repository contains the PyTorch implementation of **Latent diffusion-based attention-guided neighborhood perturbation for transferable adversarial attacks**.

## Overview

The method trains a perturbation generator in the latent space of a frozen diffusion model. A frozen surrogate classifier provides a channel-wise Grad-CAM representation. A learnable sampler explores a Frobenius neighborhood of that representation and maps each sampled support to a latent gating mask. During training, two latent codes share the same attention support, and the generator is optimized with the untargeted surrogate loss, neighborhood alignment, and attention-space diversity terms. The decoded outputs are clipped to the image-space $L_{\infty}$ budget before evaluation.

The public implementation follows the paper's DDIM inversion and denoising path. It requires the complete diffusion checkpoint and the classifier weights supplied by the user. Datasets, checkpoints, generated images, and diffusion weights are intentionally excluded from this repository.

## Repository layout

```text
configs/default.yaml       Paper configuration
src/ldan_attack/           Diffusion, attention, generator, losses, and pipeline
scripts/train_generator.py Train the latent perturbation generator
scripts/generate_attack.py Generate adversarial images
scripts/evaluate.py        Evaluate target-model and ensemble ASR
data/README.md             Public dataset description
checkpoints/README.md      Checkpoint policy
outputs/                   Ignored generated outputs
```

## Environment

The code targets Python 3.10 or newer and a CUDA-enabled PyTorch installation for practical training. Install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The default configuration uses the public Stable Diffusion v1.5 checkpoint through Diffusers. The checkpoint is not redistributed here; set `diffusion_model` to the locally available checkpoint identifier or path. The implementation uses DDIM inversion and denoising with the fixed operating point specified in the configuration.

## Data

The experiments use 10,000 randomly selected images from the public ImageNet validation set for generator training and 1,000 images from the NeurIPS 2017 adversarial competition set for transfer evaluation. The auxiliary robustness experiment uses the public Caltech101 dataset. The repository contains no data and does not require a dataset download during installation.

Arrange an ImageFolder-compatible layout, or change `train_dir` and `eval_dir` in the YAML configuration:

```text
data/
  imagenet/val/<class>/*.JPEG
  nips2017/<class>/*
```

For a flat evaluation directory, provide a manifest instead. Set `eval_manifest: labels.csv`; the CSV must contain a header with `filename` (or `path`) and integer `label` columns, with paths relative to `eval_dir`.

## Training

The following command trains the proposed method with the paper defaults. It loads the supplied Stable Diffusion checkpoint and uses the full DDIM inversion and denoising path.

```bash
python scripts/train_generator.py \
  --config configs/default.yaml \
  --surrogate inception_v3 \
  --target-layer Mixed_7c
```

The training checkpoint stores the attack configuration, source model, and target layer used for the run. Generation and evaluation restore the configuration saved with the checkpoint.

## Generation and evaluation

```bash
python scripts/generate_attack.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/inception_v3_latent_attention.pt

python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/inception_v3_latent_attention.pt \
  --targets inception_v3 inception_v4 inception_resnet_v2 resnet_v2 vgg16 vit_b_16 swin_b
```

The ensemble evaluation excludes the surrogate model when its name matches the configured source model. The target models remain frozen and are used only for evaluation.

The supported benchmark target names are `inception_v3`, `inception_v4`, `inception_resnet_v2`, `resnet_v2`, `vgg16`, `vit_b_16`, and `swin_b`. `resnet_v2` uses the pre-activation `preresnet152` model supplied by `pytorchcv`, with a `timm` ResNet-v2 fallback when `pytorchcv` is unavailable. The model weights and the diffusion checkpoint are loaded by the user and are not included in this repository.

## Reproducibility notes

The paper configuration uses `N_diff=50`, `tau*=25`, `r=0.10`, `K=2`, `epsilon=16/255`, `lambda_attn=10`, and `lambda_div=1000`. Here, `K` is the number of sampled attention supports per training example. Record the diffusion checkpoint, scheduler and library versions, surrogate model and target layer, preprocessing, data manifest, and random seed with every run. Do not commit private checkpoints or generated adversarial images.

The default diffusion arithmetic is `float32` because training differentiates through the frozen denoising and decoder path with respect to the latent variables. Set `diffusion_dtype` to `float16` only after validating mixed-precision training with the selected hardware, checkpoint, and Diffusers version.

The repository stores the attack-shape and diffusion configuration in each generator checkpoint. The checkpoint does not contain the diffusion or classifier weights, software-version metadata, or benchmark labels. Numerical reproduction therefore requires supplying the same model weights, scheduler and library versions, preprocessing, and labels used for the reported tables.

## Citation

```bibtex
@misc{bie2026latentattention,
  title={Latent diffusion-based attention-guided neighborhood perturbation for transferable adversarial attacks},
  author={Bie, Yunlong and Sun, Jiaze},
  year={2026}
}
```
