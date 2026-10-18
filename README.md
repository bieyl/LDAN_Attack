# Latent diffusion-based attention-guided neighborhood perturbation

This repository contains the PyTorch implementation of **Latent diffusion-based attention-guided neighborhood perturbation for transferable adversarial attacks**.

The implementation follows the paper's training pipeline:

1. a frozen latent-diffusion backbone maps an input image to a latent representation;
2. a frozen surrogate classifier provides a channel-wise Grad-CAM tensor;
3. a learnable sampler draws an attention support inside a Frobenius ball and maps it to a latent gate;
4. a latent residual generator produces two perturbations under the same support;
5. the objective combines surrogate attack loss, neighborhood alignment, and attention-space diversity;
6. decoded images are clipped to the image-space $L_\infty$ budget before evaluation.

The repository does not include datasets, model checkpoints, generated images, or diffusion weights. They must be supplied by the user under the paths in `configs/default.yaml`.

## Repository layout

```text
configs/default.yaml       Paper configuration
src/ldan_attack/           Diffusion, attention, generator, losses, and pipeline
scripts/train_generator.py Train the latent perturbation generator
scripts/generate_attack.py Generate adversarial images
scripts/evaluate.py        Evaluate target-model and ensemble ASR
scripts/smoke_test.py      Dependency/runtime shape test
data/README.md             Public dataset description
checkpoints/README.md      Checkpoint policy
outputs/                   Ignored generated outputs
```

## Environment

The code targets Python 3.10 or newer and a CUDA-enabled PyTorch installation for practical training. Install the Python dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The default configuration uses the public Stable Diffusion v1.5 checkpoint through Diffusers. The checkpoint is not redistributed here; set `diffusion_model` to the locally available checkpoint identifier or path.

## Data

The experiments use the public ImageNet validation images for generator training and the NeurIPS 2017 adversarial competition images for transfer evaluation. The auxiliary robustness experiment uses the public Caltech101 dataset. The repository intentionally contains no data and does not require a dataset download during installation.

Arrange an ImageFolder-compatible layout, or change `train_dir` and `eval_dir` in the YAML configuration:

```text
data/
  imagenet/train/<class>/*.JPEG
  nips2017/<class>/*
```

For a flat evaluation directory, provide a manifest instead. Set
`eval_manifest: labels.csv`; the CSV must contain a header with `filename` (or
`path`) and integer `label` columns, with paths relative to `eval_dir`.

## Training

The following command trains the proposed method with the paper defaults. `--use-ddim` enables the full DDIM inverse/denoise path and requires a complete Stable Diffusion checkpoint.

```bash
python scripts/train_generator.py \
  --config configs/default.yaml \
  --surrogate inception_v3 \
  --target-layer Mixed_7c \
  --use-ddim
```

For a quick tensor-shape check without a diffusion checkpoint, run:

```bash
python scripts/smoke_test.py
```

The toy adapter is for software tests only and must not be used to reproduce paper results.

## Generation and evaluation

```bash
python scripts/generate_attack.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/inception_v3_latent_attention.pt \
  --use-ddim

python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/inception_v3_latent_attention.pt \
  --targets inception_v3 inception_v4 inception_resnet_v2 resnet_v2 vgg16 vit_b_16 swin_b \
  --use-ddim
```

The ensemble evaluation excludes the surrogate model when its name matches the configured source model. The target models remain frozen and are used only for evaluation.

The supported benchmark target names are `inception_v3`, `inception_v4`,
`inception_resnet_v2`, `resnet_v2`, `vgg16`, `vit_b_16`, and `swin_b`.
`resnet_v2` uses the pre-activation `preresnet152` model supplied by
`pytorchcv`, with a `timm` ResNet-v2 fallback when `pytorchcv` is unavailable.
The model weights and the diffusion checkpoint are loaded by the user and are
not included in this repository.

## Reproducibility notes

The attack uses `N_diff=50`, `tau*=25`, `r=0.10`, `K=2`, `epsilon=16/255`, `lambda_attn=10`, and `lambda_div=1000` by default. The exact diffusion checkpoint, scheduler version, target layer, preprocessing, and random seed should be recorded with every run. Do not commit private checkpoints or generated adversarial images.

The default diffusion arithmetic is float32 because training differentiates
through the frozen decoder and the attention losses. Set `diffusion_dtype` to
`float16` only when the hardware and the installed checkpoint have been
validated with mixed-precision training.

The repository provides the method implementation and saves the attack-shape
and diffusion settings in each generator checkpoint. Numerical reproduction
also requires supplying the same diffusion/classifier weights and benchmark
labels used for the reported tables. The included smoke test checks tensor
shapes and gradient connectivity only; it is not a substitute for the
ImageNet/NeurIPS evaluation.

## Citation

```bibtex
@article{bie2026latentattention,
  title={Latent diffusion-based attention-guided neighborhood perturbation for transferable adversarial attacks},
  author={Bie, Yunlong and Sun, Jiaze},
  journal={Image and Vision Computing},
  year={2026},
  note={Manuscript ID: IMAVIS-D-26-01040}
}
```

