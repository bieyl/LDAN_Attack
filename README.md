# Latent diffusion-based attention-guided neighborhood perturbation

This repository provides the core PyTorch implementation of **Latent diffusion-based attention-guided neighborhood perturbation for transferable adversarial attacks**. The method trains a latent residual generator with a frozen diffusion backbone and a frozen surrogate classifier. A channel-wise Grad-CAM representation is sampled within a bounded Frobenius neighborhood and mapped to a latent gating mask. The training objective combines untargeted surrogate loss, neighborhood alignment, and attention-response diversity. Decoded examples are clipped to the prescribed image-space `L_inf` budget.

The repository contains the attack implementation, training script, image-generation script, and ASR evaluation script. Datasets, model weights, generated images, and baseline implementations are not included.

## Requirements

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data and model weights

The experiments use public datasets: ImageNet validation images for generator training, the NeurIPS 2017 adversarial competition set for transfer evaluation, and Caltech101 for the auxiliary adversarial-training experiment. The repository contains no data; provide local dataset paths through `configs/default.yaml`.

Provide a compatible latent-diffusion checkpoint and the pretrained classifier weights required by the selected surrogate and target models. The default configuration uses Stable Diffusion v1.5 through Diffusers.

## Paper configuration

The default configuration follows the main transfer protocol in the paper: `224x224` inputs, `N_diff=50`, `tau*=25`, `r=0.10`, `K=2`, `epsilon=16/255`, `lambda_attn=10`, `lambda_div=1000`, batch size 8, and 100 generator-training epochs. The training and evaluation image limits are 10,000 and 1,000, respectively.

## Usage

Train the generator:

```bash
python scripts/train_generator.py \
  --config configs/default.yaml \
  --surrogate inception_v3
```

Generate adversarial images:

```bash
python scripts/generate_attack.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/inception_v3_latent_attention.pt
```

Evaluate transfer ASR and the source-excluded ensemble:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/inception_v3_latent_attention.pt \
  --targets inception_v3 inception_v4 inception_resnet_v2 resnet_v2 vgg16 vit_b_16 swin_b
```

For comparisons with the reported tables, use the same classifier weights, diffusion checkpoint, preprocessing, data split, and random seeds. Repeat training and evaluation runs when estimating the multi-run means reported in the paper.

## Repository layout

```text
configs/default.yaml       Paper configuration
src/ldan_attack/           Attack modules and diffusion adapter
scripts/train_generator.py Generator training
scripts/generate_attack.py Adversarial-image generation
scripts/evaluate.py        Transfer-ASR evaluation
data/README.md             Dataset layout
checkpoints/README.md      Checkpoint policy
```

## Citation

```bibtex
@misc{bie2026latentattention,
  title={Latent diffusion-based attention-guided neighborhood perturbation for transferable adversarial attacks},
  author={Bie, Yunlong and Sun, Jiaze},
  year={2026}
}
```


