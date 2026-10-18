"""Data, reproducibility, and image-space utilities."""

import os
import random
import csv
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestImageDataset(Dataset):
    """Image dataset backed by a ``filename,label`` CSV manifest.

    This supports flat benchmark directories such as the NeurIPS 2017
    transfer set while retaining ImageFolder support for ImageNet subsets.
    Paths in the manifest are interpreted relative to ``root``.
    """

    def __init__(self, root: str, manifest: str, image_size: int):
        self.root = Path(root)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        self.samples = self._read_manifest(manifest_path)
        if not self.samples:
            raise ValueError("Manifest contains no image-label pairs: {}".format(manifest_path))

    @staticmethod
    def _read_manifest(path: Path):
        if not path.exists():
            raise FileNotFoundError("Label manifest not found: {}".format(path))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first = handle.readline()
            handle.seek(0)
            delimiter = "," if "," in first else None
            if delimiter:
                reader = csv.DictReader(handle)
                fields = {field.strip().lower(): field for field in (reader.fieldnames or [])}
                image_field = next((fields[key] for key in ("filename", "file", "path", "image")
                                    if key in fields), None)
                label_field = next((fields[key] for key in ("label", "class", "target")
                                    if key in fields), None)
                if image_field is None or label_field is None:
                    raise ValueError("CSV manifest must contain filename/path and label columns")
                return [(row[image_field].strip(), int(row[label_field]))
                        for row in reader if row.get(image_field, "").strip()]
            samples = []
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                image, label = line.rsplit(None, 1)
                samples.append((image, int(label)))
            return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative_path, label = self.samples[index]
        with Image.open(self.root / relative_path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def make_imagefolder(path: str, image_size: int, batch_size: int, shuffle: bool,
                     num_workers: int = 4, manifest: Optional[str] = None,
                     max_samples: Optional[int] = None, seed: Optional[int] = None) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    root = Path(path)
    manifest_path = manifest
    if manifest_path is None:
        for candidate in ("labels.csv", "metadata.csv", "labels.txt"):
            if (root / candidate).exists():
                manifest_path = candidate
                break
    if manifest_path is not None:
        dataset = ManifestImageDataset(path, manifest_path, image_size)
    else:
        dataset = ImageFolder(path, transform=transform)
    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        generator = random.Random(seed)
        indices = generator.sample(range(len(dataset)), max_samples)
        dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def to_diffusion_range(images: Tensor) -> Tensor:
    return images.mul(2.0).sub(1.0)


def from_diffusion_range(images: Tensor) -> Tensor:
    return images.add(1.0).div(2.0)


def clip_image(images: Tensor) -> Tensor:
    return images.clamp(0.0, 1.0)


def save_tensor_image(images: Tensor, path: str):
    from torchvision.utils import save_image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(images.detach().cpu().clamp(0.0, 1.0), path, nrow=1)

