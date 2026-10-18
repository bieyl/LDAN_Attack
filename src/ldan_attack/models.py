"""Classifier construction and ImageNet preprocessing."""

from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass
class ModelSpec:
    name: str
    input_size: int = 224


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(images: Tensor, device=None) -> Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=device or images.device, dtype=images.dtype)
    std = torch.tensor(IMAGENET_STD, device=device or images.device, dtype=images.dtype)
    return (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def unwrap_model_output(output):
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError("Unsupported model output: {}".format(type(output)))


def build_classifier(name: str, pretrained: bool = True) -> nn.Module:
    """Build a torchvision ImageNet classifier by canonical paper name."""
    import torchvision.models as models

    name = name.lower()
    if name == "inception_v3":
        try:
            weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
            model = models.inception_v3(weights=weights,
                                        aux_logits=False)
        except (AttributeError, TypeError):
            model = models.inception_v3(pretrained=pretrained, aux_logits=False)
    elif name in {"resnet_v2", "resnet152"}:
        # The benchmark's ResNet-v2 is a pre-activation model.  timm exposes
        # the published BiT ResNet-v2 weights; do not silently substitute a
        # post-activation torchvision ResNet because that changes the target.
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "resnet_v2 requires timm (see requirements.txt) so that the "
                "pre-activation ResNet-v2 target is used."
            ) from exc
        model = timm.create_model("resnetv2_152x2_bitm", pretrained=pretrained,
                                  num_classes=1000)
    elif name in {"vgg16", "vgg_16"}:
        try:
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            model = models.vgg16(weights=weights)
        except (AttributeError, TypeError):
            model = models.vgg16(pretrained=pretrained)
    elif name in {"inception_v4", "inception_resnet_v2"}:
        try:
            import pretrainedmodels
        except ImportError as exc:
            raise RuntimeError(
                "{} requires the optional pretrainedmodels package".format(name)
            ) from exc
        model = pretrainedmodels.__dict__[name.replace("_", "")](
            num_classes=1000, pretrained="imagenet" if pretrained else None
        )
    elif name in {"vit_b_16", "vit_b16", "vit-b/16"}:
        try:
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
        except (AttributeError, TypeError):
            model = models.vit_b_16(pretrained=pretrained)
    elif name in {"swin_b", "swin-b"}:
        try:
            weights = models.Swin_B_Weights.DEFAULT if pretrained else None
            model = models.swin_b(weights=weights)
        except (AttributeError, TypeError):
            model = models.swin_b(pretrained=pretrained)
    else:
        raise ValueError("Unsupported classifier '{}'.".format(name))
    return model


def default_target_layer(name: str) -> str:
    return {
        "inception_v3": "Mixed_7c",
        "inception_v4": "features.21",
        "inception_resnet_v2": "conv2d_7b",
        "resnet_v2": "stages.3.blocks.2.conv3",
        "vgg16": "features.28",
        "vgg_16": "features.28",
    }.get(name.lower(), "")

