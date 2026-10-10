from __future__ import annotations

from torch import nn
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models.squeezenet import SqueezeNet

FINETUNE_STRATEGIES: tuple[str, ...] = ("head_only", "last_n_blocks", "full")


def apply_finetune_strategy(
    model: nn.Module,
    strategy: str,
    unfreeze_last_n: int = 0,
) -> nn.Module:
    """Set requires_grad according to the selected fine-tuning strategy."""
    if strategy not in FINETUNE_STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; expected one of {list(FINETUNE_STRATEGIES)}"
        )
    if not isinstance(model, (MobileNetV2, SqueezeNet)):
        raise TypeError("fine-tuning strategies support MobileNetV2 and SqueezeNet only")

    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if strategy == "head_only":
        return model

    blocks = list(model.features.children())
    if unfreeze_last_n < 1:
        raise ValueError("unfreeze_last_n must be at least 1 for last_n_blocks")
    if unfreeze_last_n > len(blocks):
        raise ValueError(f"cannot unfreeze {unfreeze_last_n} blocks; model has only {len(blocks)}")

    for block in blocks[-unfreeze_last_n:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    return model


def replace_classifier(
    model: nn.Module,
    num_classes: int,
    dropout: float = 0.2,
) -> nn.Module:
    """Replace an ImageNet head with a num_classes head."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1)")

    if isinstance(model, MobileNetV2):
        old_head = model.classifier[1]
        if not isinstance(old_head, nn.Linear):
            raise TypeError("MobileNetV2 classifier[1] must be Linear")

        model.classifier[0] = nn.Dropout(p=dropout)
        new_head = nn.Linear(old_head.in_features, num_classes)
        nn.init.normal_(new_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(new_head.bias)
        model.classifier[1] = new_head

    elif isinstance(model, SqueezeNet):
        old_head = model.classifier[1]
        if not isinstance(old_head, nn.Conv2d):
            raise TypeError("SqueezeNet classifier[1] must be Conv2d")

        model.classifier[0] = nn.Dropout(p=dropout)
        new_head = nn.Conv2d(
            old_head.in_channels,
            num_classes,
            kernel_size=1,
        )
        nn.init.normal_(new_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(new_head.bias)
        model.classifier[1] = new_head

    else:
        raise TypeError("replace_classifier supports MobileNetV2 and SqueezeNet only")

    model.num_classes = num_classes
    return model


def count_trainable_vs_total(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return trainable, total
