from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchvision.models import (
    MobileNet_V2_Weights,
    SqueezeNet1_1_Weights,
    mobilenet_v2,
    squeezenet1_1,
)

from edgecnn.models.pretrained.finetune import (
    apply_finetune_strategy,
    replace_classifier,
)
from edgecnn.models.registry import register

if TYPE_CHECKING:
    from torch import nn


@register("mobilenet_v2")
def build_mobilenet_v2(
    num_classes: int,
    input_shape: tuple[int, int, int] = (3, 64, 64),
    **overrides: object,
) -> nn.Module:
    settings = dict(overrides)
    torchvision_name = str(settings.pop("torchvision_name", "mobilenet_v2"))
    if torchvision_name != "mobilenet_v2":
        raise ValueError(f"expected torchvision_name='mobilenet_v2', got {torchvision_name!r}")

    weights = _resolve_weights(
        MobileNet_V2_Weights,
        settings.pop("weights", "IMAGENET1K_V1"),
    )
    strategy = str(settings.pop("finetune_strategy", "last_n_blocks"))
    unfreeze_last_n = int(settings.pop("unfreeze_last_n", 3))
    should_replace = bool(settings.pop("replace_classifier", True))
    dropout = float(settings.pop("dropout", 0.2))
    input_resolution = int(settings.pop("input_resolution", input_shape[-1]))

    settings.pop("backbone_lr_scale", None)

    if settings:
        raise TypeError("unsupported MobileNetV2 settings: " + ", ".join(sorted(settings)))
    if not should_replace:
        raise ValueError("the classifier must be replaced for EuroSAT")

    _check_resolution(input_shape, input_resolution)

    model = mobilenet_v2(weights=weights)
    replace_classifier(model, num_classes, dropout)
    apply_finetune_strategy(model, strategy, unfreeze_last_n)
    return model


@register("squeezenet1_1")
def build_squeezenet(
    num_classes: int,
    input_shape: tuple[int, int, int] = (3, 64, 64),
    **overrides: object,
) -> nn.Module:
    settings = dict(overrides)
    torchvision_name = str(settings.pop("torchvision_name", "squeezenet1_1"))
    if torchvision_name != "squeezenet1_1":
        raise ValueError(f"expected torchvision_name='squeezenet1_1', got {torchvision_name!r}")

    weights = _resolve_weights(
        SqueezeNet1_1_Weights,
        settings.pop("weights", "IMAGENET1K_V1"),
    )
    strategy = str(settings.pop("finetune_strategy", "last_n_blocks"))
    unfreeze_last_n = int(settings.pop("unfreeze_last_n", 2))
    should_replace = bool(settings.pop("replace_classifier", True))
    dropout = float(settings.pop("dropout", 0.5))
    input_resolution = int(settings.pop("input_resolution", input_shape[-1]))

    settings.pop("backbone_lr_scale", None)

    if settings:
        raise TypeError("unsupported SqueezeNet settings: " + ", ".join(sorted(settings)))
    if not should_replace:
        raise ValueError("the classifier must be replaced for EuroSAT")

    _check_resolution(input_shape, input_resolution)

    model = squeezenet1_1(weights=weights)
    replace_classifier(model, num_classes, dropout)
    apply_finetune_strategy(model, strategy, unfreeze_last_n)
    return model


def param_groups(
    model: nn.Module,
    base_lr: float,
    backbone_lr_scale: float,
) -> list[dict[str, object]]:
    """Return separate trainable backbone and classifier groups."""
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    if not 0 < backbone_lr_scale <= 1:
        raise ValueError("backbone_lr_scale must be in the range (0, 1]")
    if not hasattr(model, "classifier"):
        raise TypeError("pretrained model must expose .classifier")

    head_parameters = [
        parameter for parameter in model.classifier.parameters() if parameter.requires_grad
    ]
    head_ids = {id(parameter) for parameter in head_parameters}

    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]

    groups: list[dict[str, object]] = []
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": base_lr * backbone_lr_scale,
                "name": "backbone",
            }
        )
    if head_parameters:
        groups.append(
            {
                "params": head_parameters,
                "lr": base_lr,
                "name": "head",
            }
        )

    if not groups:
        raise ValueError("model has no trainable parameters")

    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    grouped_ids = {id(parameter) for group in groups for parameter in group["params"]}
    if grouped_ids != expected_ids:
        raise RuntimeError("parameter groups must contain every trainable parameter exactly once")

    return groups


def _resolve_weights(enum_type: Any, value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    if not isinstance(value, str):
        return value
    try:
        return enum_type[value]
    except KeyError as exc:
        valid = [item.name for item in enum_type]
        raise ValueError(f"unknown weights {value!r}; expected one of {valid}") from exc


def _check_resolution(
    input_shape: tuple[int, int, int],
    input_resolution: int,
) -> None:
    if len(input_shape) != 3 or input_shape[0] != 3:
        raise ValueError("pretrained models require RGB (3, H, W) input")
    if input_shape[1:] != (input_resolution, input_resolution):
        raise ValueError("input_shape and pretrained.input_resolution do not match")
