"""Lightweight SOTA backbones, fine-tuned.        Assignment Section 5 [20]

Owner: Member 4.

    +---------------------------------------------------------------------+
    |  IN   num_classes, input_shape=(3,64,64), **overrides from           |
    |       configs/stages/pretrained.yaml -> inputs.backbones[]           |
    |  OUT  nn.Module, forward -> (B, num_classes) RAW LOGITS              |
    |       plus .param_groups() for discriminative learning rates         |
    +---------------------------------------------------------------------+

Both models register into the SAME registry as Model A and Model B. Member 3's
trainer and Member 1's benchmark code therefore need no special case for them,
which is precisely what makes the Section 6 comparison fair: the only thing
that differs between a Model B row and a MobileNetV2 row in the final table is
the architecture.

Backbones are chosen from the list the assignment itself names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edgecnn.models.registry import register

if TYPE_CHECKING:
    from torch import nn


@register("mobilenet_v2")
def build_mobilenet_v2(
    num_classes: int,
    input_shape: tuple[int, int, int] = (3, 64, 64),
    **overrides: object,
) -> nn.Module:
    """MobileNetV2 with an ImageNet backbone and a replaced classifier.

    Steps:

    1. ``torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")``
    2. Freeze per ``finetune_strategy``: ``head_only``, ``last_n_blocks``
       (default, ``unfreeze_last_n: 3``) or ``full``.
    3. Replace ``model.classifier[1]`` - a ``Linear(1280, 1000)`` - with
       ``Linear(1280, num_classes)``. Forgetting this gives a model that
       trains without error and reports nonsense, because the loss is computed
       over 1000 logits of which only 10 are ever correct.
    4. Always leave BatchNorm running statistics trainable in unfrozen blocks;
       EuroSAT's channel statistics are nothing like ImageNet's.

    The 64x64 input is a deliberate choice, not an oversight - see
    ``configs/stages/pretrained.yaml``. MobileNetV2 has a total stride of 32,
    so a 64x64 input reaches the classifier as a 2x2 spatial map where the
    architecture expects 7x7. That is a genuine finding for Section 6: a model
    designed around a resolution cannot be dropped to a quarter of it for free,
    and it explains any accuracy gap better than "pretrained models are
    better/worse" would.

    Must expose ``param_groups()``; see :func:`param_groups`.
    """
    raise NotImplementedError("Member 4: implement MobileNetV2 fine-tuning")


@register("squeezenet1_1")
def build_squeezenet(
    num_classes: int,
    input_shape: tuple[int, int, int] = (3, 64, 64),
    **overrides: object,
) -> nn.Module:
    """SqueezeNet 1.1 with an ImageNet backbone and a replaced classifier.

    Note the structural difference from MobileNetV2, because contrasting the
    two *mechanisms* is what Section 6 rewards - not just their accuracies.
    SqueezeNet reaches parameter efficiency with fire modules: a 1x1 "squeeze"
    that cuts channel count, then a mixed 1x1 / 3x3 "expand". MobileNetV2 uses
    depthwise separables with inverted residuals. Both cut parameters; they
    trade off differently against MACs and against activation memory, which is
    often the real constraint on an MCU.

    SqueezeNet's classifier is a ``Conv2d(512, 1000, 1)`` followed by global
    average pooling - not a Linear layer. Replace ``model.classifier[1]`` with
    ``Conv2d(512, num_classes, kernel_size=1)`` and set
    ``model.num_classes = num_classes``. A ``Linear`` substituted here will
    fail on shape, so this one fails loudly rather than silently.
    """
    raise NotImplementedError("Member 4: implement SqueezeNet fine-tuning")


def param_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
) -> list[dict[str, object]]:
    """Split parameters into backbone and head groups for the optimizer.

    Returns something like::

        [{"params": [...], "lr": lr_backbone, "name": "backbone"},
         {"params": [...], "lr": lr_head,     "name": "head"}]

    Member 3's optimizer factory consumes this when present and falls back to
    a single group otherwise, so custom models need no equivalent.

    The reason for two rates: the backbone already encodes useful features, and
    a head-sized learning rate applied to it destroys them in the first few
    hundred steps - the classic fine-tuning failure where validation accuracy
    peaks at epoch 1 and then falls. The freshly initialised head, by contrast,
    is random and needs a full-sized rate. Worth stating in Section 5.

    Exclude frozen parameters (``requires_grad=False``); passing them to an
    optimizer wastes memory on unused state tensors.
    """
    raise NotImplementedError("Member 4: implement param_groups")
