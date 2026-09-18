"""Freezing and unfreezing strategies for transfer learning.

Owner: Member 4.  Section 5.

Which layers are trainable is the main experimental knob in fine-tuning, and
it directly determines the `trainable_params` vs `total_params` gap that the
Section 5 table reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

#: Strategies selectable from configs/stages/pretrained.yaml.
FINETUNE_STRATEGIES: tuple[str, ...] = ("head_only", "last_n_blocks", "full")


def apply_finetune_strategy(
    model: nn.Module,
    strategy: str,
    unfreeze_last_n: int = 0,
) -> nn.Module:
    """Set `requires_grad` across the model according to `strategy`.

    * ``head_only``      - only the replaced classifier trains. Fastest, lowest
      memory, and usually the weakest here: ImageNet features transfer poorly
      to overhead satellite imagery, whose texture statistics are unlike
      object-centric photographs.
    * ``last_n_blocks``  - classifier plus the final ``n`` feature blocks. The
      default, and normally the best accuracy-per-epoch trade-off.
    * ``full``           - everything trains. Best ceiling, most compute, and
      most prone to overfitting on a dataset this size.

    Whichever is used, report it in Section 5 alongside both parameter counts.
    A reader cannot interpret "MobileNetV2: 2.2M parameters" without knowing
    how many of them actually moved.

    Raises:
        ValueError: on an unknown strategy - a typo must not silently fall
            through to training everything, which would quietly change both
            the epoch time and the accuracy being reported.
    """
    raise NotImplementedError("Member 4: implement apply_finetune_strategy")


def replace_classifier(model: nn.Module, num_classes: int, dropout: float = 0.2) -> nn.Module:
    """Swap the ImageNet 1000-way head for a `num_classes`-way one.

    Architecture-specific, so dispatch on type:

    * MobileNetV2  -> ``classifier[1]`` is ``Linear(1280, 1000)``
    * SqueezeNet   -> ``classifier[1]`` is ``Conv2d(512, 1000, 1)``, not Linear

    Initialise the new head sensibly (Kaiming or the torchvision default) and
    return raw logits - no softmax, exactly like the custom models.
    """
    raise NotImplementedError("Member 4: implement replace_classifier")


def count_trainable_vs_total(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)``.

    These differ whenever anything is frozen, and `resources.schema.json`
    requires both. Useful while iterating; the authoritative numbers in the
    report come from Member 1's ``profile_model`` so every model is counted by
    one code path.
    """
    raise NotImplementedError("Member 4: implement count_trainable_vs_total")
