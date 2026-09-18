"""Pretrained lightweight backbones - Assignment Section 5 [20 marks].

Owner: Member 4.

Importing this module registers ``mobilenet_v2`` and ``squeezenet1_1`` into the
shared registry, alongside Member 2's custom models. Same registry, same
builder signature, same logits contract - so Member 3's trainer and Member 1's
benchmark treat all four models identically.
"""

from edgecnn.models.pretrained import backbones  # noqa: F401  (registration side effect)
from edgecnn.models.pretrained.backbones import param_groups
from edgecnn.models.pretrained.finetune import (
    FINETUNE_STRATEGIES,
    apply_finetune_strategy,
    count_trainable_vs_total,
    replace_classifier,
)

__all__ = [
    "FINETUNE_STRATEGIES",
    "apply_finetune_strategy",
    "count_trainable_vs_total",
    "param_groups",
    "replace_classifier",
]
