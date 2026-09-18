"""Model A - standard CNN baseline.                  Assignment Section 2 [20]

Owner: Member 2.

    +---------------------------------------------------------------------+
    |  IN   num_classes, input_shape=(3,64,64), **overrides from           |
    |       configs/stages/models.yaml -> inputs.model_a                   |
    |  OUT  nn.Module, forward -> (B, num_classes) RAW LOGITS              |
    +---------------------------------------------------------------------+

Role in the report: this is the *control*. Model B's parameter and MAC savings
are only meaningful relative to a standard convolutional network trained on
identical data with an identical loop, so Model A is deliberately NOT
parameter-constrained. It shows what depthwise-separable convolutions cost in
accuracy and save in compute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edgecnn.models.registry import register

if TYPE_CHECKING:
    from torch import nn


@register("model_a")
def build_model_a(
    num_classes: int,
    input_shape: tuple[int, int, int] = (3, 64, 64),
    **overrides: object,
) -> nn.Module:
    """Interleaved Conv2d / MaxPool2d stack with a fully connected head.

    Reference topology (Member 2 owns the final choice; the report must
    justify it either way)::

        64x64x3
          Conv3x3(32)  -> BN -> ReLU -> MaxPool2  ->  32x32x32
          Conv3x3(64)  -> BN -> ReLU -> MaxPool2  ->  16x16x64
          Conv3x3(128) -> BN -> ReLU -> MaxPool2  ->   8x8x128
          Flatten(8192) -> Dropout -> FC(128) -> ReLU -> FC(num_classes)

    Watch the flatten. 8*8*128 = 8192 into FC(128) is 1,048,576 weights in a
    single layer - roughly ten times Model B's entire budget. That one line is
    the clearest illustration in the whole report of where parameters actually
    go in a small CNN, and it is worth calling out explicitly in Section 2
    rather than only comparing the totals.

    Requirements:
        * return RAW LOGITS - no softmax (see protocols.ClassifierModel)
        * expose ``.num_classes``
        * work at batch size 1, for Member 1's latency measurement
        * derive per-layer parameter counts BY HAND for the report; Section 2
          asks for the calculation, not a torchsummary dump
    """
    raise NotImplementedError("Member 2: implement Model A")
