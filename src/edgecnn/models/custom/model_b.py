"""Model B - lightweight depthwise-separable CNN.     Assignment Section 2 [20]

Owner: Member 2.

    +---------------------------------------------------------------------+
    |  IN   num_classes, input_shape=(3,64,64), **overrides from           |
    |       configs/stages/models.yaml -> inputs.model_b                   |
    |  OUT  nn.Module, forward -> (B, num_classes) RAW LOGITS              |
    |                                                                      |
    |  HARD CONSTRAINT: <= 100,000 trainable parameters.                   |
    |  tests/test_param_budget.py fails the build if exceeded.             |
    +---------------------------------------------------------------------+

This is the model the whole report is about. It is the subject of the Section 3
optimizer study and the left-hand side of the Section 6 comparison against
MobileNetV2 and SqueezeNet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edgecnn.models.registry import register

if TYPE_CHECKING:
    from torch import nn


@register("model_b")
def build_model_b(
    num_classes: int,
    input_shape: tuple[int, int, int] = (3, 64, 64),
    **overrides: object,
) -> nn.Module:
    """Depthwise-separable CNN under the 100k parameter budget.

    Reference topology (Member 2 owns the final choice)::

        64x64x3
          Conv3x3(16) -> BN -> ReLU6                    ->  64x64x16
          DSConv(32)  -> MaxPool2                       ->  32x32x32
          DSConv(64)  -> MaxPool2                       ->  16x16x64
          DSConv(96)  -> MaxPool2                       ->   8x8x96
          GlobalAvgPool -> Dropout -> FC(num_classes)

    Two things carry the parameter saving, and the report should separate them
    because they are different mechanisms:

    1. **Depthwise separable convolution.** A standard Conv3x3 from C_in to
       C_out costs ``9 * C_in * C_out``. Splitting it into a depthwise
       ``3x3`` (one filter per input channel, ``9 * C_in``) followed by a
       pointwise ``1x1`` (``C_in * C_out``) costs ``9*C_in + C_in*C_out``.
       The ratio is ``1/C_out + 1/9`` - so for C_out=64 it is about an 8x
       reduction in both parameters and MACs.

    2. **Global average pooling instead of flatten.** Model A's
       ``Flatten -> FC(128)`` is ~1.05M weights. GAP replaces it with
       ``96 -> num_classes``, i.e. under a thousand. In a sub-100k budget this
       saving is larger than everything the depthwise convolutions save, and
       saying so is a stronger Section 2 answer than attributing it all to
       the convolutions.

    Activation is ReLU6 rather than ReLU, and Section 2 asks for the
    hardware-aware reason: both are a single ``max()`` with no exponential and
    no FPU requirement, but ReLU6's bounded output range keeps activation
    quantisation well-scaled, which is what actually gets the model onto an
    int8 microcontroller runtime.

    Requirements:
        * ``sum(p.numel() for p in model.parameters() if p.requires_grad) <= 100_000``
        * return RAW LOGITS
        * expose ``.num_classes``
        * work at batch size 1
        * hand-derive the per-layer parameter table for the report
    """
    raise NotImplementedError("Member 2: implement Model B (<=100k trainable params)")
