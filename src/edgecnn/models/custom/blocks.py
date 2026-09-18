"""Reusable building blocks for the custom models.

Owner: Member 2.

Keeping the depthwise-separable block here rather than inline in model_b.py
means the parameter arithmetic is written down once, in the place the report
quotes it from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def conv_bn_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    activation: str = "relu",
    batch_norm: bool = True,
) -> nn.Sequential:
    """Standard Conv2d -> BatchNorm2d -> activation. Used by Model A.

    Parameter count: ``kernel_size^2 * in_channels * out_channels`` weights,
    plus ``out_channels`` biases when batch norm is absent. With batch norm
    the conv bias is redundant (BN re-centres immediately after), so set
    ``bias=False`` on the conv - it is a small saving but the report should
    show it is understood rather than accidental.
    """
    raise NotImplementedError("Member 2: implement conv_bn_act")


def depthwise_separable_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    activation: str = "relu6",
    batch_norm: bool = True,
) -> nn.Sequential:
    """Depthwise 3x3 then pointwise 1x1. The core of Model B.

    Structure::

        Conv2d(in, in, k, groups=in, bias=False)   <- depthwise: groups=in is
        BatchNorm2d(in)                               what makes it depthwise
        activation
        Conv2d(in, out, 1, bias=False)             <- pointwise: mixes channels
        BatchNorm2d(out)
        activation

    Cost, for the Section 2 derivation::

        standard  : k*k * in * out
        separable : k*k * in  +  in * out
        ratio     : 1/out + 1/(k*k)

    At k=3, out=64 that is 1/64 + 1/9 ~= 0.127, so roughly an 8x reduction in
    both parameters and MACs for that layer.

    The ``groups=in_channels`` argument is the entire trick: it gives each
    input channel its own filter, so no cross-channel mixing happens until the
    1x1. Get this wrong (``groups=1``) and the model still trains - it is just
    silently a normal convolution with far more parameters, which is exactly
    the kind of error the 100k budget test catches.
    """
    raise NotImplementedError("Member 2: implement depthwise_separable_conv")


def get_activation(name: str) -> nn.Module:
    """Map a config activation name to a module.

    Supported: ``relu``, ``relu6``, ``leaky_relu``, ``hardswish``.

    Section 2 asks for hardware-aware justification, so the argument to make
    in the report is roughly: relu and relu6 are a single max() - no
    exponential, no lookup table, no FPU needed. sigmoid, tanh, ELU, GELU and
    Swish each cost a transcendental per activation per inference, which
    dominates runtime on a microcontroller. hardswish is the interesting
    middle case: it approximates Swish with only add/multiply/clamp, which is
    why MobileNetV3 uses it on mobile silicon - worth a sentence contrasting
    it with what Model B chooses.
    """
    raise NotImplementedError("Member 2: implement get_activation")


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters.

    Convenience for Member 2's own iteration. The authoritative number that
    reaches the report comes from Member 1's
    ``edgecnn.evaluation.benchmark.profile_model`` so every model is counted
    identically - do not put this number in the report directly.
    """
    raise NotImplementedError("Member 2: implement count_parameters")
