"""Model A - standard CNN baseline.                  Assignment Section 2 [20]

Owner: Member 2.   Exercised in notebooks/02_custom_architectures.ipynb,
trained in notebooks/04_custom_training_evaluation.ipynb.

    +---------------------------------------------------------------------+
    |  IN   num_classes, input_shape=(3,64,64), and as **overrides the     |
    |       keys of configs/stages/models.yaml -> model_a                  |
    |       (blocks, head, activation, batch_norm)                         |
    |  OUT  nn.Module, forward -> (B, num_classes) RAW LOGITS              |
    +---------------------------------------------------------------------+

Call it through ``build_model_from_config(cfg, num_classes)``, which passes the
``model_a`` section as the overrides - never assemble them by hand.

Role in the report: this is the *control*. Model B's parameter and MAC savings
are only meaningful relative to a standard convolutional network trained on
identical data with an identical loop, so Model A is deliberately NOT
parameter-constrained. It shows what depthwise-separable convolutions cost in
accuracy and save in compute.
"""

from __future__ import annotations

import torch
from torch import nn

from edgecnn.config.loader import load_stage
from edgecnn.models.custom.blocks import conv_bn_act, get_activation
from edgecnn.models.registry import register


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
    # Settings: the model_a section of models.yaml, replaced by any overrides given.
    settings = {**load_stage("models")["model_a"], **overrides}
    blocks = settings["blocks"]
    head = settings["head"]
    activation = settings["activation"]
    batch_norm = settings["batch_norm"]

    # Conv part: [Conv -> BatchNorm -> activation -> MaxPool] for every block.
    layers: list[nn.Module] = []
    channels = input_shape[0]
    for block in blocks:
        conv_args = {k: block[k] for k in ("kernel_size", "stride", "padding") if k in block}
        layers.append(
            conv_bn_act(
                channels,
                block["out_channels"],
                activation=activation,
                batch_norm=batch_norm,
                **conv_args,
            )
        )
        if block.get("pool"):
            layers.append(nn.MaxPool2d(block["pool"]))
        channels = block["out_channels"]

    # Turn the feature maps into one flat list of numbers.
    if head["global_pool"]:
        layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())

    # How many numbers come out of the Conv part? Find out with one dummy picture.
    features = nn.Sequential(*layers)
    features.eval()
    with torch.no_grad():
        in_features = features(torch.zeros(1, *input_shape)).shape[1]
    features.train()

    # FC part: [FC -> activation -> Dropout] for every hidden layer, then the final FC.
    for units in head["fc_units"]:
        layers.append(nn.Linear(in_features, units))
        layers.append(get_activation(activation))
        layers.append(nn.Dropout(head["dropout"]))
        in_features = units
    if not head["fc_units"]:
        layers.append(nn.Dropout(head["dropout"]))  # no hidden layer: Dropout before the final FC
    layers.append(nn.Linear(in_features, num_classes))  # raw scores, no softmax

    model = nn.Sequential(*layers)
    model.num_classes = num_classes
    return model
