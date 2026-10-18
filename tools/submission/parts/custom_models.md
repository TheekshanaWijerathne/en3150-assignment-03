## The custom models

**Model A** is a standard CNN: three blocks of 3×3 convolution, batch normalisation, ReLU and 2×2 max
pooling, then a fully connected head on the flattened feature maps. It is not limited in size; it is
the reference point that shows what Model B saves and what it costs.

**Model B** stays under 100,000 trainable parameters through two separate mechanisms:

1. **Depthwise-separable convolutions.** A standard 3×3 convolution from `C_in` to `C_out` channels
   has `9 × C_in × C_out` weights. A depthwise 3×3 convolution (one filter per input channel,
   `9 × C_in`) followed by a pointwise 1×1 convolution (`C_in × C_out`) has `9 × C_in + C_in × C_out`,
   about `1/C_out + 1/9` of the standard cost in both parameters and multiply-accumulates.
2. **Global average pooling instead of flattening.** Model A's flatten into a 128-unit layer alone
   holds 8,192 × 128 weights; Model B averages each feature map to one number, so its classifier
   holds only 320 × 10 weights.

Model B uses ReLU6, a single `min(max(x, 0), 6)`: no exponential and no lookup table, so it costs
almost nothing on a microcontroller, and its bounded output range keeps 8-bit quantisation of the
activations well scaled. Convolutions followed by batch normalisation have no bias, because batch
normalisation re-centres the output straight away.

Both models return raw scores (logits); the softmax is part of the cross-entropy loss.

```python
from torch import nn

MODEL_A_SETTINGS = {
    "blocks": [  # each block: convolution, batch norm, ReLU, then 2x2 max pooling
        {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "pool": 2},
        {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "pool": 2},
        {"out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "pool": 2},
    ],
    "head": {"global_pool": False, "fc_units": [128], "dropout": 0.3},
    "activation": "relu",
    "batch_norm": True,
}

MODEL_B_SETTINGS = {
    "stem": {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
    "blocks": [  # each block: depthwise-separable convolution, then optional 2x2 max pooling
        {"out_channels": 32, "stride": 1, "pool": 2},
        {"out_channels": 64, "stride": 1, "pool": 2},
        {"out_channels": 160, "stride": 1, "pool": 2},
        {"out_channels": 320, "stride": 1},
    ],
    "head": {"global_pool": True, "fc_units": [], "dropout": 0.2},
    "activation": "relu6",
    "batch_norm": True,
}

MODEL_B_PARAMETER_BUDGET = 100_000


def get_activation(name):
    """The activation layer for a name in the settings."""
    activations = {"relu": nn.ReLU, "relu6": nn.ReLU6}
    return activations[name]()


def conv_bn_act(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                activation="relu", batch_norm=True):
    """Convolution, batch normalisation, activation. ``k*k*in*out`` weights."""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                        padding=padding, bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(get_activation(activation))
    return nn.Sequential(*layers)


def depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                             activation="relu6", batch_norm=True):
    """Depthwise k x k convolution, then pointwise 1x1 convolution, each with batch norm and activation.

    ``groups=in_channels`` gives every input channel its own filter, so channels
    are only mixed by the 1x1 convolution. Cost: ``k*k*in + in*out`` weights,
    against ``k*k*in*out`` for a standard convolution.
    """
    layers = [nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                        groups=in_channels, bias=not batch_norm)]                    # depthwise
    if batch_norm:
        layers.append(nn.BatchNorm2d(in_channels))
    layers.append(get_activation(activation))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not batch_norm))  # pointwise
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(get_activation(activation))
    return nn.Sequential(*layers)


def build_head(layers, settings, input_shape, num_classes):
    """Add the classifier to a list of feature layers and return the whole model.

    The number of features entering the classifier is found by passing one
    blank image through the feature layers.
    """
    head = settings["head"]
    if head["global_pool"]:
        layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())

    features = nn.Sequential(*layers)
    features.eval()
    with torch.no_grad():
        in_features = features(torch.zeros(1, *input_shape)).shape[1]
    features.train()

    for units in head["fc_units"]:  # hidden layers: linear, activation, dropout
        layers.append(nn.Linear(in_features, units))
        layers.append(get_activation(settings["activation"]))
        layers.append(nn.Dropout(head["dropout"]))
        in_features = units
    if not head["fc_units"]:
        layers.append(nn.Dropout(head["dropout"]))  # no hidden layer: dropout before the output layer
    layers.append(nn.Linear(in_features, num_classes))  # raw scores, no softmax

    model = nn.Sequential(*layers)
    model.num_classes = num_classes
    return model


def build_model_a(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE, settings=MODEL_A_SETTINGS):
    """Model A: [convolution, batch norm, ReLU, max pool] x 3, then a fully connected head."""
    layers = []
    channels = input_shape[0]
    for block in settings["blocks"]:
        layers.append(conv_bn_act(channels, block["out_channels"], block["kernel_size"],
                                  block["stride"], block["padding"],
                                  settings["activation"], settings["batch_norm"]))
        if block.get("pool"):
            layers.append(nn.MaxPool2d(block["pool"]))
        channels = block["out_channels"]
    return build_head(layers, settings, input_shape, num_classes)


def build_model_b(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE, settings=MODEL_B_SETTINGS):
    """Model B: a standard 3x3 stem, depthwise-separable blocks, global average pooling."""
    stem = settings["stem"]
    layers = [conv_bn_act(input_shape[0], stem["out_channels"], stem["kernel_size"],
                          stem["stride"], stem["padding"],
                          settings["activation"], settings["batch_norm"])]
    channels = stem["out_channels"]
    for block in settings["blocks"]:
        layers.append(depthwise_separable_conv(channels, block["out_channels"],
                                               stride=block["stride"],
                                               activation=settings["activation"],
                                               batch_norm=settings["batch_norm"]))
        if block.get("pool"):
            layers.append(nn.MaxPool2d(block["pool"]))
        channels = block["out_channels"]
    return build_head(layers, settings, input_shape, num_classes)


MODEL_BUILDERS["model_a"] = build_model_a
MODEL_BUILDERS["model_b"] = build_model_b
```
