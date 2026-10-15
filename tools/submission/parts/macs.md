## Counting multiply-accumulates

A multiply-accumulate (MAC) is one multiplication and one addition; one image's forward pass is
counted at batch size 1. Only convolutions and fully connected layers are counted, the convention
of the MobileNet and SqueezeNet papers and of torchvision's published figures (MobileNetV2 0.301 G
and SqueezeNet 1.1 0.349 G MACs at 224×224, which this count reproduces). Batch normalisation,
activations, pooling and bias additions are cheap next to the convolutions and are left out.

- a convolution costs (output values) × (input channels / groups) × (kernel area), so a depthwise
  convolution, with one group per channel, costs a fraction of a standard one;
- a fully connected layer costs (output values) × (input features).

```python
import copy
import math

from torch import nn


def count_macs(model, input_shape=INPUT_SHAPE):
    """Multiply-accumulates of convolutions and fully connected layers, for one image.

    Measured on a copy in evaluation mode, so the model itself is untouched.
    """
    probe = copy.deepcopy(model).cpu().eval()
    total = 0

    def conv(module, inputs, output):
        nonlocal total
        total += output[0].numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)

    def linear(module, inputs, output):
        nonlocal total
        total += output[0].numel() * module.in_features

    hooks = []
    for module in probe.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear))
    try:
        with torch.no_grad():
            probe(torch.zeros(1, *input_shape))
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)
```
