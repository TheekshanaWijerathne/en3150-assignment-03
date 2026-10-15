## The pretrained models

MobileNetV2 (Sandler et al., 2018) and SqueezeNet 1.1 (Iandola et al., 2016) come with weights
trained on ImageNet. They reach efficiency in two different ways: MobileNetV2 uses inverted
residual blocks built from depthwise-separable convolutions, while SqueezeNet's fire modules first
"squeeze" the channels with 1×1 convolutions, then expand them again.

Fine-tuning strategy, the same for both:

- the ImageNet classifier is replaced by a new 10-class layer, initialised with small random weights;
- only the new classifier and the last few feature blocks are trained (3 blocks of MobileNetV2,
  2 fire modules of SqueezeNet); the earlier blocks keep their ImageNet features frozen;
- the trained feature blocks learn at a tenth of the classifier's learning rate, so the pretrained
  features are adjusted gently rather than overwritten.

Both models are fed the same 64×64 images as the custom models, not upsampled to their native
224×224. This keeps the cost comparison honest, at the resolution the sensor actually produces,
but it handicaps them: MobileNetV2's total stride of 32 leaves a 2×2 feature map. Their inputs are
normalised with the ImageNet statistics their weights were trained with.

```python
from torch import nn
from torchvision.models import (
    MobileNet_V2_Weights,
    SqueezeNet1_1_Weights,
    mobilenet_v2,
    squeezenet1_1,
)

# Keep the downloaded ImageNet weights inside this folder, with the dataset.
torch.hub.set_dir(str(DATA_DIR / "torch_hub"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PRETRAINED_SETTINGS = {
    "mobilenet_v2": {"unfreeze_last_n": 3, "dropout": 0.2, "backbone_lr_scale": 0.1},
    "squeezenet1_1": {"unfreeze_last_n": 2, "dropout": 0.5, "backbone_lr_scale": 0.1},
}


def replace_classifier(model, num_classes, dropout):
    """Swap the 1000-class ImageNet output layer for a ``num_classes`` one.

    MobileNetV2 ends in a linear layer; SqueezeNet in a 1x1 convolution followed
    by global average pooling. The new layer starts from small random weights
    (standard deviation 0.01) and zero biases.
    """
    old_head = model.classifier[1]
    model.classifier[0] = nn.Dropout(p=dropout)
    if isinstance(old_head, nn.Linear):          # MobileNetV2
        new_head = nn.Linear(old_head.in_features, num_classes)
    else:                                        # SqueezeNet
        new_head = nn.Conv2d(old_head.in_channels, num_classes, kernel_size=1)
    nn.init.normal_(new_head.weight, mean=0.0, std=0.01)
    nn.init.zeros_(new_head.bias)
    model.classifier[1] = new_head
    model.num_classes = num_classes
    return model


def unfreeze_last_blocks(model, count):
    """Freeze every weight, then unfreeze the classifier and the last ``count`` feature blocks."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    for block in list(model.features.children())[-count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    return model


def build_mobilenet_v2(num_classes=NUM_CLASSES):
    settings = PRETRAINED_SETTINGS["mobilenet_v2"]
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    replace_classifier(model, num_classes, settings["dropout"])
    return unfreeze_last_blocks(model, settings["unfreeze_last_n"])


def build_squeezenet1_1(num_classes=NUM_CLASSES):
    settings = PRETRAINED_SETTINGS["squeezenet1_1"]
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    replace_classifier(model, num_classes, settings["dropout"])
    return unfreeze_last_blocks(model, settings["unfreeze_last_n"])


MODEL_BUILDERS["mobilenet_v2"] = build_mobilenet_v2
MODEL_BUILDERS["squeezenet1_1"] = build_squeezenet1_1
```
