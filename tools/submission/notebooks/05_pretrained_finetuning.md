# 5. Fine-tuning pretrained models

MobileNetV2 and SqueezeNet 1.1, pretrained on ImageNet, are fine-tuned on the same split, at the
same 64×64 resolution, through the same training loop as the custom models, and evaluated on the
same test images.

**Run time:** about 12 minutes on an NVIDIA T4 GPU.

**Writes:** for each run (`mobilenet_v2_adam`, `squeezenet1_1_adam`): `results/<run>/history.json`,
`test_metrics.json`, `curves.png` and `confusion_matrix.png`, and `checkpoints/<run>/best.pt`.

<!-- include: common -->
<!-- include: data -->
<!-- include: pretrained -->
<!-- include: training -->
<!-- include: evaluation -->

## Data

Normalised with the ImageNet statistics. One set of data loaders serves both models, as in the
reported runs.

```python
data = prepare_data()
loaders = make_loaders(data, IMAGENET_MEAN, IMAGENET_STD)
print(", ".join(f"{split} {len(data.splits[split]):,}" for split in SPLIT_NAMES)
      + f" images, in batches of {BATCH_SIZE}")
histories, test_results = {}, {}
```

## 1. MobileNetV2

```python
model = build_model("mobilenet_v2")
print(f"{count_parameters(model):,} trainable of {count_parameters(model, trainable_only=False):,} "
      f"parameters | {model_size_kb(model) / 1024:.2f} MB")
```

```python
histories["mobilenet_v2"] = train(
    model, "mobilenet_v2", "adam", "mobilenet_v2_adam", loaders,
    backbone_lr_scale=PRETRAINED_SETTINGS["mobilenet_v2"]["backbone_lr_scale"],
)
```

```python
plot_loss_curves(histories["mobilenet_v2"])
```

```python
test_results["mobilenet_v2"] = evaluate_on_test("mobilenet_v2", "mobilenet_v2_adam", loaders,
                                                data.class_names)
```

```python
plot_confusion_matrix(test_results["mobilenet_v2"], "MobileNetV2 (Adam)")
```

## 2. SqueezeNet 1.1

```python
model = build_model("squeezenet1_1")
print(f"{count_parameters(model):,} trainable of {count_parameters(model, trainable_only=False):,} "
      f"parameters | {model_size_kb(model) / 1024:.2f} MB")
```

```python
histories["squeezenet1_1"] = train(
    model, "squeezenet1_1", "adam", "squeezenet1_1_adam", loaders,
    backbone_lr_scale=PRETRAINED_SETTINGS["squeezenet1_1"]["backbone_lr_scale"],
)
```

```python
plot_loss_curves(histories["squeezenet1_1"])
```

```python
test_results["squeezenet1_1"] = evaluate_on_test("squeezenet1_1", "squeezenet1_1_adam", loaders,
                                                 data.class_names)
```

```python
plot_confusion_matrix(test_results["squeezenet1_1"], "SqueezeNet 1.1 (Adam)")
```
