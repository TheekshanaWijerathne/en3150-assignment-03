# 2. Custom architectures

This notebook defines the two custom models, checks their outputs, derives every layer's parameter
count and counts the multiply-accumulates of one image. Nothing is trained here: Model B is trained
in notebook 03 and Model A in notebook 04.

**Writes:** `results/tables/custom_architectures.md`, `results/tables/param_breakdown.md`.

<!-- include: common -->
<!-- include: custom_models -->
<!-- include: macs -->
<!-- include: architecture_tables -->

## Build both models

```python
from IPython.display import Markdown, display

model_a = build_model("model_a")
model_b = build_model("model_b")
model_b
```

## Output check

Both models must return one raw score per class at any batch size. Rows summing to 1 would mean a
softmax had been left inside the model.

```python
for name, model in (("Model A", model_a), ("Model B", model_b)):
    model.eval()
    with torch.no_grad():
        for batch in (1, 2):
            scores = model(torch.randn(batch, *INPUT_SHAPE))
            softmax_inside = torch.allclose(scores.sum(dim=1), torch.ones(batch), atol=1e-3)
            print(f"{name}, batch of {batch}: output {tuple(scores.shape)}, softmax inside: {softmax_inside}")
```

## Model A, layer by layer

Most of Model A's parameters sit in one layer: the flattened 8×8×128 feature maps feed a
128-unit fully connected layer.

```python
layer_table(model_a)
```

## Model B, layer by layer

```python
layer_table(model_b)
```

## Parameters and multiply-accumulates

```python
for name, model in (("Model A", model_a), ("Model B", model_b)):
    print(f"{name}: {count_parameters(model):,} trainable parameters, "
          f"{count_macs(model):,} MACs per image")
assert count_parameters(model_b) <= MODEL_B_PARAMETER_BUDGET, "Model B is over its parameter budget"
print(f"Model B uses {count_parameters(model_b) / MODEL_B_PARAMETER_BUDGET:.0%} "
      f"of its {MODEL_B_PARAMETER_BUDGET:,}-parameter budget.")
```

## The Section 2 tables

```python
custom, breakdown = architecture_tables(model_a, model_b)
display(Markdown(custom))
```
