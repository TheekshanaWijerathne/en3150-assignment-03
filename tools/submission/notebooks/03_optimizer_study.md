# 3. Optimizer study

Model B is trained three times: with plain SGD, with SGD and momentum 0.9, and with Adam. The runs
are compared on convergence speed and on final accuracy. Adam is the optimizer used for Model B in
the rest of the study.

- **Learning rates.** Adam's steps are normalised per parameter, so it uses 0.001. SGD steps in
  proportion to the raw gradient and gets 0.01; at 0.001 it would be unfairly slow. Both SGD runs
  use the same rate, so only the momentum term differs between them.
- **Why Adam for Model B.** A depthwise kernel holds 9 weights while a pointwise kernel holds
  hundreds, so their gradients differ in scale by orders of magnitude. A single global SGD learning
  rate cannot suit both; Adam adapts the step size of every parameter.

**Run time:** about 6 minutes per run on an NVIDIA T4 GPU, about 20 minutes in all. A CPU takes
hours.

**Writes:** for each run (`model_b_sgd`, `model_b_sgd_momentum`, `model_b_adam`):
`results/<run>/history.json`, `test_metrics.json` and `curves.png`, and `checkpoints/<run>/best.pt`.
Also `results/figures/optimizer_overlay.png` and `results/model_b_adam/confusion_matrix.png`.

<!-- include: common -->
<!-- include: data -->
<!-- include: custom_models -->
<!-- include: training -->
<!-- include: evaluation -->
<!-- include: optimizer_study -->

## Data

The three runs start from identical initial weights, with the same data, learning-rate schedule
and random seed. They share one set of data loaders, as in the reported runs, so the training
loader continues its shuffling from one run to the next and each run sees its batches in a
different order.

```python
data = prepare_data()
loaders = make_loaders(data, data.norm_mean, data.norm_std)
print(", ".join(f"{split} {len(data.splits[split]):,}" for split in SPLIT_NAMES)
      + f" images, in batches of {BATCH_SIZE}")
histories, test_results = {}, {}
```

## 1. Plain SGD

```python
model = build_model("model_b")
histories["sgd"] = train(model, "model_b", "sgd", "model_b_sgd", loaders)
```

```python
plot_loss_curves(histories["sgd"])
```

```python
test_results["sgd"] = evaluate_on_test("model_b", "model_b_sgd", loaders, data.class_names)
```

## 2. SGD with momentum

Only the momentum term differs from the run above.

```python
model = build_model("model_b")
histories["sgd_momentum"] = train(model, "model_b", "sgd_momentum", "model_b_sgd_momentum", loaders)
```

```python
plot_loss_curves(histories["sgd_momentum"])
```

```python
test_results["sgd_momentum"] = evaluate_on_test("model_b", "model_b_sgd_momentum", loaders,
                                                data.class_names)
```

## 3. Adam

This run is Model B's result for the rest of the study.

```python
model = build_model("model_b")
histories["adam"] = train(model, "model_b", "adam", "model_b_adam", loaders)
```

```python
plot_loss_curves(histories["adam"])
```

```python
test_results["adam"] = evaluate_on_test("model_b", "model_b_adam", loaders, data.class_names)
```

## All three on one axis

```python
order = ("sgd", "sgd_momentum", "adam")
plot_optimizer_overlay([histories[name] for name in order])
```

## Summary

```python
summary = summarize_study([histories[name] for name in order])
summary["test accuracy"] = [test_results[name]["accuracy"] for name in order]
summary
```

## Confusion matrix of Model B with Adam

Normalised by true class, so the diagonal reads as per-class recall.

```python
plot_confusion_matrix(test_results["adam"], "Model B (Adam)")
```
