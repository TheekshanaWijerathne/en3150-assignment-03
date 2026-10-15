# 4. Training and evaluating the custom models

Model A is trained with Adam, the optimizer chosen in notebook 03, and evaluated on the test split.
It is then set against Model B, whose run comes from notebook 03, so Model B is never trained
twice. The table comparing their size, computation, training time and accuracy is built in
notebook 07.

**Run time:** about 6 minutes on an NVIDIA T4 GPU.

**Writes:** `results/model_a_adam/history.json`, `test_metrics.json`, `curves.png` and
`confusion_matrix.png`, and `checkpoints/model_a_adam/best.pt`.

<!-- include: common -->
<!-- include: data -->
<!-- include: custom_models -->
<!-- include: training -->
<!-- include: evaluation -->

## Data

```python
data = prepare_data()
loaders = make_loaders(data, data.norm_mean, data.norm_std)
print(", ".join(f"{split} {len(data.splits[split]):,}" for split in SPLIT_NAMES)
      + f" images, in batches of {BATCH_SIZE}")
```

## Build Model A

```python
model = build_model("model_a")
print(f"{count_parameters(model):,} trainable parameters")
```

## Train

```python
history_a = train(model, "model_a", "adam", "model_a_adam", loaders)
```

## Loss curves

The gap between the training and validation curves is the overfitting signal; the dotted line marks
the checkpoint that is evaluated.

```python
plot_loss_curves(history_a)
```

## Test-set evaluation

```python
metrics_a = evaluate_on_test("model_a", "model_a_adam", loaders, data.class_names)
```

## Confusion matrix

Normalised by true class, so the diagonal reads as per-class recall.

```python
plot_confusion_matrix(metrics_a, "Model A (Adam)")
```

## Model B, from notebook 03

Read from the result files of notebook 03.

```python
run_b = RESULTS_DIR / "model_b_adam"
if (run_b / "test_metrics.json").exists():
    history_b = read_json(run_b / "history.json")
    metrics_b = read_json(run_b / "test_metrics.json")
    plot_loss_curves(history_b)
    plot_confusion_matrix(metrics_b, "Model B (Adam)")
else:
    metrics_b = None
    print("Model B's results are missing: run notebook 03 first.")
```

## Side by side

```python
if metrics_b is not None:
    for name, metrics in (("Model A", metrics_a), ("Model B", metrics_b)):
        print(f"{name}: accuracy {metrics['accuracy']:.4f} | macro precision "
              f"{metrics['macro_precision']:.4f} | macro recall {metrics['macro_recall']:.4f}")
```
