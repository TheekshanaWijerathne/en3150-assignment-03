# 6. Resource benchmark

The cost side of the comparison: parameters, size on disk, multiply-accumulates, latency and peak
memory of all four models, measured by the same code, in one session, on one machine. The models
are built fresh, so no training or checkpoint is needed.

Latency is the only number here that depends on the machine. It is measured on one CPU thread, as
a small device without a GPU would run the model, so run this notebook on a computer that is
otherwise idle, plugged in if it is a laptop.

**Run time:** about a minute.

**Writes:** `results/benchmark.json`.

<!-- include: common -->
<!-- include: custom_models -->
<!-- include: pretrained -->
<!-- include: macs -->
<!-- include: benchmark -->

## The machine

```python
print("Latency is measured on:", describe_device("cpu"))
```

## Measure every model

```python
benchmark = {}
for name in ("model_a", "model_b", "mobilenet_v2", "squeezenet1_1"):
    benchmark[name] = profile_model(name)
write_json(RESULTS_DIR / "benchmark.json", benchmark)
```

## Results

```python
from IPython.display import Markdown, display

columns = [  # (key, heading, format)
    ("trainable_params", "Trainable parameters", None),
    ("total_params", "Total parameters", None),
    ("model_size_kb", "Size (KB)", ",.1f"),
    ("macs", "MACs", None),
    ("latency_ms", "Latency (ms)", ".2f"),
    ("latency_median_ms", "Median latency (ms)", ".2f"),
    ("peak_memory_mb", "Peak memory (MB)", ".2f"),
]
display(Markdown(markdown_table(
    ["Model"] + [heading for _, heading, _ in columns],
    [[MODEL_LABELS[name]] + [costs[key] for key, _, _ in columns] for name, costs in benchmark.items()],
    caption=f"One image at a time, latency on one CPU thread of {benchmark['model_b']['device']}.",
    formats=[None] + [spec for _, _, spec in columns],
)))
```
