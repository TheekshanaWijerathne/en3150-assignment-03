## Building the comparison tables

The tables are built only from the saved result files, so they can be rebuilt without training
anything:

- `results/<run>/history.json`: each run's per-epoch record, written by notebooks 03 to 05;
- `results/<run>/test_metrics.json`: each run's test scores, written by the same notebooks;
- `results/benchmark.json`: each model's costs, written by notebook 06.

Mean epoch times come from the training runs, on the GPU; latency from notebook 06, on one CPU thread.

```python
from IPython.display import Markdown, display
from matplotlib.ticker import EngFormatter

RUNS = {  # run name -> (model, optimizer)
    "model_b_sgd": ("model_b", "sgd"),
    "model_b_sgd_momentum": ("model_b", "sgd_momentum"),
    "model_b_adam": ("model_b", "adam"),
    "model_a_adam": ("model_a", "adam"),
    "mobilenet_v2_adam": ("mobilenet_v2", "adam"),
    "squeezenet1_1_adam": ("squeezenet1_1", "adam"),
}


def history_of(run):
    return read_json(RESULTS_DIR / run / "history.json")


def metrics_of(run):
    return read_json(RESULTS_DIR / run / "test_metrics.json")


def mean_epoch_time(history):
    return sum(row["epoch_time_s"] for row in history["epochs"]) / len(history["epochs"])


def optimizer_table():
    """Section 3: Model B with each optimizer, convergence and final performance."""
    rows = []
    for run in ("model_b_sgd", "model_b_sgd_momentum", "model_b_adam"):
        history, metrics = history_of(run), metrics_of(run)
        best = history["best_val_acc"]
        reached = next(row["epoch"] for row in history["epochs"] if row["val_acc"] >= 0.9 * best)
        rows.append([OPTIMIZER_LABELS[history["optimizer"]], history["hyperparameters"]["lr"],
                     history["best_epoch"], best * 100, metrics["accuracy"] * 100, reached,
                     mean_epoch_time(history)])
    return markdown_table(
        ["Optimizer", "Learning rate", "Best epoch", "Best validation accuracy (%)",
         "Test accuracy (%)", "Epoch to 90% of best", "Mean epoch time (s)"],
        rows, "Model B optimizer comparison.", [None, None, None, ".2f", ".2f", None, ".1f"])


def custom_model_table(benchmark):
    """Section 4: Model A against Model B."""
    rows = []
    for run in ("model_a_adam", "model_b_adam"):
        model, _ = RUNS[run]
        costs, metrics = benchmark[model], metrics_of(run)
        rows.append([MODEL_LABELS[model], costs["total_params"], costs["model_size_kb"],
                     costs["macs"], mean_epoch_time(history_of(run)), metrics["accuracy"] * 100,
                     metrics["macro_precision"] * 100, metrics["macro_recall"] * 100])
    return markdown_table(
        ["Model", "Total parameters", "Model size (KB)", "MACs", "Mean epoch time (s)",
         "Test accuracy (%)", "Macro precision (%)", "Macro recall (%)"],
        rows, "Model A and Model B comparison.",
        [None, None, ",.1f", None, ".1f", ".2f", ".2f", ".2f"])


def final_table(benchmark):
    """Section 6: Model B against the two pretrained models."""
    rows = []
    for run in ("model_b_adam", "mobilenet_v2_adam", "squeezenet1_1_adam"):
        model, _ = RUNS[run]
        costs, metrics = benchmark[model], metrics_of(run)
        rows.append([MODEL_LABELS[model], costs["trainable_params"], costs["total_params"],
                     costs["model_size_kb"], costs["model_size_kb"] / 1024, costs["macs"],
                     costs["latency_ms"], costs["peak_memory_mb"], metrics["accuracy"] * 100,
                     metrics["macro_precision"] * 100, metrics["macro_recall"] * 100])
    return markdown_table(
        ["Model", "Trainable parameters", "Total parameters", "Size (KB)", "Size (MB)", "MACs",
         "CPU latency (ms)", "Peak memory (MB)", "Test accuracy (%)", "Macro precision (%)",
         "Macro recall (%)"],
        rows, "Model B versus pretrained lightweight models.",
        [None, None, None, ",.1f", ".2f", None, ".2f", ".2f", ".2f", ".2f", ".2f"])
```

The trade-off in pictures. Parameters stand for memory: a device stores every weight, frozen or
not, so the total count is used. MACs stand for computation. The dashed line joins the models that
no other model beats on both cost and accuracy (the Pareto frontier); it is drawn only when more
than one model is on it.

```python
def plot_tradeoff(benchmark, cost):
    """Test accuracy against ``total_params`` or ``macs`` for Model B and the pretrained models."""
    points = []
    for run in ("model_b_adam", "mobilenet_v2_adam", "squeezenet1_1_adam"):
        model, _ = RUNS[run]
        points.append((model, float(benchmark[model][cost]), metrics_of(run)["accuracy"] * 100))

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, value, accuracy in points:
        ax.scatter(value, accuracy, color=MODEL_COLORS[model], s=70)
        ax.annotate(MODEL_LABELS[model], (value, accuracy), xytext=(6, 6), textcoords="offset points")

    frontier, best = [], float("-inf")
    for point in sorted(points, key=lambda p: p[1]):
        if point[2] > best:
            frontier.append(point)
            best = point[2]
    if len(frontier) > 1:
        ax.plot([p[1] for p in frontier], [p[2] for p in frontier], linestyle="--", color="0.35",
                label="Pareto frontier")
        ax.legend()

    label, name = {"total_params": ("Total parameters", "parameters"),
                   "macs": ("MACs per image", "MACs")}[cost]
    values = [p[1] for p in points]
    log_scale = max(values) > 10 * min(values)  # a log axis only when the costs span over a decade
    if log_scale:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(EngFormatter(sep=""))  # 100k, 1M, 14M
    ax.margins(x=0.2, y=0.15)  # room for the labels beside the outermost points
    ax.set_xlabel(f"{label} (log scale)" if log_scale else label)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"Test accuracy versus {name}")
    fig.tight_layout()
    file_name = "accuracy_vs_macs.png" if cost == "macs" else "accuracy_vs_params.png"
    save_figure(fig, RESULTS_DIR / "figures" / file_name)
    plt.show()
```
