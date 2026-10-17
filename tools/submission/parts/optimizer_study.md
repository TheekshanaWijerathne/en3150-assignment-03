## Comparing the optimizers

The three runs differ only in the optimizer: the same initial weights, the same batches in the same
order, the same augmentation, learning-rate schedule and number of epochs. Two questions are
answered separately, because they can disagree:

- **convergence speed:** the first epoch at which validation accuracy reaches 90% of that run's own
  best. The x axis below is logarithmic, which spreads out the early epochs where the optimizers
  differ most;
- **final performance:** the best validation accuracy, and the test accuracy of that checkpoint.

```python
import pandas as pd


def plot_optimizer_overlay(histories):
    """Validation loss and accuracy of every optimizer on shared axes; dots mark each best epoch."""
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for history in histories:
        epochs = history["epochs"]
        name = history["optimizer"]
        style = {"color": OPTIMIZER_COLORS[name]}
        x = [row["epoch"] for row in epochs]
        best = next(row for row in epochs if row["epoch"] == history["best_epoch"])
        loss_ax.plot(x, [row["val_loss"] for row in epochs], label=OPTIMIZER_LABELS[name], **style)
        acc_ax.plot(x, [row["val_acc"] for row in epochs], label=OPTIMIZER_LABELS[name], **style)
        loss_ax.scatter([best["epoch"]], [best["val_loss"]], marker="o", s=28, zorder=3, **style)
        acc_ax.scatter([best["epoch"]], [best["val_acc"]], marker="o", s=28, zorder=3, **style)
    for ax in (loss_ax, acc_ax):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Epoch (log scale)")
        ax.legend()
    loss_ax.set(title="Validation loss", ylabel="Cross-entropy loss")
    acc_ax.set(title="Validation accuracy", ylabel="Accuracy", ylim=(0.0, 1.0))
    fig.suptitle("Model B: optimizer comparison")
    save_figure(fig, RESULTS_DIR / "figures" / "optimizer_overlay.png")
    plt.show()


def summarize_study(histories):
    """Convergence and final performance of each run, side by side."""
    rows = {}
    for history in histories:
        epochs = history["epochs"]
        best = max(epochs, key=lambda row: row["val_acc"])
        reached = next(row["epoch"] for row in epochs if row["val_acc"] >= 0.9 * best["val_acc"])
        last = epochs[-1]
        rows[OPTIMIZER_LABELS[history["optimizer"]]] = {
            "best val accuracy": best["val_acc"],
            "best epoch": best["epoch"],
            "epochs to 90% of best": reached,
            "final train accuracy": last["train_acc"],
            "final val accuracy": last["val_acc"],
            "final train - val gap": last["train_acc"] - last["val_acc"],
            "mean epoch time (s)": sum(row["epoch_time_s"] for row in epochs) / len(epochs),
        }
    # One row per run, so that each column keeps its own type: epochs stay whole numbers.
    return pd.DataFrame.from_dict(rows, orient="index")
```
