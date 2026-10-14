## Evaluation

Every model is scored on the test split by the same functions, right after training, from its best
checkpoint:

- **accuracy**, and **macro-averaged precision, recall and F1**: every class counts equally.
  Micro-averaged precision would equal the accuracy for single-label classification;
- a class that is never predicted gets precision 0 rather than an error;
- the **confusion matrix** counts, in row `i` and column `j`, the test images of true class `i`
  predicted as class `j`. Its figure is normalised by true class, so the diagonal reads as
  per-class recall.

```python
import textwrap

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def predict(model, loader):
    """True and predicted classes for every image of ``loader``, in its order."""
    model.eval()
    truths, predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE, non_blocking=True))
            predictions.append(logits.argmax(dim=1).cpu())
            truths.append(torch.as_tensor(labels).cpu())
    return torch.cat(truths).numpy(), torch.cat(predictions).numpy()


def compute_metrics(y_true, y_pred, class_names):
    """Accuracy, macro and weighted precision / recall, per-class scores and the confusion matrix."""
    labels = list(range(len(class_names)))  # explicit, so a class that is never predicted keeps its row
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]),
               "f1": float(f1[i]), "support": int(support[i])}
        for i, name in enumerate(class_names)
    }
    total = int(support.sum())

    def weighted(key):  # each class's score weighted by its number of test images
        value = sum(scores[key] * scores["support"] for scores in per_class.values()) / total
        return min(1.0, value)  # guards a rounding error just above 1

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": weighted("precision"),
        "weighted_recall": weighted("recall"),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": list(class_names),
        "num_samples": total,
    }


def evaluate_on_test(model_name, run_name, loaders, class_names):
    """Score a run's best checkpoint on the test split; save ``results/<run_name>/test_metrics.json``."""
    checkpoint = torch.load(CHECKPOINT_DIR / run_name / "best.pt", map_location=DEVICE,
                            weights_only=False)
    model = build_model(model_name, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    y_true, y_pred = predict(model, loaders["test"])
    metrics = {"run": run_name, "model": model_name, "split": "test",
               "checkpoint_epoch": int(checkpoint["epoch"]),
               **compute_metrics(y_true, y_pred, class_names)}
    write_json(RESULTS_DIR / run_name / "test_metrics.json", metrics)
    print(f"Test accuracy {metrics['accuracy']:.4f} | macro precision {metrics['macro_precision']:.4f}"
          f" | macro recall {metrics['macro_recall']:.4f} | {metrics['num_samples']:,} images")
    return metrics
```

The loss curves show training and validation on the same axes: the gap between them is the
overfitting signal. The dotted line marks the epoch whose checkpoint was evaluated.

```python
def run_title(history):
    """A readable run name, such as ``Model B (Adam)``."""
    return f"{MODEL_LABELS[history['model']]} ({OPTIMIZER_LABELS[history['optimizer']]})"


def plot_loss_curves(history):
    """Training and validation loss and accuracy per epoch; saved as ``curves.png``."""
    epochs = history["epochs"]
    x = [row["epoch"] for row in epochs]
    best = history["best_epoch"]
    color = MODEL_COLORS[history["model"]]
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, title, label in ((loss_ax, "loss", "Loss", "Cross-entropy loss"),
                                  (acc_ax, "acc", "Accuracy", "Accuracy")):
        ax.plot(x, [row[f"train_{key}"] for row in epochs], label="Training", color=color)
        ax.plot(x, [row[f"val_{key}"] for row in epochs], label="Validation", color=color,
                linestyle="--")
        ax.axvline(best, color="0.35", linestyle=":", label=f"Best epoch ({best})")
        ax.set(title=title, xlabel="Epoch", ylabel=label)
        ax.legend()
    acc_ax.set_ylim(0.0, 1.0)
    fig.suptitle(run_title(history))
    save_figure(fig, RESULTS_DIR / history["run"] / "curves.png")
    plt.show()


def readable_class_name(name):
    """``AnnualCrop`` becomes ``Annual Crop``."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def plot_confusion_matrix(metrics, title):
    """The confusion matrix normalised by true class; saved as ``confusion_matrix.png``."""
    names = metrics["class_names"]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    rows = matrix.sum(axis=1, keepdims=True)
    values = np.divide(matrix, rows, out=np.zeros_like(matrix), where=rows != 0)

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    image = ax.imshow(values, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    wrapped = ["\n".join(textwrap.wrap(readable_class_name(n), width=11)) for n in names]
    ax.set(title=f"Confusion matrix: {title}", xlabel="Predicted class", ylabel="True class",
           xticks=range(len(names)), yticks=range(len(names)),
           xticklabels=wrapped, yticklabels=[readable_class_name(n) for n in names])
    ax.tick_params(axis="x", labelrotation=30, labelsize=8, pad=5)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(False)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    threshold = values.max() / 2
    for i in range(len(names)):
        for j in range(len(names)):
            text = "" if values[i, j] < 0.005 else f"{values[i, j]:.2f}"  # blank for near-zero cells
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > threshold else "black")
    fig.tight_layout(pad=1.4)
    save_figure(fig, RESULTS_DIR / metrics["run"] / "confusion_matrix.png")
    plt.show()
```
