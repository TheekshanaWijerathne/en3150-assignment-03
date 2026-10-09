"""Training and validation loss curves.       Assignment Section 4 [25]

Owner: Member 3.  (Metrics and tables belong to Members 1 and 4; these curves
belong to the section Member 3 is marked on.) Called in notebooks 03, 04, 05.

    +---------------------------------------------------------------------+
    |  IN   a run_id (reads the committed history.json)                    |
    |       OR a TrainResult / history dict still in memory (debug runs)   |
    |  OUT  matplotlib Figure - shown inline by the notebook               |
    |       + results/figures/<run_id>/curves.png   (write=True only)      |
    |       + results/figures/optimizer_overlay.png (write=True only)      |
    +---------------------------------------------------------------------+

Every function returns the Figure, so a notebook cell ending in
``plot_loss_curves(result, write=cfg.is_official)`` displays it. The PNG is
saved only when ``write`` is True - which notebooks set from
``cfg.is_official``, so a debug run can never overwrite an official figure.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from edgecnn.contracts.types import TrainResult

#: What a plotting function accepts: a committed run, or a result in memory.
HistorySource = Union[str, "TrainResult", dict[str, Any]]


def plot_loss_curves(source: HistorySource, write: bool = False) -> Figure:
    """Plot train and validation loss (and accuracy) for one run.

    The assignment asks specifically for training/validation loss curves, so
    loss is the required panel; a second accuracy panel is worth adding because
    the two together are what let the report argue about overfitting.

    Plot both on the same axis, not on separate figures - the gap between them
    IS the overfitting signal, and it is invisible across two plots. Mark the
    best epoch (the checkpoint ``evaluate_run`` uses) with a vertical line, so
    the reader can see which point the reported test accuracy came from.

    Args:
        source: ``run_id`` -> read the committed ``history.json`` via
            ``schema.read_json``; ``TrainResult`` or dict -> use it directly.
        write: Also save to ``paths.curves_png(run_id)``. Pass ``cfg.is_official``.
    """
    import matplotlib.pyplot as plt

    from edgecnn.contracts import paths
    from edgecnn.evaluation.plotting import MODEL_COLORS

    history = _normalise_history(source)
    epochs = history["epochs"]
    if not epochs:
        raise ValueError("cannot plot an empty training history")

    x = [row["epoch"] for row in epochs]
    best_epoch = int(
        history.get("best_epoch")
        or max(epochs, key=lambda row: row["val_acc"])["epoch"]
    )
    model_name = str(history.get("model", "model"))
    run_id = str(history.get("run_id", model_name))
    accent = MODEL_COLORS.get(model_name, "#4C72B0")

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    loss_ax.plot(x, [row["train_loss"] for row in epochs], label="Train", color=accent)
    loss_ax.plot(
        x,
        [row["val_loss"] for row in epochs],
        label="Validation",
        color=accent,
        linestyle="--",
    )
    loss_ax.axvline(best_epoch, color="0.35", linestyle=":", label=f"Best epoch ({best_epoch})")
    loss_ax.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()

    acc_ax.plot(x, [row["train_acc"] for row in epochs], label="Train", color=accent)
    acc_ax.plot(
        x,
        [row["val_acc"] for row in epochs],
        label="Validation",
        color=accent,
        linestyle="--",
    )
    acc_ax.axvline(best_epoch, color="0.35", linestyle=":")
    acc_ax.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0.0, 1.0))
    acc_ax.grid(alpha=0.25)
    acc_ax.legend()
    fig.suptitle(run_id)

    if write:
        destination = paths.curves_png(run_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)  # the returned figure still displays - once; left open, pyplot shows it twice
    return fig


def plot_optimizer_overlay(sources: list[HistorySource], write: bool = False) -> Figure:
    """Overlay the three Section 3 optimizer runs on shared axes.

    This single figure carries most of the Section 3 argument, so it has to
    support claims about both convergence speed and final performance:

    * validation loss for all three optimizers on one axis, coloured by
      ``plotting.OPTIMIZER_COLORS``;
    * a log-scaled x-axis option - early-epoch differences are where the
      momentum effect shows, and they are compressed to nothing on a linear
      axis over 30 epochs;
    * mark each run's best epoch.

    Expect the shape of the story to be: plain SGD converges slowly and noisily,
    momentum closes most of the gap, and Adam converges fastest early. Whether
    Adam also *ends* highest is an empirical question - report what the curves
    show rather than what is expected.

    With ``write=True``, also saves ``paths.OPTIMIZER_OVERLAY_PNG``. Notebook 03
    passes True only in official mode.
    """
    import matplotlib.pyplot as plt

    from edgecnn.contracts import paths
    from edgecnn.evaluation.plotting import OPTIMIZER_COLORS

    if not sources:
        raise ValueError("optimizer overlay requires at least one history")

    histories = [_normalise_history(source) for source in sources]
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for history in histories:
        epochs = history["epochs"]
        if not epochs:
            raise ValueError("cannot plot an empty training history")
        optimizer = str(history.get("optimizer", "optimizer"))
        color = OPTIMIZER_COLORS.get(optimizer, "0.4")
        x = [row["epoch"] for row in epochs]
        best_epoch = int(
            history.get("best_epoch")
            or max(epochs, key=lambda row: row["val_acc"])["epoch"]
        )
        loss_ax.plot(
            x,
            [row["val_loss"] for row in epochs],
            label=optimizer.replace("_", " ").title(),
            color=color,
        )
        acc_ax.plot(
            x,
            [row["val_acc"] for row in epochs],
            label=optimizer.replace("_", " ").title(),
            color=color,
        )
        loss_ax.scatter(
            [best_epoch],
            [_value_at_epoch(epochs, best_epoch, "val_loss")],
            color=color,
            marker="o",
            s=28,
            zorder=3,
        )
        acc_ax.scatter(
            [best_epoch],
            [_value_at_epoch(epochs, best_epoch, "val_acc")],
            color=color,
            marker="o",
            s=28,
            zorder=3,
        )

    for axis in (loss_ax, acc_ax):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Epoch (log scale)")
        axis.grid(alpha=0.25)
        axis.legend()
    loss_ax.set(title="Validation loss", ylabel="Cross-entropy loss")
    acc_ax.set(title="Validation accuracy", ylabel="Accuracy", ylim=(0.0, 1.0))
    fig.suptitle("Model B optimizer comparison")

    if write:
        paths.OPTIMIZER_OVERLAY_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.OPTIMIZER_OVERLAY_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)  # the returned figure still displays - once; left open, pyplot shows it twice
    return fig


def _normalise_history(source: HistorySource) -> dict[str, Any]:
    """Convert a run id, TrainResult, or mapping into one plotting shape."""
    from edgecnn.contracts import paths, schema

    if isinstance(source, str):
        return schema.read_json(paths.history_json(source), "history")
    if isinstance(source, dict):
        history = dict(source)
    elif is_dataclass(source):
        history = asdict(source)
    else:
        raise TypeError("source must be a run_id, TrainResult, or history dictionary")

    epoch_rows = []
    for row in history.get("epochs", []):
        epoch_rows.append(asdict(row) if is_dataclass(row) else dict(row))
    history["epochs"] = epoch_rows
    history.setdefault("model", history.pop("model_name", None))
    history.setdefault("optimizer", history.pop("optimizer_name", None))
    return history


def _value_at_epoch(epochs: list[dict[str, Any]], epoch: int, key: str) -> float:
    for row in epochs:
        if int(row["epoch"]) == epoch:
            return float(row[key])
    raise ValueError(f"best epoch {epoch} is not present in the history")
