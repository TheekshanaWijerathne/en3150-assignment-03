from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from matplotlib.figure import Figure

from edgecnn.contracts import paths, schema
from edgecnn.evaluation.plotting import (
    color_for,
    save_figure,
)

if TYPE_CHECKING:
    from edgecnn.contracts.types import EvalResult

EvalSource = Union[str, "EvalResult"]


def _table_runs(table_name: str) -> list[str]:
    from edgecnn.config import load_stage

    reporting = load_stage("evaluation")["reporting"]
    for table in reporting["tables"]:
        if table["name"] == table_name:
            return list(table["runs"])

    raise KeyError(f"unknown table: {table_name}")


def _read_artifact(
    run_id: str,
    artifact: str,
) -> dict[str, Any]:
    file_info = {
        "history": (
            paths.history_json(run_id),
            "history",
        ),
        "test_metrics": (
            paths.test_metrics_json(run_id),
            "test_metrics",
        ),
        "resources": (
            paths.resources_json(run_id),
            "resources",
        ),
    }

    path, schema_name = file_info[artifact]
    if not path.exists():
        raise FileNotFoundError(f"{run_id} is missing {path.name}")

    return schema.read_json(path, schema_name)


def _save_table(
    headers: list[str],
    rows: list[list[Any]],
    destination: Path,
    write: bool,
    caption: str,
) -> str:
    from edgecnn.utils.io import markdown_table

    text = markdown_table(headers, rows, caption)

    if write:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.write_text(
            text,
            encoding="utf-8",
        )

    return text


def load_all_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}

    for run_id in paths.discover_runs():
        test_path = paths.test_metrics_json(run_id)

        if not test_path.exists():
            warnings.warn(
                f"skipping {run_id}: missing test_metrics.json",
                stacklevel=2,
            )
            continue

        payloads: dict[str, Any] = {
            "test_metrics": schema.read_json(
                test_path,
                "test_metrics",
            )
        }

        history_path = paths.history_json(run_id)
        resources_path = paths.resources_json(run_id)

        if history_path.exists():
            payloads["history"] = schema.read_json(
                history_path,
                "history",
            )

        if resources_path.exists():
            payloads["resources"] = schema.read_json(
                resources_path,
                "resources",
            )

        runs[run_id] = payloads

    return runs


def check_same_hardware(
    run_ids: list[str],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}

    for run_id in run_ids:
        history_path = paths.history_json(run_id)

        if history_path.exists():
            history = schema.read_json(
                history_path,
                "history",
            )
            device = str(history.get("device", "unknown device"))
        else:
            device = "missing history"

        groups.setdefault(device, []).append(run_id)

    if len(groups) > 1:
        warnings.warn(
            "the selected runs were not trained on one device",
            stacklevel=2,
        )

    return groups


def build_custom_comparison_table(
    write: bool = True,
) -> str:
    rows = []

    for run_id in _table_runs("custom_model_comparison"):
        metrics = _read_artifact(
            run_id,
            "test_metrics",
        )
        resources = _read_artifact(
            run_id,
            "resources",
        )

        rows.append(
            [
                metrics["model"],
                resources["total_params"],
                resources["model_size_kb"],
                resources["macs"],
                resources["mean_epoch_time_s"],
                metrics["accuracy"] * 100,
            ]
        )

    return _save_table(
        [
            "Model",
            "Total parameters",
            "Model size (KB)",
            "MACs",
            "Mean epoch time (s)",
            "Test accuracy (%)",
        ],
        rows,
        paths.CUSTOM_COMPARISON_TABLE,
        write,
        "Model A and Model B comparison.",
    )


def build_optimizer_comparison_table(
    write: bool = True,
) -> str:
    rows = []

    for run_id in _table_runs("optimizer_comparison"):
        history = _read_artifact(run_id, "history")
        metrics = _read_artifact(
            run_id,
            "test_metrics",
        )

        epochs = history["epochs"]
        best_accuracy = float(history["best_val_acc"])
        target = 0.9 * best_accuracy

        epoch_to_target = next(
            (epoch["epoch"] for epoch in epochs if epoch["val_acc"] >= target),
            "not reached",
        )

        mean_epoch_time = sum(epoch["epoch_time_s"] for epoch in epochs) / len(epochs)

        learning_rate = history.get(
            "hyperparameters",
            {},
        ).get("lr", epochs[0]["lr"])

        rows.append(
            [
                history["optimizer"],
                learning_rate,
                history["best_epoch"],
                best_accuracy * 100,
                metrics["accuracy"] * 100,
                epoch_to_target,
                mean_epoch_time,
            ]
        )

    return _save_table(
        [
            "Optimizer",
            "Learning rate",
            "Best epoch",
            "Best validation accuracy (%)",
            "Test accuracy (%)",
            "Epoch to 90% of best",
            "Mean epoch time (s)",
        ],
        rows,
        paths.OPTIMIZER_COMPARISON_TABLE,
        write,
        "Model B optimizer comparison.",
    )


def build_final_comparison_table(
    write: bool = True,
) -> str:
    rows = []

    for run_id in _table_runs("final_comparison"):
        metrics = _read_artifact(
            run_id,
            "test_metrics",
        )
        resources = _read_artifact(
            run_id,
            "resources",
        )

        size_kb = float(resources["model_size_kb"])

        rows.append(
            [
                metrics["model"],
                resources["trainable_params"],
                resources["total_params"],
                size_kb,
                size_kb / 1024,
                resources["macs"],
                resources["inference_latency_ms"],
                resources.get("peak_mem_mb"),
                metrics["accuracy"] * 100,
                metrics["macro_precision"] * 100,
                metrics["macro_recall"] * 100,
            ]
        )

    return _save_table(
        [
            "Model",
            "Trainable parameters",
            "Total parameters",
            "Size (KB)",
            "Size (MB)",
            "MACs",
            "CPU latency (ms)",
            "Peak memory (MB)",
            "Test accuracy (%)",
            "Macro precision (%)",
            "Macro recall (%)",
        ],
        rows,
        paths.FINAL_COMPARISON_TABLE,
        write,
        "Model B versus pretrained lightweight models.",
    )


def plot_confusion_matrix(
    source: EvalSource,
    normalize: bool = True,
    write: bool = False,
) -> Figure:
    if isinstance(source, str):
        payload = schema.read_json(paths.test_metrics_json(source), "test_metrics")
        run_id = source
        class_names = list(payload["class_names"])
        matrix = np.asarray(payload["confusion_matrix"], dtype=float)
    else:
        run_id = source.run_id
        class_names = list(source.class_names)
        matrix = np.asarray(source.confusion_matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion matrix must be square")
    if matrix.shape[0] != len(class_names):
        raise ValueError("confusion matrix size must match class_names")

    values = matrix
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        values = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix),
            where=row_sums != 0,
        )

    size = max(6.0, len(class_names) * 0.7)
    figure = Figure(figsize=(size, size))
    axis = figure.subplots()
    image = axis.imshow(values, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set(
        title=f"Confusion matrix — {run_id}",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", labelrotation=45)

    threshold = values.max() / 2 if values.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            label = f"{values[row, column]:.2f}" if normalize else str(int(matrix[row, column]))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )

    figure.tight_layout()
    if write:
        save_figure(figure, paths.confusion_matrix_png(run_id))
    return figure


def plot_tradeoff_scatter(
    x_metric: str = "trainable_params",
    write: bool = True,
) -> Figure:
    if x_metric not in {"trainable_params", "macs"}:
        raise ValueError("x_metric must be trainable_params or macs")

    points = []

    for run_id in _table_runs("final_comparison"):
        metrics = _read_artifact(
            run_id,
            "test_metrics",
        )
        resources = _read_artifact(
            run_id,
            "resources",
        )

        points.append(
            (
                metrics["model"],
                float(resources[x_metric]),
                float(metrics["accuracy"]) * 100,
            )
        )

    figure = Figure(figsize=(7, 5))
    axis = figure.subplots()

    for model_name, cost, accuracy in points:
        axis.scatter(
            cost,
            accuracy,
            color=color_for(model_name),
            s=70,
        )
        axis.annotate(
            model_name,
            (cost, accuracy),
            xytext=(6, 6),
            textcoords="offset points",
        )

    frontier = []
    best_accuracy = float("-inf")
    for point in sorted(points, key=lambda item: item[1]):
        if point[2] > best_accuracy:
            frontier.append(point)
            best_accuracy = point[2]

    if frontier:
        axis.plot(
            [point[1] for point in frontier],
            [point[2] for point in frontier],
            linestyle="--",
            color="0.35",
            label="Pareto frontier",
        )
        axis.legend()

    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters" if x_metric == "trainable_params" else "MACs per image")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("Accuracy versus model cost")
    figure.tight_layout()

    filename = (
        "accuracy_vs_params.png" if x_metric == "trainable_params" else "accuracy_vs_macs.png"
    )

    if write:
        save_figure(
            figure,
            paths.FIGURES_DIR / filename,
        )

    return figure


def export_report_figures(
    write: bool = True,
) -> list[Path]:
    sources = sorted(paths.FIGURES_DIR.rglob("*.png"))

    destinations = [
        paths.REPORT_FIGURES_DIR / source.relative_to(paths.FIGURES_DIR) for source in sources
    ]

    if write:
        for source, destination in zip(
            sources,
            destinations,
            strict=True,
        ):
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(source, destination)

    return destinations
