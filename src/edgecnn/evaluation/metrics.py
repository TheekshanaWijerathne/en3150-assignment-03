"""Test-set metrics - one code path that scores every model.

Every model in the report, custom and pretrained alike, is scored by these
functions right after it is trained. Separately written versions of "precision"
would differ in averaging, in how a class that is never predicted is handled,
and in class order - and the comparison tables would then compare numbers that
are not comparable, with nothing looking wrong.

Conventions, fixed here once:

* **Macro averages** for the headline precision, recall and F1: every class
  counts equally. (Micro-averaged precision equals accuracy for single-label
  classification, so it would only repeat the accuracy column.)
* A class that is **never predicted** gets precision 0 rather than an error.
* **Confusion matrix:** ``cm[i][j]`` counts the images of true class ``i``
  predicted as class ``j``, as raw counts. It is always square, with one row
  and one column per class - even for a class that is absent from the test images.

Team notes:
Owner: Member 1. Called inline in notebooks 03, 04 and 05, right after each
training run.

    IN   y_true, y_pred : int arrays of shape (N,); class_names from the DataBundle
    OUT  EvalResult - always returned, in every mode
         -> results/metrics/<run_id>/test_metrics.json  (official only)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgecnn.utils.io import repo_relative
from edgecnn.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    from torch import nn

    from edgecnn.contracts.types import DataBundle, EvalResult, ResolvedConfig

log = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    run_id: str,
    model_name: str,
) -> EvalResult:
    """Accuracy, macro precision / recall / F1, per-class metrics and the confusion matrix.

    Args:
        y_true: The true class index of each image.
        y_pred: The predicted class index of each image, in the same order.
        class_names: ``class_names[i]`` is the name of class ``i``.
        run_id: The run being scored.
        model_name: Its model key, e.g. ``"model_b"``.

    Returns:
        An :class:`~edgecnn.contracts.types.EvalResult` holding only plain
        Python numbers, so it serialises to JSON as it is.

    Raises:
        ValueError: if the two arrays differ in shape, are empty, or hold a
            class index outside ``range(len(class_names))``.
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    from edgecnn.contracts.types import EvalResult

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must be 1-D and the same length, got {y_true.shape} and "
            f"{y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("there are no predictions to score")
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if values.min() < 0 or values.max() >= len(class_names):
            raise ValueError(f"{name} holds a class index outside 0..{len(class_names) - 1}")

    labels = list(range(len(class_names)))  # explicit, so absent classes keep their row
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    return EvalResult(
        run_id=run_id,
        model_name=model_name,
        accuracy=float(np.mean(y_true == y_pred)),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        class_names=list(class_names),
    )


def predict(model: nn.Module, loader: object, device: str) -> tuple[np.ndarray, np.ndarray]:
    """The predicted class of every image a loader yields, in the loader's order.

    Runs in evaluation mode without gradients, taking the class with the
    highest raw score - the same class a softmax would pick. The model's
    previous train / eval mode is restored afterwards. The model must already
    be on ``device``.

    Returns:
        ``(y_true, y_pred)``: two ``int64`` arrays of the same length.

    Raises:
        ValueError: if the loader yields no images.
    """
    import numpy as np
    import torch

    target = torch.device(device)
    was_training = model.training
    model.eval()
    truths, predictions = [], []
    try:
        with torch.no_grad():
            for images, labels in loader:  # type: ignore[attr-defined]
                logits = model(images.to(target, non_blocking=True))
                predictions.append(logits.argmax(dim=1).cpu())
                truths.append(torch.as_tensor(labels).cpu())
    finally:
        model.train(was_training)
    if not truths:
        raise ValueError("the loader yielded no images")
    return (
        torch.cat(truths).numpy().astype(np.int64),
        torch.cat(predictions).numpy().astype(np.int64),
    )


def evaluate_run(
    cfg: ResolvedConfig,
    data: DataBundle,
    checkpoint_path: Path | None = None,
) -> EvalResult:
    """Score a trained run on the test split, from its best checkpoint.

    1. Loads ``checkpoint_path`` - by default the run's ``best.pt``, the epoch
       with the highest validation accuracy - and checks that it holds every
       required key and was saved for this model, with these classes in this
       order, at this input size.
    2. Rebuilds the model with ``build_model_from_config`` and loads the weights.
    3. Predicts the test split - the only time any code reads it.
    4. Checks that the confusion matrix counts every test image exactly once.
    5. Only if ``cfg.is_official``, writes ``results/metrics/<run_id>/test_metrics.json``.

    Returns:
        The :class:`~edgecnn.contracts.types.EvalResult`, in every mode, so the
        notebook can show it.

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
        ContractViolation: if the checkpoint is incomplete or was saved for a
            different model, class order or input size, or if the scored
            images do not add up to the test split.
    """
    import torch

    from edgecnn.contracts import paths, schema
    from edgecnn.contracts.schema import ContractViolation
    from edgecnn.models.registry import build_model_from_config
    from edgecnn.utils.device import resolve_device

    path = Path(checkpoint_path) if checkpoint_path is not None else paths.best_checkpoint(
        cfg.run_id, official=cfg.is_official
    )
    schema.validate_checkpoint(path)
    device = resolve_device(cfg.device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    expected = {
        "model_name": cfg.model_name,
        "class_names": list(data.class_names),
        "input_shape": list(data.input_shape),
    }
    for key, value in expected.items():
        found = checkpoint[key] if isinstance(value, str) else list(checkpoint[key])
        if found != value:
            raise ContractViolation(
                f"{repo_relative(path)} was saved with {key} = {found!r}, but this run expects "
                f"{value!r}: the checkpoint belongs to a different run or split"
            )

    model = build_model_from_config(cfg, data.num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    y_true, y_pred = predict(model, data.test, device)
    result = compute_metrics(y_true, y_pred, data.class_names, cfg.run_id, cfg.model_name)
    _check_whole_test_split(result, data, cfg)
    log.info("%s: test accuracy %.4f on %d images", cfg.run_id, result.accuracy, len(y_true))

    if cfg.is_official:
        payload = _test_metrics_payload(result, path)
        schema.write_json(paths.test_metrics_json(cfg.run_id), payload, "test_metrics")
    return result


def _check_whole_test_split(result: EvalResult, data: DataBundle, cfg: ResolvedConfig) -> None:
    """Raise unless the confusion matrix counts every test image exactly once."""
    from edgecnn.contracts import paths, schema
    from edgecnn.contracts.schema import ContractViolation

    scored = sum(sum(row) for row in result.confusion_matrix)
    expected = {"the test loader": len(data.test.dataset)}  # type: ignore[arg-type]
    if cfg.section("dataset").get("name") != "synthetic" and paths.SPLIT_META.exists():
        counts = schema.read_json(paths.SPLIT_META, "split_meta")["counts"]
        expected["split_meta.json"] = counts["test"]
    for source, count in expected.items():
        if scored != count:
            raise ContractViolation(
                f"{scored} test images were scored, but {source} has {count}: the wrong split "
                "was evaluated"
            )


def _test_metrics_payload(result: EvalResult, checkpoint: Path | None) -> dict[str, Any]:
    """The contents of ``test_metrics.json`` for ``result``.

    Adds what the metrics file records beyond the ``EvalResult``: the split,
    the sample count, support-weighted precision and recall, which checkpoint
    was scored, and when.
    """
    names = result.class_names
    support = {name: result.per_class[name]["support"] for name in names}
    total = sum(support.values())

    def weighted(key: str) -> float:
        value = sum(result.per_class[name][key] * support[name] for name in names) / total
        return min(1.0, value)  # guards a rounding error just above 1

    payload: dict[str, Any] = {
        "run_id": result.run_id,
        "model": result.model_name,
        "split": "test",
        "accuracy": result.accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "weighted_precision": weighted("precision"),
        "weighted_recall": weighted("recall"),
        "per_class": result.per_class,
        "confusion_matrix": result.confusion_matrix,
        "class_names": names,
        "num_samples": total,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if checkpoint is not None:
        payload["checkpoint"] = repo_relative(checkpoint)
    return payload
