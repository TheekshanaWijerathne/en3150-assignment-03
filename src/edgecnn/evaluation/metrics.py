"""SEAM 4 - test-set metrics.                  Assignment Section 4 [25]

Owner: Member 1.  Used for EVERY model, custom and pretrained alike.

    +---------------------------------------------------------------------+
    |  IN   y_true, y_pred : int arrays of shape (N,)                      |
    |       class_names    : from DataBundle.class_names                   |
    |  OUT  EvalResult -> results/metrics/<run_id>/test_metrics.json       |
    +---------------------------------------------------------------------+

Why one owner rather than each member scoring their own model: four
implementations of "precision" will differ in averaging, in zero-division
handling, and in class ordering. The Section 6 table would then compare
numbers that are not comparable, and the error is invisible - every column
looks plausible. One code path removes the possibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from torch import nn

    from edgecnn.contracts.types import DataBundle, EvalResult, ResolvedConfig


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    run_id: str,
    model_name: str,
) -> EvalResult:
    """Accuracy, macro precision/recall/F1, per-class metrics, confusion matrix.

    Implementation notes:

    * Use ``sklearn.metrics.precision_recall_fscore_support`` with
      ``average="macro"`` and ``zero_division=0``. Macro, not micro: for
      single-label classification micro-precision is numerically equal to
      accuracy, so a micro column would silently duplicate the accuracy column
      in the Section 4 table.
    * Pass ``labels=range(len(class_names))`` explicitly to
      ``confusion_matrix``. Without it, sklearn infers labels from the data,
      so a class absent from the predictions vanishes from the matrix and the
      result is no longer square - which then breaks the plot and the report
      table, far from the cause.
    * Orientation is frozen: ``cm[i][j]`` is true class ``i`` predicted as
      ``j``. sklearn's default already matches this; do not transpose.
    * Cast to plain Python ``int`` / ``float`` before serialising - NumPy
      scalars are not JSON-serialisable and the failure is confusing.
    """
    raise NotImplementedError("Member 1: implement compute_metrics")


def predict(model: nn.Module, loader: object, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader. Returns ``(y_true, y_pred)``.

    ``model.eval()`` and ``torch.no_grad()``. Take ``argmax`` over the logits -
    remember the models return raw logits, which is fine here because argmax
    is invariant under softmax.

    Do not shuffle. Order must be reproducible so that a per-sample error
    analysis can be repeated.
    """
    raise NotImplementedError("Member 1: implement predict")


def evaluate_run(cfg: ResolvedConfig, data: DataBundle) -> EvalResult:
    """Full evaluation of one run, from checkpoint to committed JSON.

    1. Load ``artifacts/checkpoints/<run_id>/best.pt``, validating its keys
       with ``schema.validate_checkpoint``.
    2. Rebuild the model via ``build_model`` and load the state dict.
    3. Predict over ``data.test`` - the first and only time this split is read.
    4. ``compute_metrics``, then write ``test_metrics.json`` through
       ``schema.write_json``.

    Sanity check worth asserting: the confusion matrix must sum to
    ``split_meta.json -> counts.test``. If it does not, the wrong split was
    evaluated, and that is far better caught here than discovered in the viva.
    """
    raise NotImplementedError("Member 1: implement evaluate_run")
