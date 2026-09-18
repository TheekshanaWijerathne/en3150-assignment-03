"""The Section 3 optimizer comparison.           Assignment Section 3 [15]

Owner: Member 3.

    +---------------------------------------------------------------------+
    |  IN   configs/stages/training.yaml -> optimizer_study                |
    |  OUT  three runs, identical except for the optimizer:                |
    |         results/metrics/model_b__sgd__seed42/history.json            |
    |         results/metrics/model_b__sgd_momentum__seed42/history.json   |
    |         results/metrics/model_b__adam__seed42/history.json           |
    |       + results/figures/optimizer_overlay.png                        |
    +---------------------------------------------------------------------+

The experiment only isolates the optimizer if everything else is held fixed:
same seed, same initial weights, same data order, same scheduler, same epochs.
Change two things and the report cannot attribute the difference to either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgecnn.contracts.types import ResolvedConfig, TrainResult


def run_optimizer_study(cfg: ResolvedConfig) -> dict[str, TrainResult]:
    """Train the same model under each optimizer variant.

    Returns:
        ``{optimizer_name: TrainResult}``.

    Reseed before EACH variant, not once at the start. Otherwise the second
    and third runs inherit a different RNG state and therefore different
    initial weights, and the comparison silently measures initialisation
    variance alongside the optimizer.
    """
    raise NotImplementedError("Member 3: implement run_optimizer_study")


def summarize_study(results: dict[str, TrainResult]) -> dict[str, Any]:
    """Reduce the three runs to the Section 3 comparison table.

    Report convergence AND final performance - the assignment asks about both,
    and they can disagree. Suggested columns:

    * ``best_val_acc``          - final performance
    * ``epochs_to_90pct_best``  - convergence speed, defined as the first epoch
      reaching 90% of that run's own best validation accuracy. Self-relative,
      so a slow-but-strong optimizer is not penalised twice.
    * ``mean_epoch_time_s``     - wall-clock cost. Adam holds two extra state
      tensors per parameter, which is worth a sentence in a report about
      memory-constrained devices: on an MCU that can matter more than the
      convergence advantage.
    * ``final_train_val_gap``   - overfitting signal.

    The momentum discussion Section 3 asks for should separate the two effects:
    the velocity term damps oscillation across narrow ravines, AND accelerates
    travel along consistently descending directions. Show both in the curves
    rather than asserting them.
    """
    raise NotImplementedError("Member 3: implement summarize_study")
