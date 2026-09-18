"""Early stopping and checkpoint selection.

Owner: Member 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class EarlyStopping:
    """Stop when the monitored metric has not improved for `patience` epochs.

    One constraint specific to this assignment: `min_epochs` defaults to 20 and
    early stopping must not fire before it. The assignment mandates at least 20
    epochs, and `history.schema.json` enforces `minItems: 20` on the epochs
    array - so a run that stops at epoch 12 cannot be committed, and finding
    that out after the run is a wasted half hour.
    """

    def __init__(
        self,
        monitor: str = "val_acc",
        mode: str = "max",
        patience: int = 8,
        min_delta: float = 0.0,
        min_epochs: int = 20,
    ) -> None:
        raise NotImplementedError("Member 3: implement EarlyStopping.__init__")

    def step(self, metrics: dict[str, float], epoch: int) -> bool:
        """Record this epoch's metrics; return True to stop training."""
        raise NotImplementedError("Member 3: implement EarlyStopping.step")


class CheckpointManager:
    """Keeps `best.pt` and `last.pt` for one run.

    `best` is selected on validation accuracy, never on test accuracy - the
    test split is not visible to training code at all. Member 1 evaluates
    `best.pt`, so which epoch this picks directly determines the number that
    lands in the report.
    """

    def __init__(
        self,
        run_id: str,
        monitor: str = "val_acc",
        mode: str = "max",
        save_last: bool = True,
    ) -> None:
        raise NotImplementedError("Member 3: implement CheckpointManager.__init__")

    def maybe_save(self, model: object, epoch: int, metrics: dict[str, float]) -> Path | None:
        """Save when the monitored metric improves. Returns the path, or None."""
        raise NotImplementedError("Member 3: implement CheckpointManager.maybe_save")
