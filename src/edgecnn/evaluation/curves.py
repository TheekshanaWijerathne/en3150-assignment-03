"""Training and validation loss curves.       Assignment Section 4 [25]

Owner: Member 3.  (Metrics and tables belong to Members 1 and 4; these curves
belong to the section Member 3 is marked on.)

    +---------------------------------------------------------------------+
    |  IN   results/metrics/<run_id>/history.json                          |
    |  OUT  results/figures/<run_id>/curves.png              COMMITTED     |
    |       results/figures/optimizer_overlay.png            COMMITTED     |
    +---------------------------------------------------------------------+

Reads from disk, not from a TrainResult in memory, so a figure can be redrawn
from committed JSON without re-running a 30-epoch job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def plot_loss_curves(run_id: str, out_path: Path | None = None) -> Path:
    """Plot train and validation loss (and accuracy) for one run.

    The assignment asks specifically for training/validation loss curves, so
    loss is the required panel; a second accuracy panel is worth adding because
    the two together are what let the report argue about overfitting.

    Plot both on the same axis, not on separate figures - the gap between them
    IS the overfitting signal, and it is invisible across two plots. Mark the
    best epoch (the checkpoint Member 1 evaluates) with a vertical line, so the
    reader can see which point the reported test accuracy came from.
    """
    raise NotImplementedError("Member 3: implement plot_loss_curves")


def plot_optimizer_overlay(run_ids: list[str], out_path: Path | None = None) -> Path:
    """Overlay the three Section 3 optimizer runs on shared axes.

    This single figure carries most of the Section 3 argument, so it has to
    support claims about both convergence speed and final performance:

    * validation loss for all three optimizers on one axis, coloured by
      OPTIMIZER_COLORS;
    * a log-scaled x-axis option - early-epoch differences are where the
      momentum effect shows, and they are compressed to nothing on a linear
      axis over 30 epochs;
    * mark each run's best epoch.

    Expect the shape of the story to be: plain SGD converges slowly and noisily,
    momentum closes most of the gap, and Adam converges fastest early. Whether
    Adam also *ends* highest is an empirical question - report what the curves
    show rather than what is expected.
    """
    raise NotImplementedError("Member 3: implement plot_optimizer_overlay")
