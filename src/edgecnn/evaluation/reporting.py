"""SEAM 5 - aggregation into report tables and figures.

Owner: Member 4.                  Assignment Sections 4 [25], 5 [20], 6 [20]

    +---------------------------------------------------------------------+
    |  IN   results/metrics/*/history.json       (Member 3)                |
    |       results/metrics/*/test_metrics.json  (Member 1)                |
    |       results/metrics/*/resources.json     (Member 1)                |
    |                                                                      |
    |  OUT  results/tables/custom_model_comparison.md    Section 4         |
    |       results/tables/optimizer_comparison.md       Section 3         |
    |       results/tables/final_comparison.md           Section 6         |
    |       results/figures/*.png  ->  report/figures/                     |
    +---------------------------------------------------------------------+

This is the last stage, and it consumes only committed JSON. That is why
``results/metrics/**/*.json`` is committed while checkpoints are not: Member 4
must be able to rebuild every table without a GPU and without re-running
anyone else's training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def load_all_runs() -> dict[str, dict[str, Any]]:
    """Load every discoverable run's three JSON artifacts.

    Uses ``paths.discover_runs()`` rather than a hard-coded list, so a run
    joins the tables as soon as its JSON is committed.

    Returns:
        ``{run_id: {"history": ..., "test_metrics": ..., "resources": ...}}``.

    A run missing ``test_metrics.json`` has been trained but not evaluated -
    skip it with a warning naming the run and the missing file, rather than
    crashing. Half the team's runs landing is the normal state of the repo for
    most of the project.
    """
    raise NotImplementedError("Member 4: implement load_all_runs")


def build_custom_comparison_table() -> Path:
    """Section 4: Model A vs Model B.

    Required columns, from the assignment text: total parameter count,
    estimated model size on disk (KB), training time per epoch, test accuracy.
    Add MACs - it is the number that actually demonstrates the depthwise
    separable saving, and accuracy per parameter is a weak proxy for it.

    The accompanying discussion has to state the trade-off in both directions:
    what accuracy was given up, and what was bought with it.
    """
    raise NotImplementedError("Member 4: implement build_custom_comparison_table")


def build_optimizer_comparison_table() -> Path:
    """Section 3: SGD vs SGD+momentum vs the chosen optimizer.

    Columns: optimizer, learning rate, best epoch, best validation accuracy,
    test accuracy, epochs to reach 90% of that run's own best validation
    accuracy, and mean epoch time.

    Include a memory column or note: Adam holds two extra state tensors per
    parameter, roughly tripling optimizer memory versus plain SGD. In a report
    about memory-constrained devices that is a relevant cost, not a footnote -
    though note it applies at training time, not at inference.
    """
    raise NotImplementedError("Member 4: implement build_optimizer_comparison_table")


def build_final_comparison_table() -> Path:
    """Section 6: Model B vs the fine-tuned SOTA backbones.

    The central table of the report. Columns: model, trainable params, total
    params, size (KB and MB), MACs, CPU inference latency, peak memory, test
    accuracy, macro precision, macro recall.

    Points the discussion should make, each of which the table should evidence:

    * Model B is likely to lose on accuracy and win by one to two orders of
      magnitude on parameters and size. State the ratio, not just both numbers.
    * The SOTA models were designed for 224x224. At 64x64 MobileNetV2's 32x
      stride leaves a 2x2 feature map, so they are handicapped here - a fair
      report says so rather than claiming a clean win for the custom model.
    * Pretrained weights bring ImageNet features for free, which matters most
      when training data is scarce. EuroSAT has 27,000 images, which is enough
      to train from scratch; the advantage would be far larger at 2,700.
    * Peak activation memory, not parameter count, is often what actually
      blocks MCU deployment. A model can fit in flash and still not run.
    * The honest conclusion may well be "it depends on the memory budget" -
      identify the threshold at which the answer flips, rather than declaring a
      winner. That is a stronger answer than either extreme.
    """
    raise NotImplementedError("Member 4: implement build_final_comparison_table")


def plot_confusion_matrix(run_id: str, normalize: bool = True) -> Path:
    """Render one run's confusion matrix.

    Stored counts are raw; normalise at plot time only. Normalise by TRUE class
    (rows sum to 1) so the diagonal reads as per-class recall - normalising by
    column gives precision instead, and an unlabelled plot leaves the reader
    unable to tell which.

    Label both axes with ``class_names`` and rotate the x labels; EuroSAT class
    names are long enough to overlap otherwise.
    """
    raise NotImplementedError("Member 4: implement plot_confusion_matrix")


def plot_tradeoff_scatter(x_metric: str = "trainable_params") -> Path:
    """The Section 6 argument in one figure.

    Test accuracy against cost, one point per model, log-scaled x. Produce it
    for both ``trainable_params`` and ``macs`` - they tell different stories,
    because parameter count tracks memory while MACs track compute, and a model
    can be cheap in one and expensive in the other.

    Label each point with its model name and draw the Pareto frontier. A reader
    should be able to see at a glance which models are dominated and which
    represent a real choice.
    """
    raise NotImplementedError("Member 4: implement plot_tradeoff_scatter")


def export_report_figures() -> list[Path]:
    """Copy every committed figure into ``report/figures/``.

    Keeps one source of truth: figures are generated into ``results/figures/``
    and mirrored, never hand-copied or regenerated separately for the report.
    """
    raise NotImplementedError("Member 4: implement export_report_figures")
