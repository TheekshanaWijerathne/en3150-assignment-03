"""Frozen data types that cross member boundaries.

    +---------------------------------------------------------------------+
    |  FROZEN INTERFACE.  Changing anything in this file changes work      |
    |  that is already in flight for three other people.  Open a PR, get   |
    |  one review, and tell the team in chat.  Do not edit on a feature    |
    |  branch and merge quietly.                                           |
    +---------------------------------------------------------------------+

Every one of these types is a *pure declaration*. No logic lives here, so the
file can be imported by anyone at any time without pulling in torch models,
matplotlib, or a dataset download.

The five seams these types carry (see the root README for the diagram):

    (1) DataBundle      Member 1 -> Members 2, 3, 4
    (2) build_model     Members 2, 4 -> Member 3     (see protocols.py)
    (3) TrainResult     Member 3 -> Members 1, 4
    (4) EvalResult      Member 1 -> Member 4
        ResourceProfile Member 1 -> Member 4
    (5) tables/figures  Member 4 -> the report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # keeps this module importable without torch installed
    from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project-wide constants. Read these; do not redefine them locally.
# ---------------------------------------------------------------------------

#: Assignment caps image resolution at 64x64. EuroSAT is natively this size,
#: so no downscaling loss is incurred.
IMAGE_SIZE: tuple[int, int] = (64, 64)

#: (C, H, W) fed to every model, custom and pretrained alike.
INPUT_SHAPE: tuple[int, int, int] = (3, 64, 64)

#: Assignment-mandated split proportions.
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

#: The only legal values of the `split` column in split_manifest.csv.
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

#: Hard cap on Model B trainable parameters (assignment Section 2).
#: Enforced by tests/test_param_budget.py, not by eyeballing a printout.
MODEL_B_PARAM_BUDGET: int = 100_000

#: Default seed. Overridable per-experiment, but every committed run uses 42
#: so the comparison table is reproducible.
DEFAULT_SEED: int = 42

#: Keys that MUST be present in artifacts/checkpoints/<run_id>/best.pt.
#: Asserted by tests/contracts/test_checkpoint_contract.py.
CHECKPOINT_KEYS: tuple[str, ...] = (
    "model_name",
    "state_dict",
    "epoch",
    "val_acc",
    "class_names",
    "input_shape",
    "seed",
    "config_snapshot",
)


# ---------------------------------------------------------------------------
# SEAM 1 - Data.  Member 1 produces; Members 2, 3 and 4 consume.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataBundle:
    """Everything a downstream member needs about the data, in one object.

    Member 1 returns this from ``edgecnn.data.build_dataloaders(cfg)``.
    Members 2, 3 and 4 never touch the filesystem to learn about the data -
    they read it off this object.

    Batch contract - identical for train, val and test, and non-negotiable::

        images : torch.float32, shape (B, 3, 64, 64), already normalised
        labels : torch.int64,   shape (B,), values in [0, num_classes)

    Normalisation statistics are fitted on the TRAIN split only. Fitting them
    on the full dataset leaks test information and the report's reproducibility
    section would be wrong.
    """

    train: DataLoader
    val: DataLoader
    test: DataLoader

    num_classes: int
    """10 for EuroSAT."""

    class_names: list[str]
    """Ordered so that ``class_names[i]`` is the label for integer class ``i``.

    This ordering is FROZEN once ``split_meta.json`` is written. Every
    confusion matrix, per-class metric and figure legend is indexed by it.
    Re-sorting this list silently invalidates every committed result.
    """

    input_shape: tuple[int, int, int]
    """(C, H, W). Always ``INPUT_SHAPE`` unless an ablation says otherwise."""

    norm_mean: tuple[float, ...]
    norm_std: tuple[float, ...]

    split_manifest_path: Path
    """Provenance: which manifest these loaders were built from."""

    seed: int

    def describe(self) -> str:
        """One-line summary for logging. Safe to call anywhere."""
        return (
            f"{self.num_classes} classes, input={self.input_shape}, "
            f"seed={self.seed}, manifest={self.split_manifest_path.name}"
        )


# ---------------------------------------------------------------------------
# SEAM 3 - Training.  Member 3 produces; Members 1 and 4 consume.
# ---------------------------------------------------------------------------


@dataclass
class EpochRecord:
    """One row of ``results/metrics/<run_id>/history.json`` -> ``epochs[]``.

    ``epoch_time_s`` is wall-clock time for the training pass plus the
    validation pass. It feeds the "training time per epoch" column of the
    Section 4 comparison table, so every member must measure it the same way -
    which is why only Member 3's trainer ever writes it.
    """

    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    lr: float
    epoch_time_s: float


@dataclass
class TrainResult:
    """Returned by the trainer and serialised to ``history.json``.

    Consumers:
        Member 3 -> plots the loss curves (Section 4).
        Member 1 -> reads ``mean_epoch_time_s`` into ``resources.json``.
        Member 4 -> aggregates across runs for the Section 6 table.
    """

    run_id: str
    model_name: str
    optimizer_name: str
    seed: int
    epochs: list[EpochRecord] = field(default_factory=list)

    best_epoch: int = -1
    best_val_acc: float = 0.0
    checkpoint_path: Path | None = None
    total_train_time_s: float = 0.0

    @property
    def mean_epoch_time_s(self) -> float:
        """Mean wall-clock seconds per epoch. Zero for an empty run."""
        if not self.epochs:
            return 0.0
        return sum(e.epoch_time_s for e in self.epochs) / len(self.epochs)


# ---------------------------------------------------------------------------
# SEAM 4 - Evaluation.  Member 1 produces; Member 4 consumes.
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Test-set outcome for one run -> ``test_metrics.json``.

    Confusion matrix orientation, fixed once so every figure agrees::

        confusion_matrix[i][j] = count of TRUE class i PREDICTED as class j

    Rows therefore sum to the support of each true class. Indices align with
    ``class_names``, which comes from ``DataBundle.class_names``.

    ``macro_*`` variants are used for the headline numbers because EuroSAT is
    close to balanced and the assignment asks for precision and recall without
    qualification; per-class values are kept in ``per_class`` for the report's
    discussion of which land-cover types get confused.
    """

    run_id: str
    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    """``{class_name: {"precision": .., "recall": .., "f1": .., "support": ..}}``"""

    confusion_matrix: list[list[int]]
    class_names: list[str]


@dataclass
class ResourceProfile:
    """Cost side of the accuracy/memory/compute trade-off -> ``resources.json``.

    This is the heart of Sections 4 and 6. Every model is measured by the same
    Member 1 code path so the numbers are comparable; a member who measures
    their own model their own way breaks the comparison.

    Units are baked into the field names on purpose - a silent KB/MB mix-up
    would make the Section 6 discussion wrong rather than merely imprecise.
    """

    run_id: str
    model_name: str
    trainable_params: int
    total_params: int
    model_size_kb: float
    """Serialised state_dict size on disk. Report as MB for pretrained models."""

    macs: int
    """Multiply-accumulate operations for a single 64x64 forward pass."""

    mean_epoch_time_s: float
    inference_latency_ms: float
    """Mean single-image forward pass. Record the device in ``device``."""

    peak_mem_mb: float
    device: str
    """e.g. "cpu" or "cuda:0 (NVIDIA RTX 3050)". Section 4 requires the hardware."""


# ---------------------------------------------------------------------------
# Config view.  Member 1 produces via edgecnn.config.loader; everyone consumes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """A merged ``base.yaml`` <- stage config <- experiment config.

    ``raw`` holds the fully merged mapping. The named fields are the handful of
    values that appear in signatures across every member's code, hoisted out so
    nobody writes ``cfg["experiment"]["seed"]`` and typoes the key path.
    """

    raw: dict[str, Any]
    run_id: str
    model_name: str
    optimizer_name: str
    seed: int
    device: str

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level config section, or an empty dict if absent."""
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}
