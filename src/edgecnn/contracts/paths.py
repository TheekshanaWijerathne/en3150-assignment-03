"""Canonical path derivation - the single source of truth for *where things go*.

    +---------------------------------------------------------------------+
    |  FROZEN INTERFACE.  PR + 1 review to change.                         |
    +---------------------------------------------------------------------+

Why this file exists
--------------------
Member 3 writes a checkpoint. Member 1 reads it back to evaluate. Member 4
globs the results directory to build the Section 6 table. If any of the three
builds that path by hand, a single typo produces an empty comparison table the
day before submission - and it fails *silently*, because a glob that matches
nothing is not an error.

So: nobody concatenates path strings. Everyone calls a function from here.

The run_id convention
---------------------
    run_id = "{model_name}__{optimizer_name}__seed{seed}"

Double underscore separates the fields, single underscores live inside them
(``sgd_momentum``, ``model_b``, ``squeezenet1_1``), so ``split("__")`` round
trips cleanly. Examples::

    model_b__sgd_momentum__seed42
    mobilenet_v2__adam__seed42

Layout produced::

    artifacts/checkpoints/<run_id>/best.pt
                                   last.pt
    results/metrics/<run_id>/history.json
                             test_metrics.json
                             resources.json
    results/figures/<run_id>/curves.png
                             confusion_matrix.png
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository anchors. Resolved from THIS file's location, so every path works
# no matter which directory a script, test or notebook was launched from.
# src/edgecnn/contracts/paths.py -> up 4 -> repository root
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

CONFIGS_DIR: Path = REPO_ROOT / "configs"
SCHEMAS_DIR: Path = CONFIGS_DIR / "contracts"

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"

ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
CHECKPOINTS_DIR: Path = ARTIFACTS_DIR / "checkpoints"
EXPORTS_DIR: Path = ARTIFACTS_DIR / "exports"

RESULTS_DIR: Path = REPO_ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"

REPORT_DIR: Path = REPO_ROOT / "report"
REPORT_FIGURES_DIR: Path = REPORT_DIR / "figures"

TESTS_DIR: Path = REPO_ROOT / "tests"
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"

# --- Seam 1 artifacts (Member 1) -------------------------------------------
SPLIT_MANIFEST: Path = SPLITS_DIR / "split_manifest.csv"
SPLIT_META: Path = SPLITS_DIR / "split_meta.json"
NORM_STATS: Path = SPLITS_DIR / "norm_stats.json"

# --- Seam 5 artifacts (Member 4) -------------------------------------------
CUSTOM_COMPARISON_TABLE: Path = TABLES_DIR / "custom_model_comparison.md"
OPTIMIZER_COMPARISON_TABLE: Path = TABLES_DIR / "optimizer_comparison.md"
FINAL_COMPARISON_TABLE: Path = TABLES_DIR / "final_comparison.md"

_RUN_ID_RE = re.compile(r"^(?P<model>[a-z0-9_]+)__(?P<optimizer>[a-z0-9_]+)__seed(?P<seed>\d+)$")


def make_run_id(model_name: str, optimizer_name: str, seed: int) -> str:
    """Build the canonical run identifier.

    >>> make_run_id("model_b", "sgd_momentum", 42)
    'model_b__sgd_momentum__seed42'

    Raises:
        ValueError: if a component contains ``__`` or an uppercase character,
            which would make the id ambiguous to :func:`parse_run_id`.
    """
    for label, value in (("model_name", model_name), ("optimizer_name", optimizer_name)):
        if "__" in value:
            raise ValueError(f"{label}={value!r} must not contain a double underscore")
        if value != value.lower():
            raise ValueError(f"{label}={value!r} must be lowercase")
    run_id = f"{model_name}__{optimizer_name}__seed{seed}"
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"constructed run_id {run_id!r} is not well formed")
    return run_id


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    """Inverse of :func:`make_run_id`.

    Member 4 uses this to group committed results by model and by optimizer
    when assembling the Section 3 and Section 6 tables.

    >>> parse_run_id("mobilenet_v2__adam__seed42")
    ('mobilenet_v2', 'adam', 42)
    """
    match = _RUN_ID_RE.match(run_id)
    if match is None:
        raise ValueError(f"{run_id!r} does not match <model>__<optimizer>__seed<n>")
    return match["model"], match["optimizer"], int(match["seed"])


# ---------------------------------------------------------------------------
# Per-run directories and files
# ---------------------------------------------------------------------------


def checkpoint_dir(run_id: str) -> Path:
    """Directory holding this run's weights. Git-ignored (too large)."""
    return CHECKPOINTS_DIR / run_id


def best_checkpoint(run_id: str) -> Path:
    """Highest-validation-accuracy checkpoint. Member 1 evaluates *this* one."""
    return checkpoint_dir(run_id) / "best.pt"


def last_checkpoint(run_id: str) -> Path:
    """Final-epoch checkpoint, kept for resuming an interrupted run."""
    return checkpoint_dir(run_id) / "last.pt"


def metrics_dir(run_id: str) -> Path:
    """Directory holding this run's JSON artifacts. COMMITTED to git."""
    return METRICS_DIR / run_id


def history_json(run_id: str) -> Path:
    """Seam 3: per-epoch training record. Written by Member 3."""
    return metrics_dir(run_id) / "history.json"


def test_metrics_json(run_id: str) -> Path:
    """Seam 4: accuracy / precision / recall / confusion matrix. Member 1."""
    return metrics_dir(run_id) / "test_metrics.json"


def resources_json(run_id: str) -> Path:
    """Seam 4: params, size, MACs, timing. Member 1."""
    return metrics_dir(run_id) / "resources.json"


def figures_dir(run_id: str) -> Path:
    """Per-run figures. COMMITTED so the report can be built without re-running."""
    return FIGURES_DIR / run_id


def curves_png(run_id: str) -> Path:
    """Section 4 training/validation loss curves. Written by Member 3."""
    return figures_dir(run_id) / "curves.png"


def confusion_matrix_png(run_id: str) -> Path:
    """Section 4 confusion matrix figure. Written by Member 4."""
    return figures_dir(run_id) / "confusion_matrix.png"


def experiment_config(run_id: str) -> Path:
    """The composed config that defines this run."""
    return CONFIGS_DIR / "experiments" / f"{run_id.rsplit('__seed', 1)[0]}.yaml"


def ensure_run_dirs(run_id: str) -> None:
    """Create every directory this run writes into. Safe to call repeatedly."""
    for directory in (checkpoint_dir(run_id), metrics_dir(run_id), figures_dir(run_id)):
        directory.mkdir(parents=True, exist_ok=True)


def discover_runs() -> list[str]:
    """Every run_id with a metrics directory on disk, sorted.

    This is how Member 4 finds work to aggregate without hard-coding the
    experiment list - a run appears in the tables as soon as its JSON is
    committed.
    """
    if not METRICS_DIR.exists():
        return []
    return sorted(
        path.name
        for path in METRICS_DIR.iterdir()
        if path.is_dir() and _RUN_ID_RE.match(path.name)
    )
