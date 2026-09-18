"""Download EuroSAT, resize, draw the split, write the committed artifacts.

Owner: Member 1.  Run once per dataset, by one person, and the outputs are
committed so the other three never run it.

    +---------------------------------------------------------------------+
    |  IN   configs/stages/data.yaml -> inputs.*                          |
    |                                                                      |
    |  OUT  data/processed/eurosat_64/       (git-ignored, regenerable)    |
    |       data/splits/split_manifest.csv   COMMITTED                     |
    |       data/splits/split_meta.json      COMMITTED                     |
    |       data/splits/norm_stats.json      COMMITTED                     |
    +---------------------------------------------------------------------+

The split is drawn ONCE and committed. That is the reproducibility guarantee
the whole comparison rests on: Model A, Model B, MobileNetV2 and SqueezeNet
all see byte-identical splits, so an accuracy difference between them is
attributable to the architecture rather than to the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edgecnn.contracts.types import ResolvedConfig


def prepare_dataset(cfg: ResolvedConfig) -> dict[str, Path]:
    """End-to-end preparation. Idempotent - safe to re-run.

    Steps for Member 1:

    1. Download via ``torchvision.datasets.EuroSAT(root, download=True)``.
    2. Resize to 64x64 if needed. EuroSAT is already 64x64, so with
       ``resize_mode: none`` this is a copy; keep the branch anyway in case
       the team changes dataset.
    3. Draw a STRATIFIED 70/15/15 split seeded from ``cfg.seed``. Stratify so
       every class appears in all three splits - without it a rare class can
       land entirely in train and its test recall is then undefined, which
       breaks the macro average.
    4. Write the manifest, metadata and normalisation statistics.
    5. Validate all three against their schemas before returning. Never write
       an artifact that would fail its own contract.

    Returns:
        Mapping of output name to path, matching ``outputs`` in data.yaml.
    """
    raise NotImplementedError("Member 1: implement prepare_dataset")


def stratified_split(
    labels: list[int],
    fractions: dict[str, float],
    seed: int,
) -> list[str]:
    """Assign each sample to train, val or test, preserving class balance.

    Args:
        labels: Integer class label per sample, in manifest row order.
        fractions: ``{"train": .70, "val": .15, "test": .15}``.
        seed: RNG seed.

    Returns:
        List of split names, same length and order as ``labels``.

    Must be deterministic for a given ``(labels, fractions, seed)`` on every
    machine - the team runs Windows and Linux, and a split that differs
    between them silently invalidates the cross-member comparison.
    """
    raise NotImplementedError("Member 1: implement stratified_split")


def write_split_manifest(
    manifest_path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write ``split_manifest.csv`` with the frozen column order.

    Columns, in this order: ``relative_path, label_index, label_name, split``.

    ``relative_path`` uses forward slashes on every platform - use
    ``PurePosixPath`` or ``str(p).replace("\\\\", "/")``. A Windows-style path
    here makes the committed manifest unusable for anyone on Linux, and
    ``schema.validate_manifest`` rejects it.
    """
    raise NotImplementedError("Member 1: implement write_split_manifest")


def write_split_meta(meta_path: Path, manifest_path: Path, **fields: Any) -> Path:
    """Write ``split_meta.json``, including the SHA-256 of the manifest.

    The hash is how anyone checks that the manifest on disk is the one the
    committed results were produced from. If it stops matching, someone
    regenerated the split and every result in ``results/metrics/`` is stale.
    """
    raise NotImplementedError("Member 1: implement write_split_meta")
