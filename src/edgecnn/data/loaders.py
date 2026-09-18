"""Build the three DataLoaders that every other member consumes.

Owner: Member 1.  This file *is* Seam 1.

    +---------------------------------------------------------------------+
    |  IN   cfg : ResolvedConfig (needs the `data` stage section)          |
    |       data/splits/split_manifest.csv   - which image, which split    |
    |       data/splits/split_meta.json      - class names, counts, seed   |
    |       data/splits/norm_stats.json      - train-only mean/std         |
    |                                                                      |
    |  OUT  DataBundle                                                     |
    |       batches: images float32 (B,3,64,64) normalised                 |
    |                labels int64  (B,) in [0, num_classes)                |
    +---------------------------------------------------------------------+
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edgecnn.contracts.types import DataBundle

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from edgecnn.contracts.types import ResolvedConfig


def build_dataloaders(cfg: ResolvedConfig) -> DataBundle:
    """Construct train/val/test loaders from the committed split manifest.

    Implementation notes for Member 1:

    * Read the manifest with pandas and filter by the ``split`` column. Do NOT
      re-draw the split here - it is drawn once by ``prepare_dataset`` and
      committed, so every member and every run sees identical data.
    * ``cfg.raw["inputs"]["dataset"]["name"] == "synthetic"`` must route to
      :func:`edgecnn.data.synthetic.make_synthetic_fixture`. That branch is
      what unblocks Members 3 and 4 before EuroSAT is downloaded, so build it
      first.
    * Augmentation applies to the train loader ONLY. Augmenting val or test
      makes the reported numbers unreproducible.
    * ``shuffle=True`` for train, ``False`` for val and test. Shuffling the
      test loader would scramble nothing statistically but makes a
      sample-by-sample error analysis impossible to reproduce.
    * Seed the loader generator from ``cfg.seed`` and set ``worker_init_fn``
      so augmentation is reproducible across machines.
    * Honour ``cfg.raw["subset_fraction"]`` by subsampling the train split
      only, stratified, for fast development runs.
    * Pretrained runs may set ``normalization.use_imagenet_stats: true``; in
      that case use the ImageNet constants and record the override in the
      returned bundle's ``norm_mean`` / ``norm_std`` so downstream code and
      the report agree on what was actually applied.

    Raises:
        FileNotFoundError: if the manifest is missing - point the caller at
            ``python scripts/prepare_data.py``.
        ContractViolation: if the manifest or metadata fails its schema.
    """
    raise NotImplementedError(
        "Member 1: implement build_dataloaders. "
        "Ship the dataset.name == 'synthetic' branch FIRST - Members 3 and 4 "
        "are blocked on it. See tests/fixtures/README.md."
    )


def build_single_loader(cfg: ResolvedConfig, split: str) -> DataLoader:
    """Build one loader in isolation.

    Convenience for benchmarking and error analysis. ``split`` must be one of
    :data:`edgecnn.contracts.types.SPLIT_NAMES`.
    """
    raise NotImplementedError("Member 1: implement build_single_loader")


def compute_norm_stats(cfg: ResolvedConfig) -> dict[str, list[float]]:
    """Compute per-channel mean and std over the TRAIN split only.

    Called once by ``prepare_dataset`` and the result written to
    ``norm_stats.json``. Fitting on val or test leaks held-out information
    into training - ``norm_stats.schema.json`` pins ``fitted_on`` to
    ``"train"`` so the mistake cannot be committed silently.

    Returns:
        ``{"mean": [r, g, b], "std": [r, g, b]}`` over pixels scaled to [0, 1].
    """
    raise NotImplementedError("Member 1: implement compute_norm_stats")
