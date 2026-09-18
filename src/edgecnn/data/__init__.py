"""SEAM 1 - dataset preparation and DataLoader construction.

Owner: Member 1.  Consumed by Members 2, 3 and 4.

The whole subpackage exists to produce one object::

    from edgecnn.data import build_dataloaders
    data = build_dataloaders(cfg)      # -> edgecnn.contracts.DataBundle

Nobody downstream opens an image file, computes a normalisation statistic or
decides which sample is in which split. They call this and read the bundle.
"""

from edgecnn.data.loaders import build_dataloaders
from edgecnn.data.prepare import prepare_dataset, write_split_manifest
from edgecnn.data.synthetic import make_synthetic_fixture

__all__ = [
    "build_dataloaders",
    "prepare_dataset",
    "write_split_manifest",
    "make_synthetic_fixture",
]
