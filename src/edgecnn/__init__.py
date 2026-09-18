"""edgecnn - EN3150 Assignment 03: resource-constrained CNNs for edge devices.

Layout, and who owns what (full table in the root README):

    contracts/   FROZEN seam definitions            shared, custodian Member 1
    config/      YAML merge: base <- stage <- run   Member 1
    data/        EuroSAT -> splits -> DataLoaders   Member 1
    models/      registry + custom/ + pretrained/   Member 2 / Member 4
    training/    trainer, optimizers, study         Member 3
    evaluation/  metrics, benchmark, curves, tables Members 1, 3, 4
    utils/       seeding, device, logging           Member 1

Only ``edgecnn.contracts`` is imported eagerly - it is pure declarations with
no torch dependency. Everything else is imported on demand so that a member
working on data does not pay for matplotlib, and a contract test does not need
a trained model.
"""

__version__ = "0.1.0"

from edgecnn import contracts

__all__ = ["contracts", "__version__"]
