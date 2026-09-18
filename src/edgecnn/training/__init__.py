"""SEAM 3 - training loop, optimizers, and the Section 3 study.

Owner: Member 3.

    from edgecnn.training import Trainer, build_optimizer
"""

from edgecnn.training.callbacks import CheckpointManager, EarlyStopping
from edgecnn.training.optimizer_study import run_optimizer_study, summarize_study
from edgecnn.training.optimizers import (
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SCHEDULERS,
    build_optimizer,
    build_scheduler,
    describe_optimizer,
)
from edgecnn.training.trainer import Trainer

__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_SCHEDULERS",
    "Trainer",
    "build_optimizer",
    "build_scheduler",
    "describe_optimizer",
    "run_optimizer_study",
    "summarize_study",
]
