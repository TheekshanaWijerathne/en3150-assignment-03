"""Shared helpers: seeding, device selection, logging, small IO.

Owner: Member 1.  Used by all four members.

Deliberately dependency-light and side-effect free on import, so any member
can pull one helper without dragging in a dataset or a model.
"""

from edgecnn.utils.device import describe_device, resolve_device
from edgecnn.utils.logging import get_logger, setup_logging
from edgecnn.utils.seed import seed_everything, worker_init_fn

__all__ = [
    "describe_device",
    "resolve_device",
    "get_logger",
    "setup_logging",
    "seed_everything",
    "worker_init_fn",
]
