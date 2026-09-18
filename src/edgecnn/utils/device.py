"""Device selection and hardware identification.

Owner: Member 1.

Section 4 requires the evaluation hardware to be reported, and an epoch-time
or latency column without a device string is meaningless. :func:`describe_device`
produces the string that goes into ``resources.json`` and the report table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a config device string into a concrete ``torch.device``.

    Args:
        spec: ``"auto"`` (cuda if available, else cpu), ``"cpu"``, ``"cuda"``
            or ``"cuda:N"``.

    Raises:
        RuntimeError: if cuda is requested explicitly but unavailable. Failing
            loudly beats silently falling back to CPU and then reporting
            epoch times 30x slower than a teammate's for the same model.
    """
    raise NotImplementedError("Member 1: implement resolve_device")


def describe_device(device: torch.device) -> str:
    """Human-readable hardware string for ``resources.json`` and the report.

    Examples: ``"cpu (11th Gen Intel Core i7-11800H)"``,
    ``"cuda:0 (NVIDIA GeForce RTX 3050 Laptop GPU)"``.

    Use ``torch.cuda.get_device_name`` for GPUs and ``platform.processor``
    for CPUs, falling back to a plain ``"cpu"`` rather than raising - a
    missing CPU model name should not abort a benchmark.
    """
    raise NotImplementedError("Member 1: implement describe_device")
