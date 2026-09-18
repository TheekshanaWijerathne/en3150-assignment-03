"""Reproducibility helpers.

Owner: Member 1.  Called at the top of every script, before anything else.

The assignment asks for random seeds to be recorded, and the four-way split
makes it load-bearing for a different reason: if Member 3's Adam run and
Member 3's SGD run do not start from the same initial weights, the Section 3
comparison measures initialisation noise alongside the optimizer.
"""

from __future__ import annotations


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA).

    Args:
        seed: The seed. Every committed run uses 42.
        deterministic: Set ``cudnn.deterministic=True`` and
            ``cudnn.benchmark=False``. Costs some throughput and is worth it -
            without it two runs of the same config differ by a few tenths of a
            percent, which is the same order as the Model A / Model B gap the
            report is trying to explain.

    Also call ``torch.use_deterministic_algorithms(True)`` where supported, and
    note in the report that full CUDA determinism additionally needs
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the environment.
    """
    raise NotImplementedError("Member 1: implement seed_everything")


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically.

    Pass to ``DataLoader(worker_init_fn=...)``. Without it, workers inherit
    non-deterministic seeds and augmentation differs run to run even when
    :func:`seed_everything` was called.
    """
    raise NotImplementedError("Member 1: implement worker_init_fn")
