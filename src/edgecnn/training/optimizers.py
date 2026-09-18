"""Optimizer and scheduler factories.       Assignment Section 3 [15 marks]

Owner: Member 3.

    +---------------------------------------------------------------------+
    |  IN   model, cfg.raw["optimizer"] from the experiment config         |
    |  OUT  torch.optim.Optimizer  (+ optional LR scheduler)               |
    +---------------------------------------------------------------------+

The registry key names must match ``history.schema.json``'s optimizer enum
exactly - ``sgd``, ``sgd_momentum``, ``adam``, ``adamw``, ``rmsprop`` - because
the optimizer name is also a component of ``run_id`` and therefore of every
output path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import nn

#: Must stay in sync with the `optimizer` enum in history.schema.json.
SUPPORTED_OPTIMIZERS: tuple[str, ...] = ("sgd", "sgd_momentum", "adam", "adamw", "rmsprop")

SUPPORTED_SCHEDULERS: tuple[str, ...] = ("none", "step", "cosine", "reduce_on_plateau")


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Construct the optimizer named in the config.

    ``sgd`` and ``sgd_momentum`` are both ``torch.optim.SGD``; they are
    separate registry keys because they are separate *experiments*, and the
    run_id has to distinguish their output directories.

    If the model exposes ``param_groups()`` - Member 4's fine-tuned backbones
    do - use it, so the pretrained weights get ``lr_backbone`` and the fresh
    head gets ``lr_head``. Otherwise pass ``model.parameters()`` as one group.
    Custom models need no special handling.

    Raises:
        ValueError: on an unsupported name, listing the valid options.
    """
    raise NotImplementedError("Member 3: implement build_optimizer")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Construct the LR scheduler, or ``None`` for ``name: none``.

    One caveat that matters for Section 3: a scheduler changes the convergence
    curve, so the three optimizer variants must use the *same* scheduler
    setting or the comparison confounds two variables. If cosine annealing is
    used for Adam it must be used for both SGD runs too.

    ``reduce_on_plateau`` steps on a metric rather than on epoch count, so the
    trainer has to call it differently - handle that in the trainer, not here.
    """
    raise NotImplementedError("Member 3: implement build_scheduler")


def describe_optimizer(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Flat hyperparameter snapshot for ``history.json -> hyperparameters``.

    Sections 3 and 4 both require the learning rate and optimizer settings to
    be stated, and reading them back off the optimizer object is more reliable
    than re-reading the config - it records what was actually used, including
    any CLI override.
    """
    raise NotImplementedError("Member 3: implement describe_optimizer")
