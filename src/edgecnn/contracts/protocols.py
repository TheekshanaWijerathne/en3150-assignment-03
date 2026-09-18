"""Structural interfaces between members - what each side may assume.

    +---------------------------------------------------------------------+
    |  FROZEN INTERFACE.  PR + 1 review to change.                         |
    +---------------------------------------------------------------------+

These are :class:`typing.Protocol` classes, so nothing needs to inherit from
them. A class satisfies a protocol by having the right methods. That matters
here because ``torchvision.models.mobilenet_v2`` is not ours to subclass, yet
it must satisfy :class:`ClassifierModel` exactly like Member 2's hand-written
Model B does.

Read the protocol that names you as *producer* before you write code, and the
one that names you as *consumer* before you make an assumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from torch import nn

    from edgecnn.contracts.types import DataBundle, EvalResult, ResolvedConfig, TrainResult


@runtime_checkable
class ClassifierModel(Protocol):
    """SEAM 2 - produced by Members 2 and 4, consumed by Members 1 and 3.

    Every model reachable through ``edgecnn.models.registry.build_model`` must
    satisfy this, whether it was written from scratch or pulled from
    torchvision.

    The forward contract::

        in  : torch.float32, shape (B, 3, 64, 64), normalised
        out : torch.float32, shape (B, num_classes)   <-- RAW LOGITS

    Raw logits, not probabilities. Three reasons, and the third is worth marks:

    1. ``nn.CrossEntropyLoss`` applies log-softmax internally. A model that
       also softmaxes trains on a double-softmaxed signal and quietly
       underperforms - it does not crash, so nobody notices for a week.
    2. Member 1's metrics code takes ``argmax`` over logits, which is
       monotonic under softmax, so both work - but only one is correct for
       the loss.
    3. Section 2 asks for hardware-aware activation justification. Softmax is
       an exponential per class per inference; on a microcontroller without an
       FPU that is real cost, and it is pure overhead when you only need the
       arg-max label. Keeping it out of the model is part of that argument.
    """

    num_classes: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a normalised image batch to raw class logits."""
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class ModelBuilder(Protocol):
    """Signature every entry in the model registry must have.

    Members 2 and 4 register builders; Member 3 only ever calls
    ``build_model(name, ...)`` and never imports a model module directly. That
    indirection is what lets the trainer stay identical across Model A,
    Model B, MobileNetV2 and SqueezeNet - which is what makes the Section 6
    comparison fair.
    """

    def __call__(
        self,
        num_classes: int,
        input_shape: tuple[int, int, int],
        **overrides: object,
    ) -> nn.Module: ...


@runtime_checkable
class DataProvider(Protocol):
    """SEAM 1 - produced by Member 1, consumed by Members 2, 3 and 4."""

    def __call__(self, cfg: ResolvedConfig) -> DataBundle: ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """SEAM 3 - produced by Member 3, consumed by Members 1 and 4.

    ``fit`` owns three side effects, and they are part of the contract:

    * writes ``artifacts/checkpoints/<run_id>/best.pt`` with exactly the keys
      in :data:`edgecnn.contracts.types.CHECKPOINT_KEYS`;
    * writes ``results/metrics/<run_id>/history.json`` validated against
      ``configs/contracts/history.schema.json``;
    * leaves the test split untouched. The trainer must never construct a
      loader over ``DataBundle.test``. Section 4 asks for an honest held-out
      number, and touching it invalidates every result in the report.
    """

    def fit(self, model: nn.Module, data: DataBundle, cfg: ResolvedConfig) -> TrainResult: ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """SEAM 4 - produced by Member 1, consumed by Member 4.

    Called once per run, after training has finished, against
    ``DataBundle.test``. Writes ``test_metrics.json`` and ``resources.json``.
    """

    def evaluate(
        self,
        model: nn.Module,
        data: DataBundle,
        cfg: ResolvedConfig,
    ) -> EvalResult: ...
