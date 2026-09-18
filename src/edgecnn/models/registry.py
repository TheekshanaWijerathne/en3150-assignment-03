"""SEAM 2 - the model registry.

    +---------------------------------------------------------------------+
    |  SHARED FILE.  Member 2 owns it; Member 4 appends pretrained         |
    |  entries.  PR + 1 review.  Coordinate before editing - this is the   |
    |  one file two members write to.                                      |
    +---------------------------------------------------------------------+

Why an indirection instead of just importing the model
-------------------------------------------------------
Member 3's trainer must be identical for Model A, Model B, MobileNetV2 and
SqueezeNet. If the trainer imported model modules directly it would grow a
branch per architecture, and the four models would drift apart in ways that
make the Section 6 table impossible to defend - different loss reductions,
different eval-mode handling, different timing points.

So the trainer knows exactly one thing: ``build_model(name, ...)``. Members 2
and 4 register builders; the trainer never learns which is which.

    +---------------------------------------------------------------------+
    |  IN   name         registry key, e.g. "model_b"                      |
    |       num_classes  from DataBundle.num_classes                       |
    |       input_shape  from DataBundle.input_shape, i.e. (3, 64, 64)     |
    |       **overrides  architecture knobs from configs/stages/models.yaml|
    |                                                                      |
    |  OUT  torch.nn.Module satisfying contracts.protocols.ClassifierModel |
    |       forward: (B,3,64,64) float32 -> (B,num_classes) float32 LOGITS |
    +---------------------------------------------------------------------+
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from torch import nn

#: Registry key -> builder. Populated by the ``@register`` decorator when
#: ``edgecnn.models.custom`` and ``edgecnn.models.pretrained`` are imported.
_REGISTRY: dict[str, Callable[..., nn.Module]] = {}

#: Keys the assignment requires. Checked by tests/contracts/test_registry.py so
#: a missing model is a red test, not a surprise during report week.
REQUIRED_KEYS: tuple[str, ...] = (
    "model_a",         # Member 2 - Section 2 standard CNN
    "model_b",         # Member 2 - Section 2 depthwise-separable, <=100k params
    "mobilenet_v2",    # Member 4 - Section 5 SOTA backbone 1
    "squeezenet1_1",   # Member 4 - Section 5 SOTA backbone 2
)


class ModelNotFound(KeyError):
    """Raised when a config names a model key nobody registered."""


def register(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator registering a builder under ``name``.

    Usage, in ``models/custom/model_b.py``::

        @register("model_b")
        def build_model_b(num_classes, input_shape, **overrides) -> nn.Module:
            ...

    Raises:
        ValueError: on a duplicate key. Two members silently registering the
            same name would make which model you got depend on import order.
    """

    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _REGISTRY:
            raise ValueError(
                f"model key {name!r} is already registered by "
                f"{_REGISTRY[name].__module__}. Pick a different key and tell the team."
            )
        _REGISTRY[name] = builder
        return builder

    return decorator


def build_model(
    name: str,
    num_classes: int,
    input_shape: tuple[int, int, int] = (3, 64, 64),
    **overrides: object,
) -> nn.Module:
    """Construct a registered model. The only way Member 3 gets a model.

    Every returned module must satisfy
    :class:`edgecnn.contracts.protocols.ClassifierModel`: raw logits out, no
    softmax, ``.num_classes`` exposed.

    Raises:
        ModelNotFound: listing the available keys, so a typo in a config is a
            one-line fix rather than a hunt.
    """
    _import_builders()
    if name not in _REGISTRY:
        raise ModelNotFound(
            f"no model registered as {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](num_classes=num_classes, input_shape=input_shape, **overrides)


def available_models() -> list[str]:
    """Every registered key, sorted."""
    _import_builders()
    return sorted(_REGISTRY)


def _import_builders() -> None:
    """Import the model subpackages so their decorators run.

    Lazy on purpose: importing the registry must not pull torchvision (and a
    possible weights download) into a process that only wanted to read a
    config or validate a schema.
    """
    import edgecnn.models.custom  # noqa: F401
    import edgecnn.models.pretrained  # noqa: F401
