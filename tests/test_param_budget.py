"""Model B must stay under 100,000 trainable parameters.

Assignment Section 2 states the cap as a hard requirement, so it is asserted
here rather than read off a printout. Member 2 will change Model B's width and
depth repeatedly while tuning; this is what catches the change that quietly
crosses the line.

The classic way to blow the budget without noticing is a depthwise convolution
built with ``groups=1`` instead of ``groups=in_channels``. It trains fine. It
is simply a normal convolution with several times the parameters, and nothing
else in the pipeline complains.
"""

from __future__ import annotations

import pytest

from edgecnn.contracts.types import INPUT_SHAPE, MODEL_B_PARAM_BUDGET
from tests.conftest import pending

torch = pytest.importorskip("torch", reason="torch not installed")


def _trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_b_is_within_the_parameter_budget() -> None:
    from edgecnn.models.registry import build_model

    with pending("Member 2"):
        model = build_model("model_b", num_classes=10, input_shape=INPUT_SHAPE)

    count = _trainable(model)
    assert count <= MODEL_B_PARAM_BUDGET, (
        f"Model B has {count:,} trainable parameters, budget is "
        f"{MODEL_B_PARAM_BUDGET:,} (over by {count - MODEL_B_PARAM_BUDGET:,}). "
        f"Check that depthwise convolutions use groups=in_channels, and that "
        f"the head uses global average pooling rather than flatten."
    )


def test_model_b_budget_holds_for_the_real_class_count() -> None:
    """EuroSAT has 10 classes; the budget must hold at the deployed width."""
    from edgecnn.models.registry import build_model

    with pending("Member 2"):
        model = build_model("model_b", num_classes=10, input_shape=INPUT_SHAPE)

    assert _trainable(model) <= MODEL_B_PARAM_BUDGET


def test_model_b_actually_uses_its_budget() -> None:
    """A model far under budget is leaving accuracy on the table.

    Not a hard requirement - but if Model B sits at 8k parameters, Member 2
    should ask whether a wider network would score better while still fitting.
    Warns rather than fails.
    """
    from edgecnn.models.registry import build_model

    with pending("Member 2"):
        model = build_model("model_b", num_classes=10, input_shape=INPUT_SHAPE)

    count = _trainable(model)
    if count < MODEL_B_PARAM_BUDGET * 0.4:
        pytest.skip(
            f"Model B uses only {count:,} of {MODEL_B_PARAM_BUDGET:,} parameters "
            f"({count / MODEL_B_PARAM_BUDGET:.0%}). Consider widening it - the "
            f"budget is a ceiling, not a target to undershoot."
        )


def test_model_a_is_larger_than_model_b() -> None:
    """The comparison only means something if A is the unconstrained baseline."""
    from edgecnn.models.registry import build_model

    with pending("Member 2"):
        a = _trainable(build_model("model_a", num_classes=10, input_shape=INPUT_SHAPE))
        b = _trainable(build_model("model_b", num_classes=10, input_shape=INPUT_SHAPE))

    assert a > b, (
        f"Model A ({a:,}) should have more parameters than Model B ({b:,}) - "
        f"otherwise there is no trade-off to discuss in Section 4."
    )
