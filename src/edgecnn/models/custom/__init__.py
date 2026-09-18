"""Custom architectures - Assignment Section 2 [20 marks].

Owner: Member 2.

Importing this module registers ``model_a`` and ``model_b`` into the shared
registry. ``edgecnn.models.registry._import_builders`` does that automatically,
so nothing outside this package should import these modules by name.
"""

from edgecnn.models.custom import model_a, model_b  # noqa: F401  (registration side effect)
from edgecnn.models.custom.blocks import (
    conv_bn_act,
    count_parameters,
    depthwise_separable_conv,
    get_activation,
)

__all__ = [
    "conv_bn_act",
    "count_parameters",
    "depthwise_separable_conv",
    "get_activation",
]
