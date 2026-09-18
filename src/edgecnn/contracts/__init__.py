"""The frozen interface layer - import everything cross-member from here.

    from edgecnn.contracts import DataBundle, paths, schema

Nothing in this subpackage imports torch at module level, so it stays cheap
and always importable. See ``types.py``, ``protocols.py``, ``paths.py`` and
``schema.py`` for the four faces of the contract, and this folder's README for
the rules about changing them.
"""

from edgecnn.contracts import paths, protocols, schema
from edgecnn.contracts.protocols import (
    ClassifierModel,
    DataProvider,
    EvaluatorProtocol,
    ModelBuilder,
    TrainerProtocol,
)
from edgecnn.contracts.schema import ContractViolation
from edgecnn.contracts.types import (
    CHECKPOINT_KEYS,
    DEFAULT_SEED,
    IMAGE_SIZE,
    INPUT_SHAPE,
    MODEL_B_PARAM_BUDGET,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    DataBundle,
    EpochRecord,
    EvalResult,
    ResolvedConfig,
    ResourceProfile,
    TrainResult,
)

__all__ = [
    # modules
    "paths",
    "protocols",
    "schema",
    # types
    "DataBundle",
    "EpochRecord",
    "EvalResult",
    "ResolvedConfig",
    "ResourceProfile",
    "TrainResult",
    # protocols
    "ClassifierModel",
    "DataProvider",
    "EvaluatorProtocol",
    "ModelBuilder",
    "TrainerProtocol",
    # constants
    "CHECKPOINT_KEYS",
    "DEFAULT_SEED",
    "IMAGE_SIZE",
    "INPUT_SHAPE",
    "MODEL_B_PARAM_BUDGET",
    "SPLIT_FRACTIONS",
    "SPLIT_NAMES",
    # errors
    "ContractViolation",
]
