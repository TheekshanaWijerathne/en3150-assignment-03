"""Model definitions and the registry that hides them from the trainer.

    custom/      Member 2 - Model A (standard CNN), Model B (<=100k params)
    pretrained/  Member 4 - MobileNetV2, SqueezeNet 1.1
    registry.py  shared   - build_model(), the Seam 2 entry point

Member 3 imports only ``build_model``. That is deliberate: see registry.py.
"""

from edgecnn.models.registry import (
    REQUIRED_KEYS,
    ModelNotFound,
    available_models,
    build_model,
    register,
)

__all__ = [
    "REQUIRED_KEYS",
    "ModelNotFound",
    "available_models",
    "build_model",
    "register",
]
