"""Configuration loading and layered merging.

Owner: Member 1.

One entry point, used by every script::

    from edgecnn.config import load_config
    cfg = load_config("configs/experiments/model_b__adam.yaml")
"""

from edgecnn.config.loader import (
    ConfigError,
    load_config,
    load_stage,
    merge,
    resolve_path,
)

__all__ = ["ConfigError", "load_config", "load_stage", "merge", "resolve_path"]
