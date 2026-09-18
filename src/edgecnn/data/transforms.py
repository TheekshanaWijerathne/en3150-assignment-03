"""Image transform pipelines.

Owner: Member 1.

One rule, and it is the difference between an honest and a dishonest result:
augmentation applies to the **train** split only. Val and test get resize plus
normalisation and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchvision.transforms import v2

#: Augmentations the config may name, mapped to torchvision v2 transforms.
#: Vertical flip and 90-degree rotation are valid here because EuroSAT is
#: overhead satellite imagery with no canonical "up" - the same augmentation
#: would be wrong for, say, street-level photographs.
SUPPORTED_AUGMENTATIONS: tuple[str, ...] = (
    "random_horizontal_flip",
    "random_vertical_flip",
    "random_rotation_90",
    "random_resized_crop",
    "color_jitter",
)


def build_transform(
    split: str,
    norm_mean: tuple[float, ...],
    norm_std: tuple[float, ...],
    augmentations: list[str] | None = None,
    image_size: tuple[int, int] = (64, 64),
    **options: Any,
) -> v2.Transform:
    """Compose the transform pipeline for one split.

    Output must always be ``float32`` in shape ``(3, 64, 64)``, normalised
    with the supplied statistics - that is the Seam 1 batch contract.

    Args:
        split: ``train``, ``val`` or ``test``.
        norm_mean: Per-channel mean, from ``norm_stats.json``.
        norm_std: Per-channel std, from ``norm_stats.json``.
        augmentations: Names from :data:`SUPPORTED_AUGMENTATIONS`. Ignored
            with a warning for non-train splits - silently accepting them
            would let a config quietly corrupt the test numbers.
        image_size: Target size; EuroSAT is already 64x64.

    Raises:
        ValueError: on an unknown augmentation name. Fail loudly - a typo that
            silently disables augmentation would show up only as an
            unexplained accuracy gap a week later.
    """
    raise NotImplementedError("Member 1: implement build_transform")
