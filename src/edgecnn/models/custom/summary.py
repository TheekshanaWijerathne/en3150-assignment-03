"""Architecture tables - the Section 2 parameter derivation, written out per layer.

Section 2 asks for each network's parameters - kernel sizes, filters, fully
connected units - to be detailed, and the total trainable parameters to be
calculated. These tables do that from the built models themselves: every layer
with its output shape, its parameter count and the formula that gives it, so the
hand derivation in the report can be checked line by line.

Team notes:
Owner: Member 2. Called from notebooks/02_custom_architectures.ipynb.

    IN   model_a, model_b (built with build_model_from_config)
    OUT  per-layer tables - always returned
         + results/tables/custom_architectures.md  } write=True only
         + results/tables/param_breakdown.md       }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from torch import nn


def layer_table(model: nn.Module, input_shape: tuple[int, int, int]) -> pd.DataFrame:
    """One row per layer: type, kernel, stride, output shape, parameters and their formula.

    Output shapes come from running one ``(1, *input_shape)`` image through a
    copy of the model. Each parameter count sits next to the formula that gives
    it - for a convolution, kernel height x width x (input channels / groups) x
    filters, plus one bias per filter when there is one - and ``share`` is the
    layer's percentage of all trainable parameters, which shows at a glance
    where the parameters go. The last row is the total.

    Batch-norm layers count their learned scale and shift (2 per channel); their
    running statistics are buffers, not trainable parameters.

    Returns:
        A table with the columns ``layer, type, kernel, stride, output, params,
        formula, share``.

    Raises:
        ValueError: if the layers' parameters do not add up to the model's
            trainable total - a parameter defined outside any layer.
    """
    import copy

    import pandas as pd
    import torch

    probe = copy.deepcopy(model).cpu().eval()
    layers = [
        (name, module) for name, module in probe.named_modules()
        if name and next(module.children(), None) is None
    ]
    shapes: dict[str, tuple[int, ...]] = {}

    def recorder(name: str) -> Any:
        def record(module: Any, inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                shapes[name] = tuple(output.shape[1:])

        return record

    hooks = [module.register_forward_hook(recorder(name)) for name, module in layers]
    try:
        with torch.no_grad():
            probe(torch.zeros(1, *input_shape))
    finally:
        for hook in hooks:
            hook.remove()

    rows = []
    for name, module in layers:
        params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        rows.append({
            "layer": name,
            "type": _layer_type(module),
            "kernel": _kernel(module),
            "stride": _stride(module),
            "output": "×".join(str(size) for size in shapes[name]) if name in shapes else "",
            "params": params,
            "formula": _formula(module, params),
        })

    total = sum(row["params"] for row in rows)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total != trainable:
        raise ValueError(
            f"the layers hold {total:,} trainable parameters but the model has {trainable:,}: "
            "some parameters are defined outside any layer"
        )
    for row in rows:
        row["share"] = f"{100 * row['params'] / total:.1f}%" if row["params"] else ""
    rows.append({
        "layer": "total", "type": "", "kernel": "", "stride": "", "output": "",
        "params": total, "formula": "", "share": "100%",
    })
    return pd.DataFrame(rows)


def architecture_tables(
    model_a: nn.Module,
    model_b: nn.Module,
    input_shape: tuple[int, int, int],
    write: bool = False,
) -> dict[str, str]:
    """Both Section 2 tables, as Markdown; with ``write=True``, also saved.

    * ``custom_architectures`` - an overview of both models (layer counts,
      activation, trainable parameters, MACs for one image), then each model's
      main layers: kernel size, filters or units, stride and output shape.
    * ``param_breakdown`` - every layer of each model with its parameter
      formula, as produced by :func:`layer_table`.

    Args:
        model_a: Model A, as built for training.
        model_b: Model B, as built for training.
        input_shape: ``(channels, height, width)`` of one image.
        write: Save to ``results/tables/custom_architectures.md`` and
            ``results/tables/param_breakdown.md``. Pass ``cfg.is_official``.

    Returns:
        ``{"custom_architectures": markdown, "param_breakdown": markdown}``.
    """
    from edgecnn.contracts import paths
    from edgecnn.evaluation.benchmark import count_macs
    from edgecnn.utils.io import markdown_table

    models = {"Model A": model_a, "Model B": model_b}
    tables = {name: layer_table(model, input_shape) for name, model in models.items()}
    size = "×".join(str(value) for value in input_shape)

    overview = []
    for name, model in models.items():
        layers = tables[name][tables[name]["layer"] != "total"]
        types = layers["type"].tolist()
        overview.append([
            name,
            types.count("Conv2d"),
            types.count("Conv2d depthwise"),
            types.count("Conv2d pointwise"),
            types.count("Linear"),
            _activation(model),
            int(tables[name]["params"].iloc[-1]),
            count_macs(model, input_shape),
        ])
    sections = [
        "# Custom architectures\n",
        markdown_table(
            ["Model", "Standard conv", "Depthwise conv", "Pointwise conv", "Fully connected",
             "Activation", "Trainable parameters", "MACs per image"],
            overview,
            caption=f"Both models take a {size} image and return one raw score per class.",
        ),
    ]
    for name in models:
        main = tables[name][tables[name]["type"].str.startswith(("Conv2d", "Linear"))]
        sections.append(f"\n## {name} - main layers\n")
        sections.append(markdown_table(
            ["Layer", "Type", "Kernel", "Filters / units", "Stride", "Output", "Parameters"],
            [
                [row.layer, row.type, row.kernel, _width(models[name], row.layer), row.stride,
                 row.output, int(row.params)]
                for row in main.itertuples()
            ],
        ))
    custom = "\n".join(sections)

    breakdown_sections = ["# Parameter breakdown\n"]
    for name, table in tables.items():
        breakdown_sections.append(f"\n## {name}\n")
        breakdown_sections.append(markdown_table(
            ["Layer", "Type", "Kernel", "Stride", "Output", "Parameters", "Formula", "Share"],
            [
                [row.layer, row.type, row.kernel, row.stride, row.output, int(row.params),
                 row.formula, row.share]
                for row in table.itertuples()
            ],
        ))
    breakdown = "\n".join(breakdown_sections)

    if write:
        for path, text in (
            (paths.CUSTOM_ARCHITECTURES_TABLE, custom),
            (paths.PARAM_BREAKDOWN_TABLE, breakdown),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    return {"custom_architectures": custom, "param_breakdown": breakdown}


# --- internals --------------------------------------------------------------------


def _layer_type(module: nn.Module) -> str:
    from torch import nn

    if isinstance(module, nn.Conv2d):
        if module.groups > 1 and module.groups == module.in_channels:
            return "Conv2d depthwise"
        if module.kernel_size == (1, 1):
            return "Conv2d pointwise"
        return "Conv2d"
    if isinstance(module, nn.AdaptiveAvgPool2d) and module.output_size in (1, (1, 1)):
        return "Global average pool"
    if isinstance(module, nn.Dropout):
        return f"Dropout (p={module.p})"
    return type(module).__name__


def _kernel(module: nn.Module) -> str:
    from torch import nn

    if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
        size = module.kernel_size
        height, width = (size, size) if isinstance(size, int) else size
        return f"{height}×{width}"
    return ""


def _stride(module: nn.Module) -> str:
    from torch import nn

    if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
        stride = module.stride
        return str(stride if isinstance(stride, int) else stride[0])
    return ""


def _formula(module: nn.Module, params: int) -> str:
    """How ``params`` is calculated for this layer, e.g. ``3×3×3×16 = 432``."""
    from torch import nn

    if not params:
        return ""
    if isinstance(module, nn.Conv2d):
        height, width = module.kernel_size
        weights = f"{height}×{width}×{module.in_channels // module.groups}×{module.out_channels}"
        bias = f" + {module.out_channels}" if module.bias is not None else ""
        return f"{weights}{bias} = {params:,}"
    if isinstance(module, nn.BatchNorm2d):
        return f"2×{module.num_features} = {params:,}"
    if isinstance(module, nn.Linear):
        bias = f" + {module.out_features}" if module.bias is not None else ""
        return f"{module.in_features}×{module.out_features}{bias} = {params:,}"
    return f"{params:,}"


def _width(model: nn.Module, layer: str) -> str:
    """Filters of a convolution, or units of a fully connected layer."""
    from torch import nn

    module = dict(model.named_modules())[layer]
    if isinstance(module, nn.Conv2d):
        return str(module.out_channels)
    if isinstance(module, nn.Linear):
        return str(module.out_features)
    return ""


def _activation(model: nn.Module) -> str:
    """The name of the first activation in the model, e.g. ``ReLU6``."""
    from torch import nn

    kinds = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.Hardswish, nn.SiLU, nn.GELU, nn.Sigmoid, nn.Tanh)
    for module in model.modules():
        if isinstance(module, kinds):
            return type(module).__name__
    return "none"
