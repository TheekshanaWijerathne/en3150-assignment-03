## Layer-by-layer parameter tables

One row per layer: its type, kernel, stride, output shape and parameter count, with the formula
that gives the count. For a convolution that is kernel height × width × (input channels / groups)
× filters, plus one bias per filter when there is a bias. A batch-normalisation layer learns a scale
and a shift per channel; its running statistics are buffers, not trainable parameters. Output shapes
come from passing one 64×64 image through the model. The total is checked against the model's own
count, so the table cannot disagree with the model.

```python
import copy

import pandas as pd
from torch import nn


def layer_type(module):
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


def kernel_of(module):
    if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
        size = module.kernel_size
        height, width = (size, size) if isinstance(size, int) else size
        return f"{height}×{width}"
    return ""


def stride_of(module):
    if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
        stride = module.stride
        return str(stride if isinstance(stride, int) else stride[0])
    return ""


def parameter_formula(module, params):
    """How ``params`` is calculated for this layer, such as ``3×3×3×16 = 432``."""
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


def layer_table(model, input_shape=INPUT_SHAPE):
    """Every layer with its output shape, parameters and their formula; the last row is the total."""
    probe = copy.deepcopy(model).cpu().eval()
    layers = [(name, module) for name, module in probe.named_modules()
              if name and next(module.children(), None) is None]
    shapes = {}

    def recorder(name):
        def record(module, inputs, output):
            shapes[name] = tuple(output.shape[1:])
        return record

    hooks = [module.register_forward_hook(recorder(name)) for name, module in layers]
    with torch.no_grad():
        probe(torch.zeros(1, *input_shape))
    for hook in hooks:
        hook.remove()

    rows = []
    for name, module in layers:
        params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        rows.append({
            "layer": name, "type": layer_type(module), "kernel": kernel_of(module),
            "stride": stride_of(module), "output": "×".join(str(s) for s in shapes[name]),
            "params": params, "formula": parameter_formula(module, params),
        })
    total = sum(row["params"] for row in rows)
    if total != count_parameters(model):
        raise ValueError("some parameters are defined outside any layer")
    for row in rows:
        row["share"] = f"{100 * row['params'] / total:.1f}%" if row["params"] else ""
    rows.append({"layer": "total", "type": "", "kernel": "", "stride": "", "output": "",
                 "params": total, "formula": "", "share": "100%"})
    return pd.DataFrame(rows)


def first_activation(model):
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            return type(module).__name__
    return "none"


def width_of(model, layer):
    """Filters of a convolution, or units of a fully connected layer."""
    module = dict(model.named_modules())[layer]
    if isinstance(module, nn.Conv2d):
        return str(module.out_channels)
    if isinstance(module, nn.Linear):
        return str(module.out_features)
    return ""


def architecture_tables(model_a, model_b, input_shape=INPUT_SHAPE):
    """The two Section 2 tables as Markdown, saved to ``results/tables/``.

    - ``custom_architectures.md``: an overview of both models, then each model's
      main layers with kernel size, filters or units, stride and output shape;
    - ``param_breakdown.md``: every layer of each model with its parameter formula.
    """
    models = {"Model A": model_a, "Model B": model_b}
    tables = {name: layer_table(model, input_shape) for name, model in models.items()}
    size = "×".join(str(value) for value in input_shape)

    overview = []
    for name, model in models.items():
        types = tables[name][tables[name]["layer"] != "total"]["type"].tolist()
        overview.append([name, types.count("Conv2d"), types.count("Conv2d depthwise"),
                         types.count("Conv2d pointwise"), types.count("Linear"),
                         first_activation(model), int(tables[name]["params"].iloc[-1]),
                         count_macs(model, input_shape)])
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
            [[row.layer, row.type, row.kernel, width_of(models[name], row.layer), row.stride,
              row.output, int(row.params)] for row in main.itertuples()],
        ))
    custom = "\n".join(sections)

    breakdown = ["# Parameter breakdown\n"]
    for name, table in tables.items():
        breakdown.append(f"\n## {name}\n")
        breakdown.append(markdown_table(
            ["Layer", "Type", "Kernel", "Stride", "Output", "Parameters", "Formula", "Share"],
            [[row.layer, row.type, row.kernel, row.stride, row.output, int(row.params),
              row.formula, row.share] for row in table.itertuples()],
        ))
    breakdown = "\n".join(breakdown)

    write_text(RESULTS_DIR / "tables" / "custom_architectures.md", custom)
    write_text(RESULTS_DIR / "tables" / "param_breakdown.md", breakdown)
    return custom, breakdown
```
