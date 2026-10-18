## Setup

Folders, the random seed, the device, and small helpers for files and figures. Every notebook in
this folder starts with the same setup.

```python
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

# Everything is read from and written to folders next to this notebook.
DATA_DIR = Path("data")                # the downloaded dataset and a decoded-pixel cache
RESULTS_DIR = Path("results")          # metrics, figures and tables
CHECKPOINT_DIR = Path("checkpoints")   # trained weights

SEED = 42            # fixes the split, the initial weights, augmentation and batch order
NUM_CLASSES = 10     # EuroSAT land-cover classes
INPUT_SHAPE = (3, 64, 64)

# cuBLAS needs this workspace setting for repeatable matrix products on a GPU.
CUBLAS_WORKSPACE = ":4096:8"


def seed_everything(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch, so a run gives the same numbers every time.

    With ``deterministic=True``, cuDNN uses deterministic algorithms and stops
    auto-tuning, and PyTorch prefers deterministic implementations. This costs a
    little speed. Without it, two runs of the same experiment can differ by a few
    tenths of a percent, the same size as some differences between the models.
    A few GPU operations have no deterministic version; PyTorch then warns
    instead of stopping.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU and every GPU
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def cpu_name():
    """The processor's model name, or an empty string if the operating system won't say."""
    name = ""
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1]
                        break
        elif sys.platform == "darwin":
            name = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout
    except Exception:
        name = ""
    name = name or platform.processor() or ""
    name = re.sub(r"\((R|TM|C)\)", "", name, flags=re.IGNORECASE)  # trademark signs
    return " ".join(name.split())


def describe_device(device):
    """A readable name for a device, such as ``cuda:0 (Tesla T4)``.

    Every timing is recorded together with this name: a time means little
    without the hardware it was measured on.
    """
    device = torch.device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    name = cpu_name()
    return f"cpu ({name})" if name else "cpu"


#: Model name -> function building it; filled in by the model definitions below.
MODEL_BUILDERS = {}


def build_model(name, num_classes=NUM_CLASSES):
    """Seed PyTorch, then build ``name``: the same seed always gives the same initial weights.

    The three optimizer runs rely on this to start from identical weights, so
    that their comparison measures the optimizer and not the initialisation.
    """
    torch.manual_seed(SEED)
    return MODEL_BUILDERS[name](num_classes)


def count_parameters(model, trainable_only=True):
    """Number of parameters; frozen ones are left out when ``trainable_only``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def model_size_kb(model):
    """Size of the saved weights (``state_dict``) in KB, where 1 KB = 1,024 bytes.

    Saving the weights, as a device would store them, rather than counting 4
    bytes per parameter, also counts buffers such as batch-norm statistics.
    """
    import io

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Python       {platform.python_version()} ({platform.system()})")
print(f"PyTorch      {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")
print(f"NumPy        {np.__version__}")
print(f"Device       {describe_device(DEVICE)}")
```

Figures are shown inline at a small size and saved at print quality. Each model and each optimizer
keeps one colour and one name in every figure and table.

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 100,            # inline figures: small, so the notebook stays light
    "savefig.dpi": 200,           # saved figures: print quality
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
})

MODEL_LABELS = {
    "model_a": "Model A",
    "model_b": "Model B",
    "mobilenet_v2": "MobileNetV2",
    "squeezenet1_1": "SqueezeNet 1.1",
}
MODEL_COLORS = {
    "model_a": "#4C72B0",
    "model_b": "#DD8452",
    "mobilenet_v2": "#55A868",
    "squeezenet1_1": "#C44E52",
}
OPTIMIZER_LABELS = {"sgd": "SGD", "sgd_momentum": "SGD + momentum", "adam": "Adam"}
OPTIMIZER_COLORS = {"sgd": "#8172B3", "sgd_momentum": "#937860", "adam": "#DA8BC3"}


def save_figure(fig, path):
    """Save a figure as a PNG, creating its folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
```

Results are saved as JSON and Markdown, so the tables can be rebuilt without retraining.

```python
def write_json(path, payload):
    """Write ``payload`` as indented JSON, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def format_cell(value, spec=None):
    """One table cell: thousands separators for whole numbers, ``spec`` for decimals."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if spec is not None and isinstance(value, (int, float)):
        return format(value, spec)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.1f}" if abs(value) >= 1000 else f"{value:.4g}"
    return str(value).replace("|", "\\|")


def markdown_table(headers, rows, caption=None, formats=None):
    """A Markdown table. Number columns are right-aligned; ``formats`` fixes their decimals."""
    formats = formats or [None] * len(headers)

    def numeric(column):
        cells = [row[column] for row in rows if row[column] is not None]
        return bool(cells) and all(
            isinstance(cell, (int, float)) and not isinstance(cell, bool) for cell in cells
        )

    lines = [f"*{caption}*", ""] if caption else []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---:" if numeric(c) else "---" for c in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(format_cell(v, s) for v, s in zip(row, formats)) + " |")
    return "\n".join(lines) + "\n"
```
