# `notebooks/` — exploration only

For looking at things: dataset samples, class balance, a confusion matrix you want to poke at, a
quick architecture sanity check.

**Nothing that produces a committed result may live only in a notebook.** Every number and figure
in the report must be reproducible by a script, from a config, by anyone.

## Imports just work here

Because the package is installed with `pip install -e .`, this works from inside `notebooks/` with
no path manipulation:

```python
from edgecnn.config import load_config
from edgecnn.data import build_dataloaders
from edgecnn.models import build_model

cfg  = load_config("configs/experiments/model_b__adam.yaml")
data = build_dataloaders(cfg)
```

**Do not add `sys.path.append('..')`.** If an import fails, you are in the wrong virtual
environment — check with `import sys; print(sys.prefix)` before changing anything. A committed
`sys.path` hack breaks for whoever's folder depth differs from yours.

## Naming

```
m<n>_<topic>.ipynb        m1_dataset_exploration.ipynb
                          m2_architecture_sanity.ipynb
                          m4_confusion_analysis.ipynb
```

The prefix makes ownership obvious and keeps four people's notebooks from colliding.

## Before committing a notebook

**Clear all outputs.** `Kernel -> Restart & Clear Output`.

Notebook outputs embed images as base64, so a notebook with plots can be several megabytes and
produces an unreadable diff on every save. `.ipynb_checkpoints/` is already git-ignored.

## If a notebook becomes useful

Move it into the package. Exploration is fine here; anything the report depends on belongs in
`src/edgecnn/` where it can be tested and re-run.
