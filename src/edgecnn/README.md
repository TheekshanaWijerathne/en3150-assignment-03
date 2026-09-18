# `edgecnn`

The project package. Import it as `edgecnn` from anywhere after `pip install -e .`.

| Subpackage | Owner | Role |
|---|---|---|
| [`contracts/`](contracts/README.md) | M1 (custodian) | **FROZEN** — types, protocols, paths, schema validation |
| [`config/`](config/README.md) | M1 | layered YAML: base ← stage ← experiment |
| [`data/`](data/README.md) | M1 | Seam 1 — EuroSAT, splits, DataLoaders |
| [`models/`](models/README.md) | M2 + M4 | Seam 2 — registry, custom models, pretrained backbones |
| [`training/`](training/README.md) | M3 | Seam 3 — trainer, optimizers, the §3 study |
| [`evaluation/`](evaluation/README.md) | M1, M3, M4 | Seams 4 and 5 — metrics, benchmark, curves, tables |
| [`utils/`](utils/README.md) | M1 | seeding, device, logging, small IO |

## Import discipline

Only `edgecnn.contracts` is imported eagerly by `__init__.py` — it is pure declarations with no
torch dependency, so it stays cheap and always importable. Everything else is imported on demand,
so a member working on data does not pay for matplotlib, and a contract test does not need a model.

## The one rule that keeps the split working

Members talk through `contracts/`, never by importing each other's modules.

In particular, **the trainer must never import a model module.** It calls `build_model(name, ...)`
and does not learn which architecture it received. That indirection is what keeps the training loop
identical across Model A, Model B, MobileNetV2 and SqueezeNet — which is what makes the §6
comparison fair.
