# `configs/` — settings, and the seams made visible

Every experiment is defined by configuration, not by editing code. No hyperparameter should ever
be hard-coded in a `.py` file.

## The three layers

```
base.yaml                     shared by everything: seed, device, paths, image size
  <- stages/<stage>.yaml      one per pipeline stage; declares its inputs and outputs
      <- experiments/<run>.yaml   the specific run
```

An experiment names the stages it pulls in via `extends:`, then overrides what it needs. Merging is
recursive for mappings; lists are **replaced wholesale**, so an experiment setting
`augmentation: [random_flip]` means exactly that, not "the base list plus this".

```python
from edgecnn.config import load_config
cfg = load_config("configs/experiments/model_b__adam.yaml")
cfg.run_id          # "model_b__adam__seed42"
cfg.section("training")["epochs"]
```

## `stages/` — one per member, each declaring its own I/O

This is the part that makes the four-way split readable. Every stage config has explicit
`inputs:`, `outputs:` and `contract:` blocks, so you can see what a stage consumes and what it
hands on **without reading a line of Python** — and the owner of the next stage downstream can
read it without asking anyone.

| File | Owner | Stage | Section |
|---|---|---|---|
| `data.yaml` | Member 1 | Seam 1 — dataset, splits, normalisation | §1 |
| `models.yaml` | Member 2 | Seam 2 — Model A and Model B architecture | §2 |
| `training.yaml` | Member 3 | Seam 3 — loop, optimizers, the §3 study | §3, §4 |
| `pretrained.yaml` | Member 4 | Seam 2 — SOTA backbones and fine-tuning | §5 |
| `evaluation.yaml` | Members 1 & 4 | Seams 4 and 5 — metrics, benchmark, tables | §4, §6 |

`tests/contracts/` asserts that every stage config declares `stage`, `owner`, `outputs` and
`contract` — so a new stage cannot be added without saying who owns it and what it produces.

## `experiments/` — one file per run

Six files, one per required run. The filename must match the derived `run_id` minus the seed
suffix: `model_b__adam.yaml` produces `model_b__adam__seed42`. A mismatch would scatter one
logical run across two directory names, and a contract test catches it.

## `contracts/` — the on-disk formats

JSON Schemas describing every artifact that crosses between members. These are not documentation
that drifts: `edgecnn.contracts.schema` validates against them on every write and every read.

| Schema | Artifact | Producer |
|---|---|---|
| `split_manifest.schema.json` | `data/splits/split_manifest.csv` rows | Member 1 |
| `split_meta.schema.json` | `data/splits/split_meta.json` | Member 1 |
| `norm_stats.schema.json` | `data/splits/norm_stats.json` | Member 1 |
| `history.schema.json` | `results/metrics/<run_id>/history.json` | Member 3 |
| `test_metrics.schema.json` | `results/metrics/<run_id>/test_metrics.json` | Member 1 |
| `resources.schema.json` | `results/metrics/<run_id>/resources.json` | Member 1 |

Some schema constraints encode assignment rules directly, so a violation cannot be committed:

- `history.epochs` has `minItems: 20` — the mandated minimum epochs.
- `norm_stats.fitted_on` is `const: "train"` — preprocessing cannot be fitted on held-out data.
- `test_metrics.split` is `const: "test"` — headline numbers come from the held-out split.
- `split_meta.image_size` items are capped at 64 — the mandated maximum resolution.

## Rules

- Paths are written relative to the repository root, so the same YAML works from `scripts/`,
  `tests/` and `notebooks/` alike.
- `base.yaml` and `contracts/` are shared files — PR + 1 review.
- `subset_fraction` must be `1.0` in any committed experiment.
