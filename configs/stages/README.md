# `configs/stages/` — one config per pipeline stage

Each file declares, in YAML, what its stage consumes and what it produces. **This is where the
seams are made readable** — you can see a stage's interface without opening a single `.py` file.

| File | Owner | Seam | Section |
|---|---|---|---|
| `data.yaml` | Member 1 | 1 — produces `DataBundle` + the split files | §1 |
| `models.yaml` | Member 2 | 2 — produces `model_a`, `model_b` | §2 |
| `training.yaml` | Member 3 | 3 — produces checkpoints + `history.json` | §3, §4 |
| `pretrained.yaml` | Member 4 | 2 — produces `mobilenet_v2`, `squeezenet1_1` | §5 |
| `evaluation.yaml` | Members 1 & 4 | 4, 5 — produces metrics JSON, then tables | §4, §6 |

## Required structure

Every stage config must have these four top-level keys. `tests/contracts/` asserts it, so a stage
cannot be added without saying who owns it and what it hands on.

```yaml
stage: data
owner: member_1              # who do I ask about this?
produces_for: [member_3]     # who is blocked if it is late?

inputs:      { ... }         # what it consumes
outputs:     { ... }         # what it produces, and where
contract:    { ... }         # what downstream members may assume
```

The `contract:` block names the Python API, the JSON Schemas that enforce the on-disk formats, and
a `guarantees:` list — the promises a consumer is entitled to rely on.

## Editing

Edit only your own stage. Changes to the `contract:` block affect someone else's in-flight work:
PR + 1 review, and say so in the group chat.
