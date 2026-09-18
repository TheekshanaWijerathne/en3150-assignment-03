# `configs/contracts/` — the on-disk seam formats

**Custodian: Member 1. PR + 1 review.**

JSON Schemas describing every artifact that crosses between members. These are not documentation
that drifts out of date: `edgecnn.contracts.schema` validates against them on **every write and
every read**, so a malformed artifact never reaches disk and a hand-edited one is caught on load.

| Schema | Artifact | Producer | Consumer |
|---|---|---|---|
| `split_manifest.schema.json` | `data/splits/split_manifest.csv` rows | M1 | everyone |
| `split_meta.schema.json` | `data/splits/split_meta.json` | M1 | everyone |
| `norm_stats.schema.json` | `data/splits/norm_stats.json` | M1 | M1, M4 |
| `history.schema.json` | `results/metrics/<run_id>/history.json` | M3 | M1, M3, M4 |
| `test_metrics.schema.json` | `results/metrics/<run_id>/test_metrics.json` | M1 | M4 |
| `resources.schema.json` | `results/metrics/<run_id>/resources.json` | M1 | M4 |

## Assignment rules encoded as constraints

Several of these are not style checks — they make it structurally impossible to commit a result
that violates the assignment:

| Constraint | Prevents |
|---|---|
| `history.epochs` → `minItems: 20` | committing a run shorter than the mandated 20 epochs |
| `norm_stats.fitted_on` → `const: "train"` | fitting preprocessing on held-out data |
| `test_metrics.split` → `const: "test"` | reporting validation accuracy as the headline number |
| `split_meta.image_size` → max 64 | exceeding the mandated maximum resolution |
| accuracy fields → `[0, 1]` | writing `85.0` where `0.85` was meant |
| `run_id` → pattern | artifacts landing in a directory nobody reads |

## Using them

```python
from edgecnn.contracts import schema, paths

schema.write_json(paths.history_json(run_id), payload, "history")   # validates first
history = schema.read_json(paths.history_json(run_id), "history")   # validates on read
```

A `ContractViolation` means an interface broke. It is not a bug in the code that happened to raise
it — tell whoever owns the producing stage.

## CSV is handled separately

`split_manifest.csv` is not JSON, so its header is checked by
`edgecnn.contracts.schema.MANIFEST_COLUMNS` and each row's values are validated against
`split_manifest.schema.json`. Use `schema.validate_manifest(path)`.
