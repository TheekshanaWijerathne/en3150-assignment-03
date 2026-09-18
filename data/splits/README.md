# `data/splits/` — the reproducibility contract

**COMMITTED.** Owner: Member 1. Consumed by all four members. This is SEAM 1.

| File | Contents |
|---|---|
| `split_manifest.csv` | `relative_path, label_index, label_name, split` — one row per image |
| `split_meta.json` | class names (frozen ordering), per-split counts, seed, manifest SHA-256 |
| `norm_stats.json` | per-channel mean and std, fitted on the **train split only** |

## Why these three files are committed

They are what guarantees Model A, Model B, MobileNetV2 and SqueezeNet saw byte-identical data. An
accuracy difference between them is then attributable to the architecture rather than to a lucky
split. A few kilobytes of text, and the entire comparison rests on them.

## Rules

- Drawn **once**, stratified 70/15/15, `seed: 42`. `prepare_data.py` needs `--force` to overwrite.
- Regenerating invalidates every result already in `results/metrics/`.
- `split_meta.json` carries the manifest's SHA-256 — if it stops matching, the committed results
  are stale.
- `class_names` ordering is frozen. Every confusion matrix and per-class metric is indexed by it.
- Forward slashes in `relative_path`; the team runs Windows and Linux.
- Never hand-edit. `edgecnn.contracts.schema.validate_manifest` will reject it, and
  `pytest tests/contracts` will go red.
