# `artifacts/` — model weights

**Git-ignored.** Everything here is large and fully regenerable from `configs/` plus committed
`data/splits/`.

| Folder | Contents | Producer |
|---|---|---|
| `checkpoints/<run_id>/` | `best.pt`, `last.pt` | Member 3 |
| `exports/` | Optional deployment formats (ONNX, TFLite) | Member 4, if attempted |

## Checkpoint contract

`best.pt` is written by Member 3's trainer and read by Member 1's evaluator, so its key set is
fixed and asserted by `tests/contracts/`:

```python
{
    "model_name": str,        # registry key, for rebuilding the architecture
    "state_dict": dict,
    "epoch": int,
    "val_acc": float,
    "class_names": list[str], # so the evaluator needs no config to label outputs
    "input_shape": tuple,
    "seed": int,
    "config_snapshot": dict,  # what actually produced this, including CLI overrides
}
```

`class_names` and `input_shape` are mandatory so that **a checkpoint is self-describing** — Member
1 can evaluate a run without re-reading the config that produced it, and without guessing whether
the label ordering matches.

`best` is selected on **validation** accuracy, never test. The test split is not visible to
training code at all.

## Sharing weights between members

Do not commit them, and do not email them. Anyone can regenerate a checkpoint:

```powershell
python scripts/train.py --config configs/experiments/model_b__adam.yaml
```

What *does* need sharing is the metrics JSON, which is why `results/metrics/` is committed. If a
checkpoint genuinely must be shared — a run that took hours on the one GPU machine — use a shared
drive and note the `run_id`, never git.

## `exports/`

Not required by the assignment. If you do export an ONNX or TFLite model, the size comparison
against `model_size_kb` makes a good §6 aside — int8 quantisation typically cuts size ~4x, and
ReLU6 is chosen partly to make that quantisation well-behaved.
