# `artifacts/checkpoints/` — trained weights

**Git-ignored** (except this file). Producer: Member 3. Consumer: Member 1.

```
artifacts/checkpoints/<run_id>/best.pt     highest validation accuracy — this is what gets evaluated
artifacts/checkpoints/<run_id>/last.pt     final epoch — for resuming an interrupted run
```

Required keys in every checkpoint, asserted by `tests/contracts/`:

```
model_name, state_dict, epoch, val_acc, class_names, input_shape, seed, config_snapshot
```

`class_names` and `input_shape` are mandatory so a checkpoint is **self-describing** — Member 1 can
evaluate it without re-reading the config that produced it.

`best` is selected on **validation** accuracy, never test.

Regenerate rather than share:

```powershell
python scripts/train.py --config configs/experiments/model_b__adam.yaml
```
