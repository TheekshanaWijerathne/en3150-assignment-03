# `results/metrics/` — the numbers, committed

**COMMITTED.** This is SEAM 4, and the hand-off Member 4 depends on.

```
results/metrics/<run_id>/
├── history.json         Member 3 — per-epoch loss, accuracy, LR, epoch time
├── test_metrics.json    Member 1 — accuracy, macro P/R/F1, per-class, confusion matrix
└── resources.json       Member 1 — params, size, MACs, latency, peak memory, device
```

`<run_id>` is `{model}__{optimizer}__seed{seed}`, from `edgecnn.contracts.paths.make_run_id`.

## Commit these as soon as a run finishes

Member 4 cannot build the Section 4, 5 or 6 tables without every other member's JSON. If these
lived only on the machine that produced them, assembling the report would mean re-running every
experiment the week it is due. They are a few kilobytes each.

## Rules

- **Never hand-edit.** Each file is schema-validated on write and on read; an edit will fail
  `pytest tests/contracts` and, worse, silently misreport a number in the table.
- **Never commit a debug run** — `subset_fraction < 1.0`, fewer than 20 epochs, or the synthetic
  dataset. The first two are refused by `scripts/train.py`; the third is on you to check.

Expected when complete: `model_a__adam__seed42`, `model_b__adam__seed42`, `model_b__sgd__seed42`,
`model_b__sgd_momentum__seed42`, `mobilenet_v2__adam__seed42`, `squeezenet1_1__adam__seed42`.
