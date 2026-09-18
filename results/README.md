# `results/` — everything the report is built from

**Committed in full.** This folder is the reason Member 4 can assemble the report in seconds, on a
laptop, with no GPU, without re-running anyone else's training.

| Folder | Contents | Producer |
|---|---|---|
| `metrics/<run_id>/` | `history.json`, `test_metrics.json`, `resources.json` | M3, M1 |
| `figures/` | Loss curves, confusion matrices, trade-off scatter plots | M3, M4 |
| `tables/` | Markdown comparison tables, pasted straight into the report | M2, M4 |

## Per-run layout

```
results/metrics/model_b__adam__seed42/
├── history.json         Member 3 — per-epoch loss, accuracy, LR, epoch time
├── test_metrics.json    Member 1 — accuracy, macro P/R/F1, per-class, confusion matrix
└── resources.json       Member 1 — params, size, MACs, latency, peak memory, device
```

`<run_id>` is always `{model}__{optimizer}__seed{seed}`, derived by
`edgecnn.contracts.paths.make_run_id`. Never assembled by hand.

## Why JSON is committed but checkpoints are not

Member 4 cannot build the §4, §5 or §6 tables without every other member's `test_metrics.json` and
`resources.json`. If those lived only on the machine that produced them, assembling the report
would mean re-running every experiment the week it is due.

These files are a few kilobytes each. Checkpoints are megabytes and fully regenerable, so
`artifacts/checkpoints/` is git-ignored instead.

**So: commit your metrics JSON as soon as a run finishes.** It is not optional tidiness — it is the
hand-off.

## Rules

- **Never hand-edit a file in here.** Every one is schema-validated on write and on read; an edited
  file will fail `pytest tests/contracts` and, worse, silently misreport a number in the table.
- **Never commit results from a debug run.** `subset_fraction < 1.0`, fewer than 20 epochs, or the
  synthetic dataset. The first two are refused by `scripts/train.py`; the third is on you to check.
- **Figures are generated, not hand-made.** `results/figures/` is the source; `report/figures/` is
  a mirror produced by `scripts/build_report_assets.py`.

## Expected contents when complete

Six run directories:

```
model_a__adam__seed42          model_b__sgd__seed42
model_b__adam__seed42          model_b__sgd_momentum__seed42
mobilenet_v2__adam__seed42     squeezenet1_1__adam__seed42
```

Three tables: `custom_model_comparison.md` (§4), `optimizer_comparison.md` (§3),
`final_comparison.md` (§6).
