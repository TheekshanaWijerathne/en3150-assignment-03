# `scripts/` — command-line entry points

Six scripts, one per stage, all driven by `--config`. Every script is a thin wrapper: it loads a
config, calls into `edgecnn`, and prints where it wrote things. **No logic lives here** — logic
goes in the package so it can be tested and reused from a notebook.

| Script | Owner | Reads | Writes |
|---|---|---|---|
| `prepare_data.py` | M1 | `configs/stages/data.yaml` | `data/splits/*` (committed), `data/processed/` |
| `summarize_models.py` | M2 | `configs/stages/models.yaml` | `results/tables/custom_architectures.md`, `param_breakdown.md` |
| `train.py` | M3 | `configs/experiments/<run>.yaml` | `artifacts/checkpoints/<run_id>/`, `results/metrics/<run_id>/history.json` |
| `run_optimizer_study.py` | M3 | `configs/stages/training.yaml` | three runs + `results/figures/optimizer_overlay.png` |
| `evaluate.py` | M1 | `configs/experiments/<run>.yaml` | `test_metrics.json`, `resources.json` |
| `build_report_assets.py` | M4 | all committed metrics JSON | `results/tables/*.md`, `report/figures/` |

## Typical order

```powershell
# once, by Member 1 — then commit data/splits/
python scripts/prepare_data.py --config configs/stages/data.yaml

# Member 2 — check the architecture and the 100k budget
python scripts/summarize_models.py --check-budget

# Member 3 — one run at a time, or the whole §3 study
python scripts/train.py --config configs/experiments/model_b__adam.yaml
python scripts/run_optimizer_study.py

# Member 1 — every model, so all rows come from one code path
python scripts/evaluate.py --all

# Member 4 — once results have landed
python scripts/build_report_assets.py
```

## Before the real data exists

```powershell
python scripts/prepare_data.py --config configs/stages/data.yaml --synthetic
python scripts/train.py --config configs/experiments/model_b__adam.yaml --epochs 2 --subset-fraction 0.05
```

Seconds to run, no download. This is how Members 3 and 4 work during Phase 0.

## Conventions every script follows

- `--config` is the only required argument; everything else is an override.
- Call `setup_logging()` then `seed_everything(cfg.seed)` first, before touching a model.
- Derive paths from `edgecnn.contracts.paths` — never concatenate strings.
- Write artifacts through `schema.write_json` so an invalid file can never reach disk.
- Print the `run_id` and every output path on exit, so the next member knows what to read.
- Refuse rather than warn when a run would produce a misleading committed artifact —
  `subset_fraction < 1.0` or fewer than 20 epochs. It is far cheaper to fail here than to discover
  it during report assembly.
