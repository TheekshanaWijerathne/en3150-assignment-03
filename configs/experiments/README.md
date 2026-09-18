# `configs/experiments/` — one file per run

Six files, one per required run. Each `extends:` the stage configs it needs, then overrides only
what makes it distinct — usually just the model and the optimizer.

| File | run_id | Owner | Serves |
|---|---|---|---|
| `model_a__adam.yaml` | `model_a__adam__seed42` | M3 | §4 baseline |
| `model_b__adam.yaml` | `model_b__adam__seed42` | M3 | §3, §4, §6 |
| `model_b__sgd.yaml` | `model_b__sgd__seed42` | M3 | §3 (a) |
| `model_b__sgd_momentum.yaml` | `model_b__sgd_momentum__seed42` | M3 | §3 (b) |
| `mobilenet_v2__adam.yaml` | `mobilenet_v2__adam__seed42` | M4 | §5, §6 |
| `squeezenet1_1__adam.yaml` | `squeezenet1_1__adam__seed42` | M4 | §5, §6 |

## The filename is not decorative

It must equal the derived `run_id` minus the `__seed<n>` suffix. A mismatch would scatter one
logical run across two directory names — and since a glob matching nothing raises nothing, it would
fail silently, at report time. `tests/contracts/` asserts the match.

## Contract tests also assert

- all six configs exist;
- all use the same seed;
- all set `epochs >= 20`;
- none has `subset_fraction < 1.0`.

## Adding a run

Copy the closest existing file, change `model.name` or `optimizer`, rename to match the new
`run_id`. Do not duplicate settings a stage config already provides — that is how two runs drift
apart in a way nobody notices.

## The §3 learning rates differ by design

`0.01` for the SGD variants, `0.001` for Adam. Plain SGD takes raw-gradient-sized steps rather than
normalised ones, so handing it Adam's rate would understate it unfairly.
