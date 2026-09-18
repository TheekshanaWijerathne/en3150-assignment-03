# `training/` — SEAM 3 producer

**Owner: Member 3.** Assignment §3 — **15 marks**, §4 training half — **25 marks**.

## Inputs

| Seam | From | What |
|---|---|---|
| 1 | Member 1 | `build_dataloaders(cfg) -> DataBundle` |
| 2 | Members 2 & 4 | `build_model(cfg.model_name, ...)` |
| — | `configs/stages/training.yaml` | epochs, optimizer, scheduler, early stopping |
| — | `configs/experiments/*.yaml` | the specific run |

## Outputs

| Path | Committed? |
|---|---|
| `artifacts/checkpoints/<run_id>/best.pt` | no — large, regenerable |
| `artifacts/checkpoints/<run_id>/last.pt` | no |
| `results/metrics/<run_id>/history.json` | **yes** |
| `results/figures/<run_id>/curves.png` | **yes** |

`best.pt` must carry every key in `contracts.types.CHECKPOINT_KEYS` — including `class_names` and
`input_shape`, so Member 1 can evaluate it without re-reading the config that produced it. A
checkpoint should be self-describing.

## One loop, four models

`trainer.py` must contain **no branch on model name**. If it ever needs one, the registry contract
is wrong and that is the thing to fix — a per-model branch here is how the §6 comparison quietly
becomes apples-to-oranges.

## Non-negotiable behaviours

Each is asserted by a contract test.

1. **Never touch `data.test`.** Not even to print a number. The held-out split stays sealed until
   Member 1 evaluates it.
2. **Time every epoch identically.** `epoch_time_s` covers the train pass plus the val pass. It is
   a column in the §4 table, so Model A and MobileNetV2 must be measured at the same two points.
   On GPU, call `torch.cuda.synchronize()` before stopping the clock — CUDA kernels are
   asynchronous, so a naive timer records queueing rather than compute.
3. **Record at least 20 epochs.** `history.schema.json` enforces `minItems: 20`.
4. **Refuse to write results when `subset_fraction < 1.0`.** A debug run must not reach a committed
   comparison table.
5. **Reseed before each optimizer variant**, not once at the start. Otherwise runs two and three
   inherit different initial weights and the §3 comparison measures initialisation noise alongside
   the optimizer.

## §3 — the optimizer study

Three runs on Model B, identical in every respect except the optimizer: `sgd`, `sgd_momentum`,
`adam`. Hold the seed, epochs, scheduler and augmentation fixed — change two things and the report
cannot attribute the difference to either.

The assignment asks specifically about **the impact of the momentum parameter**. Separate the two
effects, and show both in the curves rather than asserting them:

- the velocity term **damps oscillation** across narrow ravines in the loss surface;
- it **accelerates** travel along consistently descending directions.

Report the effect on convergence *speed* and on *final accuracy* separately — they are different
claims and can disagree.

On the choice of Adam: depthwise kernels carry 9 weights each while pointwise kernels carry
hundreds, so their gradient magnitudes differ by orders of magnitude and one global SGD learning
rate cannot serve both. Adam's per-parameter adaptive step can. Worth noting the cost too — Adam
holds two extra state tensors per parameter, roughly tripling optimizer memory versus plain SGD,
which is relevant in a report about memory-constrained devices (at training time, not inference).

Note the learning rates differ by design: 0.01 for the SGD variants, 0.001 for Adam. Plain SGD
takes raw-gradient-sized steps rather than normalised ones, so giving it Adam's rate would
understate it unfairly.

## Files

| File | Purpose |
|---|---|
| `trainer.py` | The model-agnostic loop |
| `optimizers.py` | Optimizer and scheduler factories; handles Member 4's `param_groups()` |
| `callbacks.py` | Early stopping (with the 20-epoch floor) and checkpoint selection |
| `optimizer_study.py` | The §3 three-way comparison |
