# `models/custom/` — Model A and Model B

**Owner: Member 2.** Assignment §2 — **20 marks**.

## Inputs

`num_classes` and `input_shape` from the `DataBundle`; architecture parameters from
`configs/stages/models.yaml` → `inputs.model_a` / `inputs.model_b`.

## Outputs

Two registry entries, `model_a` and `model_b`, plus two tables for the report:

- `results/tables/custom_architectures.md` — layer-by-layer topology
- `results/tables/param_breakdown.md` — per-layer parameter derivation

Both models return **raw logits**, expose `.num_classes`, and work at batch size 1 (Member 1
measures latency there).

## Model A — standard CNN

Interleaved `Conv2d` + `MaxPool2d` with a fully connected head. Deliberately **not**
parameter-constrained: it is the control that shows what Model B gives up.

Worth stating explicitly in the report: flattening an 8×8×128 map into `FC(128)` costs
`8*8*128*128 = 1,048,576` weights in a single layer — about ten times Model B's entire budget. That
one line is the clearest illustration of where parameters actually go in a small CNN.

## Model B — depthwise-separable CNN

**Hard cap: 100,000 trainable parameters.** `tests/test_param_budget.py` fails the build if
exceeded — do not rely on reading a printout.

Two distinct mechanisms carry the saving, and separating them is a stronger §2 answer than
attributing it all to the convolutions:

**1. Depthwise separable convolution.**

```
standard  : k^2 * C_in * C_out
separable : k^2 * C_in  +  C_in * C_out
ratio     : 1/C_out + 1/k^2         ~= 1/8 at k=3, C_out=64
```

**2. Global average pooling instead of flatten.** Replaces Model A's ~1.05M-weight FC layer with
`96 -> num_classes`, under a thousand. In a sub-100k budget this saves *more* than the depthwise
convolutions do.

## What §2 asks for beyond the code

- **Explicit network parameters** — kernel sizes, filter counts, FC widths, stated in the report.
- **Total trainable parameters calculated** — derive them by hand, per layer.
  `scripts/summarize_models.py` checks your arithmetic; it does not replace it.
- **Hardware-aware activation justification.** ReLU and ReLU6 are a single `max()` — no
  exponential, no lookup table, no FPU required. sigmoid, tanh, ELU, GELU and Swish each cost a
  transcendental per activation, which dominates runtime on a microcontroller. ReLU6's bounded
  range additionally keeps int8 quantisation well-scaled, which is what actually gets a model onto
  an MCU runtime. `hardswish` is the interesting middle case — a Swish approximation using only
  add/multiply/clamp, which is why MobileNetV3 uses it on mobile silicon.

## The bug that will cost you the budget

A depthwise convolution built with `groups=1` instead of `groups=in_channels` trains perfectly
well. It is simply a normal convolution with several times the parameters, and nothing else in the
pipeline complains. `tests/test_param_budget.py` is what catches it.
