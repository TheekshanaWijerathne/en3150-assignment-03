# `models/` — SEAM 2 producer

**Shared between Member 2 (`custom/`) and Member 4 (`pretrained/`).**
`registry.py` is a shared file: PR + 1 review.

## Inputs

| From | What |
|---|---|
| `DataBundle` | `num_classes`, `input_shape` |
| `configs/stages/models.yaml` | Model A and Model B architecture parameters (Member 2) |
| `configs/stages/pretrained.yaml` | backbone names, freezing strategy, learning rates (Member 4) |

## Outputs

```python
from edgecnn.models import build_model
model = build_model("model_b", num_classes=10, input_shape=(3, 64, 64))
```

An `nn.Module` satisfying `contracts.protocols.ClassifierModel`:

```
in  : float32 (B, 3, 64, 64)
out : float32 (B, num_classes)   RAW LOGITS
```

Registry keys: `model_a`, `model_b` (Member 2) · `mobilenet_v2`, `squeezenet1_1` (Member 4).

## Why the registry exists

Member 3's trainer must be identical for all four models. If it imported model modules directly it
would grow a branch per architecture, and the four models would drift apart in ways that make the
§6 comparison table indefensible — different loss reductions, different eval-mode handling,
different timing points.

So the trainer knows exactly one thing: `build_model(name, ...)`. It never learns which model it
got. **If the trainer ever needs a branch on model name, the contract is wrong — fix the contract,
not the trainer.**

## No softmax inside a model

`CrossEntropyLoss` applies log-softmax itself. A model that also softmaxes trains on a
double-softmaxed signal — it does not crash, it just quietly underperforms. `tests/contracts/`
checks for it by asserting output rows do not sum to 1.

It also matters for §2's hardware-aware activation argument: softmax is an exponential per class
per inference, and it is pure overhead when all you need is the arg-max label.

## Subfolders

- [`custom/`](custom/README.md) — Member 2, §2 [20 marks]
- [`pretrained/`](pretrained/README.md) — Member 4, §5 [20 marks]
