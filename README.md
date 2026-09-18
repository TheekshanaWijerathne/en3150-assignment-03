# EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification

A study of the trade-off between **accuracy, memory footprint and computational cost** for image
classifiers targeting embedded and low-resource devices. Two custom CNNs are designed and trained
from scratch, compared against two fine-tuned lightweight state-of-the-art backbones, on identical
data splits.

| | |
|---|---|
| **Group number** | _to be added_ |
| **Members** | _to be added — name + index number, all four_ |
| **Dataset** | EuroSAT — 27,000 satellite images, 10 classes, natively 64×64 RGB |
| **Framework** | PyTorch |
| **Report file** | `<GroupNo>_A03_EN3150.pdf` |
| **Repository** | https://github.com/ThejithaR/EN3150-Assignment-03-CNN |

---

## Quick start

```powershell
git clone https://github.com/ThejithaR/EN3150-Assignment-03-CNN.git
cd EN3150-Assignment-03-CNN
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/contracts -q             # should be green
```

`-e` is an *editable* install: it records where your source folder is, so **your edits take effect
immediately and there is no rebuild step**. You only re-run it when `pyproject.toml` dependencies
change. Full setup and day-to-day git workflow: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## How the work is split

The project is cut **horizontally** — each member owns one layer of the stack, and the layers
correspond almost one-to-one with the assignment's marking sections, so each person defends their
own marks.

| | **Member 1** — Data & Infrastructure | **Member 2** — Custom Architectures | **Member 3** — Training & Optimization | **Member 4** — SOTA & Comparison |
|---|---|---|---|---|
| **Name** | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **Owns** | [src/edgecnn/data/](src/edgecnn/data/), [utils/](src/edgecnn/utils/), [config/](src/edgecnn/config/), [contracts/](src/edgecnn/contracts/)\*, `evaluation/{metrics,benchmark,plotting}.py` | [models/custom/](src/edgecnn/models/custom/), [registry.py](src/edgecnn/models/registry.py)\* | [src/edgecnn/training/](src/edgecnn/training/), [evaluation/curves.py](src/edgecnn/evaluation/curves.py) | [models/pretrained/](src/edgecnn/models/pretrained/), [evaluation/reporting.py](src/edgecnn/evaluation/reporting.py) |
| **Scripts** | [prepare_data.py](scripts/prepare_data.py), [evaluate.py](scripts/evaluate.py) | [summarize_models.py](scripts/summarize_models.py) | [train.py](scripts/train.py), [run_optimizer_study.py](scripts/run_optimizer_study.py) | [build_report_assets.py](scripts/build_report_assets.py) |
| **Configs** | [base.yaml](configs/base.yaml), [stages/data.yaml](configs/stages/data.yaml), [contracts/](configs/contracts/) | [stages/models.yaml](configs/stages/models.yaml) | [stages/training.yaml](configs/stages/training.yaml), [experiments/](configs/experiments/) | [stages/pretrained.yaml](configs/stages/pretrained.yaml), [stages/evaluation.yaml](configs/stages/evaluation.yaml) |
| **Report section** | §1 Data + reproducibility | **§2 [20]** | **§3 [15]** + §4 training **[25]** | **§5 [20]** + **§6 [20]** |

`*` Shared file — PR + one review before changing. Member 1 is custodian of `contracts/`;
Member 2 owns `registry.py` but Member 4 appends the pretrained entries.

**No two members edit the same file.** If you find yourself needing to, that is a signal the
contract is wrong — raise it rather than working around it.

---

## The five seams

This is the important part. Members do not read each other's *code* — they exchange these five
artifacts, each with a schema that is enforced at runtime and in the test suite.

```
  M1 ──(1)──► M2 ──(2)──► M3 ──(3)──► evaluation ──(4)──► M4 ──(5)──► report
  data        models      training         ▲            reporting
    │                                      │
    └────── metrics + benchmark (M1) ──────┘
```

| # | Seam | Producer → Consumer | Carrier | Schema |
|---|---|---|---|---|
| **1** | Data | M1 → M2, M3, M4 | `DataBundle` + `data/splits/split_manifest.csv` | [split_manifest](configs/contracts/split_manifest.schema.json), [split_meta](configs/contracts/split_meta.schema.json), [norm_stats](configs/contracts/norm_stats.schema.json) |
| **2** | Models | M2, M4 → M3 | `build_model()` → `nn.Module` returning **raw logits** | [protocols.py](src/edgecnn/contracts/protocols.py) |
| **3** | Training | M3 → M1, M4 | `best.pt` + `history.json` | [history](configs/contracts/history.schema.json) |
| **4** | Evaluation | M1 → M4 | `test_metrics.json` + `resources.json` | [test_metrics](configs/contracts/test_metrics.schema.json), [resources](configs/contracts/resources.schema.json) |
| **5** | Reporting | M4 → report | `results/tables/*.md` + `results/figures/*.png` | — |

### Seam 1 — the data contract

```python
from edgecnn.data import build_dataloaders
data = build_dataloaders(cfg)        # -> edgecnn.contracts.DataBundle
```

Every batch, every split, every model:

```
images : float32, shape (B, 3, 64, 64), normalised with TRAIN-ONLY statistics
labels : int64,   shape (B,),           values in [0, num_classes)
```

`data/splits/split_manifest.csv` is the single source of truth for who sees what — columns
`relative_path, label_index, label_name, split`. It is drawn **once**, committed, and never
redrawn. That is what makes the final comparison valid: an accuracy difference between Model B and
MobileNetV2 is attributable to the architecture, not to the data.

### Seam 2 — the model contract

```python
from edgecnn.models import build_model
model = build_model("model_b", num_classes=10, input_shape=(3, 64, 64))
```

Every registered model — hand-written or from torchvision — takes `(B,3,64,64)` and returns
**raw logits** `(B, num_classes)`. **No softmax inside the model.** `CrossEntropyLoss` applies
log-softmax itself; a model that also softmaxes trains on a double-softmaxed signal, does not
crash, and quietly underperforms. Keeping the exponential out of the model is also part of the
§2 hardware-aware activation argument.

Member 3's trainer never imports a model module — only `build_model`. That indirection is what
keeps the training loop identical across all four architectures.

### Seams 3–5 — the `run_id` convention

```
run_id = "{model}__{optimizer}__seed{seed}"      e.g.  model_b__sgd_momentum__seed42
```

Derived in exactly one place — [`contracts/paths.py`](src/edgecnn/contracts/paths.py) — so Member 3
writes where Members 1 and 4 read. **Nobody builds a path string by hand.** A typo there produces
an empty table the day before submission, and it fails silently, because a glob that matches
nothing raises nothing.

```
artifacts/checkpoints/<run_id>/best.pt        git-ignored (large)
results/metrics/<run_id>/history.json         COMMITTED  (M3)
results/metrics/<run_id>/test_metrics.json    COMMITTED  (M1)
results/metrics/<run_id>/resources.json       COMMITTED  (M1)
results/figures/<run_id>/*.png                COMMITTED  (M3, M4)
```

Metrics JSON is committed deliberately: Member 4 cannot build the §6 comparison table without
everyone else's numbers, and nobody should re-run 30 epochs of training to redraw a table.

### How the contract is enforced

Schemas in [configs/contracts/](configs/contracts/) are validated at runtime — producers write via
`schema.write_json`, consumers read via `schema.read_json`, so a malformed artifact never reaches
disk. `pytest tests/contracts` re-checks them.

**A schema failure is a broken interface, not a bug in your code.** Tell whoever owns the producing
stage; do not silently adapt around it.

---

## Phase 0 — do this before anyone writes a model

The horizontal split only works because of this. Without it, Members 3 and 4 sit idle waiting for
Members 1 and 2.

1. **Freeze the contracts together.** Read [`src/edgecnn/contracts/`](src/edgecnn/contracts/) and
   [`configs/contracts/`](configs/contracts/) as a group. This is the one meeting that matters.
2. **Member 1 ships the synthetic fixture first** — `make_synthetic_fixture` in
   [data/synthetic.py](src/edgecnn/data/synthetic.py). ~200 generated 64×64 images, 4 classes, a
   real manifest. Set `dataset.name: synthetic` in any config.
   **Members 3 and 4 then have working DataLoaders on hour one**, before EuroSAT finishes
   downloading anywhere.
3. **Member 2 ships shape-only model stubs** — correct output shape, random weights, trivially
   small. Member 3's trainer then runs end-to-end immediately.

After Phase 0, all four work in parallel with no blocking dependency.

---

## Experiment matrix — 6 required runs

All at 64×64, `seed: 42`, ≥20 epochs, identical splits.

| run_id | Owner | Serves |
|---|---|---|
| `model_b__adam__seed42` | M3 | §3 chosen optimizer · §4 · §6 baseline |
| `model_b__sgd__seed42` | M3 | §3 (a) plain SGD |
| `model_b__sgd_momentum__seed42` | M3 | §3 (b) momentum |
| `model_a__adam__seed42` | M3 | §4 Model A vs Model B |
| `mobilenet_v2__adam__seed42` | M4 | §5 · §6 |
| `squeezenet1_1__adam__seed42` | M4 | §5 · §6 |

```powershell
python scripts/prepare_data.py --config configs/stages/data.yaml   # once, by Member 1
python scripts/train.py    --config configs/experiments/model_b__adam.yaml
python scripts/evaluate.py --config configs/experiments/model_b__adam.yaml
python scripts/build_report_assets.py                              # after all runs land
```

Optional if compute allows: the same three-optimizer sweep on Model A.
`configs/base.yaml` carries a `subset_fraction` knob for fast development runs — results from a
run with it below 1.0 must **not** be committed, and `scripts/train.py` refuses to write them.

### One decision to defend in §6

The pretrained backbones are fine-tuned at **64×64**, not upsampled to their native 224. This keeps
the memory and compute comparison honest — the point is cost at the resolution the sensor actually
produces. The resulting handicap is real and is a finding rather than a flaw: MobileNetV2's 32×
total stride leaves a 2×2 feature map at this resolution, so it cannot use the depth it was
designed around. Set `pretrained.input_resolution: 128` to run that ablation.

---

## Assignment requirements → where they live

| § | Requirement | Marks | Owner | Code |
|---|---|---|---|---|
| 1 | Dataset ≤64×64, 70/15/15 splits | — | M1 | [data/](src/edgecnn/data/) |
| 2 | Model A standard CNN; Model B depthwise-separable ≤100k params; activation justification | 20 | M2 | [models/custom/](src/edgecnn/models/custom/) |
| 3 | Chosen optimizer vs SGD vs SGD+momentum | 15 | M3 | [training/optimizer_study.py](src/edgecnn/training/optimizer_study.py) |
| 4 | ≥20 epochs, loss curves, accuracy/confusion/precision/recall, A-vs-B table | 25 | M3 + M1 | [training/](src/edgecnn/training/), [evaluation/](src/edgecnn/evaluation/) |
| 5 | Fine-tune two lightweight SOTA backbones, same splits | 20 | M4 | [models/pretrained/](src/edgecnn/models/pretrained/) |
| 6 | Model B vs SOTA — accuracy / memory / compute trade-off | 20 | M4 | [evaluation/reporting.py](src/edgecnn/evaluation/reporting.py) |

Model B's 100,000-parameter cap is asserted by
[tests/test_param_budget.py](tests/test_param_budget.py), not read off a printout.

---

## Repository layout

```text
EN3150-Assignment-03-CNN/
├── README.md                  this file — split, seams, run matrix
├── CONTRIBUTING.md            install, branching, staying in sync
├── pyproject.toml             package `edgecnn`; the only reason to reinstall
├── configs/
│   ├── base.yaml              seed, device, paths, image size
│   ├── stages/                one per stage — each declares its inputs & outputs
│   ├── experiments/           one composed file per run_id
│   └── contracts/             JSON Schemas — the on-disk seam formats
├── data/{raw,processed,splits}/
├── src/edgecnn/
│   ├── contracts/             FROZEN — types, protocols, paths, schema
│   ├── config/                base ← stage ← experiment merge
│   ├── data/                  M1
│   ├── models/                registry.py + custom/ (M2) + pretrained/ (M4)
│   ├── training/              M3
│   ├── evaluation/            M1 metrics+benchmark · M3 curves · M4 reporting
│   └── utils/                 M1
├── scripts/                   six CLI entry points, all --config driven
├── tests/{contracts,fixtures}/
├── artifacts/{checkpoints,exports}/
├── results/{figures,metrics,tables}/
├── notebooks/
└── report/figures/
```

Every folder has a `README.md` naming its owner, its inputs and its outputs.

## Reproducibility rules

- The test set stays sealed until final evaluation — the trainer never constructs a loader over it.
- Preprocessing statistics are fitted on the training split only; the schema pins `fitted_on` to
  `"train"` so the mistake cannot be committed silently.
- The split definition and seed live in `data/splits/` and are committed.
- Every committed run uses `seed: 42`.
- Parameter counts, model sizes, epoch times and the evaluation hardware are recorded in
  `resources.json` for every model, measured by one code path.
- Commit regularly — the assignment grades sustained development history.
