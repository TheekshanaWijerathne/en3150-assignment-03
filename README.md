# EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification

A study of the trade-off between **accuracy, memory footprint and computational cost** for image
classifiers targeting embedded and low-resource devices. Two custom CNNs are designed and trained
from scratch, compared against two fine-tuned lightweight state-of-the-art backbones, on identical
data splits.

| | |
|---|---|
| **Group** | Misclassified |
| **Members** | _to be added — name + index number, all four_ |
| **Dataset** | EuroSAT — 27,000 satellite images, 10 classes, natively 64×64 RGB |
| **Framework** | PyTorch |
| **Report file** | `Misclassified_A03_EN3150.pdf` |
| **Repository** | https://github.com/ThejithaR/EN3150-Assignment-03-CNN |

---

## Quick start

```powershell
git clone https://github.com/ThejithaR/EN3150-Assignment-03-CNN.git
cd EN3150-Assignment-03-CNN
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/contracts                # should be green
```

Then open [`notebooks/00_setup.ipynb`](notebooks/00_setup.ipynb) in VS Code, select the `.venv`
kernel, and run it top to bottom.

`-e` is an *editable* install: it records where your source folder is, so **your edits take effect
immediately, with no rebuild**. Re-run it only when `pyproject.toml` dependencies change.
**Before writing any code, read [CONTRIBUTING.md](CONTRIBUTING.md)** — the developer guide.

---

## How development works

Experiments run in **notebooks**, one per assignment section. The code those notebooks call lives in a
shared **library**, `src/edgecnn/`.

```
scratch notebook          src/edgecnn/*.py             section notebook 0X
try an idea  ──promote──►  shared library  ◄──import──  run step by step, see every result
                                                            │  MODE = "official"
                                                            ▼
                                           results/*.json + notebook with outputs
                                                            │  git push
                                                            ▼
                                           the next member's notebook reads it
```

1. **Prototype** in your own `notebooks/scratch/` notebook.
2. **Promote** it into its stub in `src/edgecnn/` as soon as it works. Every stub's docstring says
   what to build. Notebooks are never "converted to .py" at the end — only functions move, one at a
   time.
3. **Run** it in your section notebook with `MODE = "synthetic"`, then `"debug"`, checking every
   step's output.
4. **Official run:** `MODE = "official"`, Restart & Run All, then commit the notebook *with its
   outputs* together with the files it wrote.

Only official runs write results, so debug work can never leak into the report. Full workflow,
worked example and troubleshooting: [CONTRIBUTING.md](CONTRIBUTING.md). Notebook reference:
[notebooks/README.md](notebooks/README.md). Laptop too slow for a run? [COLAB.md](COLAB.md).

---

## How the work is split

The project is cut **horizontally**: each member owns one layer of the stack. The layers map almost
one-to-one onto the assignment's marking sections, so each person defends their own marks.

| | **Member 1** — Data & Infrastructure | **Member 2** — Custom Architectures | **Member 3** — Training & Optimization | **Member 4** — SOTA & Comparison |
|---|---|---|---|---|
| **Name** | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **Notebooks** | [`00_setup`](notebooks/00_setup.ipynb), [`01_data_preparation`](notebooks/01_data_preparation.ipynb), [`06_resource_benchmark`](notebooks/06_resource_benchmark.ipynb) | [`02_custom_architectures`](notebooks/02_custom_architectures.ipynb) | [`03_optimizer_study`](notebooks/03_optimizer_study.ipynb), [`04_custom_training_evaluation`](notebooks/04_custom_training_evaluation.ipynb) | [`05_pretrained_finetuning`](notebooks/05_pretrained_finetuning.ipynb), [`07_final_comparison`](notebooks/07_final_comparison.ipynb) |
| **Library** | [data/](src/edgecnn/data/), [utils/](src/edgecnn/utils/), [config/](src/edgecnn/config/), [contracts/](src/edgecnn/contracts/)\*, `evaluation/{metrics,benchmark,plotting}.py` | [models/custom/](src/edgecnn/models/custom/), [registry.py](src/edgecnn/models/registry.py)\* | [training/](src/edgecnn/training/), [evaluation/curves.py](src/edgecnn/evaluation/curves.py) | [models/pretrained/](src/edgecnn/models/pretrained/), [evaluation/reporting.py](src/edgecnn/evaluation/reporting.py) |
| **Configs** | [base.yaml](configs/base.yaml), [stages/data.yaml](configs/stages/data.yaml), [contracts/](configs/contracts/) | [stages/models.yaml](configs/stages/models.yaml) | [stages/training.yaml](configs/stages/training.yaml), [experiments/](configs/experiments/) | [stages/pretrained.yaml](configs/stages/pretrained.yaml), [stages/evaluation.yaml](configs/stages/evaluation.yaml) |
| **Report section** | §1 Data + reproducibility | **§2 [20]** | **§3 [15]** + §4 training **[25]** | **§5 [20]** + **§6 [20]** |

`*` Shared file — PR + one review before changing. Member 1 is custodian of `contracts/`;
Member 2 owns `registry.py`, but Member 4 appends the pretrained entries.

**No two members edit the same file** — notebooks included. If you find yourself needing to, the
contract is probably wrong. Raise it rather than working around it.

---

## The five seams

This is the important part. Members do not read each other's *code*. They exchange these five
artifacts, each with a schema enforced at runtime and in the test suite.

```
  M1 ──(1)──► M2 ──(2)──► M3 ──(3)──► evaluation ──(4)──► M4 ──(5)──► report
  data        models      training         ▲            reporting
    │                                      │
    └────── metrics + benchmark (M1) ──────┘
```

| # | Seam | Producer → Consumer | Carrier | Schema |
|---|---|---|---|---|
| **1** | Data | M1 → M2, M3, M4 | `DataBundle` + `data/splits/split_manifest.csv` | [split_manifest](configs/contracts/split_manifest.schema.json), [split_meta](configs/contracts/split_meta.schema.json), [norm_stats](configs/contracts/norm_stats.schema.json) |
| **2** | Models | M2, M4 → M3 | `build_model_from_config()` → `nn.Module` returning **raw logits** | [protocols.py](src/edgecnn/contracts/protocols.py) |
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

`data/splits/split_manifest.csv` is the single source of truth for which image is in which split,
with columns `relative_path, label_index, label_name, split`. It is drawn **once**, committed, and
never redrawn. That is what makes the final comparison valid: an accuracy difference between Model B
and MobileNetV2 is down to the architecture, not the data.

### Seam 2 — the model contract

```python
from edgecnn.models import build_model_from_config
model = build_model_from_config(cfg, num_classes=data.num_classes)
```

Every registered model — hand-written or from torchvision — takes `(B,3,64,64)` and returns **raw
logits** `(B, num_classes)`. **No softmax inside the model.** `CrossEntropyLoss` applies log-softmax
itself. A model that also softmaxes trains on a double-softmaxed signal: it does not crash, it just
quietly underperforms. Keeping the exponential out of the model is also part of the §2
hardware-aware activation argument.

`build_model_from_config` reads the model's settings from the config and seeds the weights. The
model trained in notebook `03` is therefore exactly the architecture profiled in `06`. The trainer
never imports a model module, which keeps one training loop for all four architectures.

### Seams 3–5 — the `run_id` convention

```
run_id = "{model}__{optimizer}__seed{seed}"      e.g.  model_b__sgd_momentum__seed42
```

Derived in exactly one place — [`contracts/paths.py`](src/edgecnn/contracts/paths.py) — so
producers write where consumers read. **Nobody builds a path string by hand.** A typo there produces
an empty table the day before submission, and fails silently, because a glob that matches nothing
raises nothing.

```
artifacts/checkpoints/<run_id>/best.pt        git-ignored (large)
artifacts/checkpoints/_debug/<run_id>/        git-ignored - synthetic and debug runs only
results/metrics/<run_id>/history.json         COMMITTED  (notebooks 03-05)
results/metrics/<run_id>/test_metrics.json    COMMITTED  (notebooks 03-05)
results/metrics/<run_id>/resources.json       COMMITTED  (notebook 06)
results/figures/<run_id>/*.png                COMMITTED
```

Metrics JSON is committed deliberately. Member 4 cannot build the §6 comparison table without
everyone's numbers, and nobody should re-run 30 epochs of training to redraw a table.

### How the contract is enforced

Schemas in [configs/contracts/](configs/contracts/) are validated at runtime: producers write through
`schema.write_json` and consumers read through `schema.read_json`, so a malformed artifact never
reaches disk. `pytest tests/contracts` checks them again, along with the config layout and the
notebooks' structure.

**A schema failure is a broken interface, not a bug in your code.** Tell whoever owns the producing
stage. Do not quietly adapt around it.

---

## Phase 0 — before anyone writes a model

The horizontal split only works because of this. Without it, Members 3 and 4 would sit idle waiting
for Members 1 and 2.

1. **Freeze the contracts together.** Read [`src/edgecnn/contracts/`](src/edgecnn/contracts/) and
   [`configs/contracts/`](configs/contracts/) as a group. This is the one meeting that matters.
2. **Member 1 ships the synthetic fixture first** — `make_synthetic_fixture` in
   [data/synthetic.py](src/edgecnn/data/synthetic.py), the synthetic branch of
   `build_dataloaders`, and `apply_style`. With `MODE = "synthetic"`, **every notebook then has working
   data within the first hour**, before EuroSAT has downloaded anywhere.
3. **Member 2 ships shape-only model stubs**: correct output shape, random weights, tiny. Member 3's
   notebook `03` then trains end to end straight away.

After Phase 0, all four work in parallel with no blocking dependency. The full stage plan, and the
gate for each stage, is in [CONTRIBUTING.md §7](CONTRIBUTING.md#7-official-runs).

---

## Experiment matrix — 6 required runs

All at 64×64, `seed: 42`, 30 epochs (≥20 required), identical splits.

| run_id | Notebook | Owner | Serves |
|---|---|---|---|
| `model_b__sgd__seed42` | `03_optimizer_study` | M3 | §3 (a) plain SGD |
| `model_b__sgd_momentum__seed42` | `03_optimizer_study` | M3 | §3 (b) momentum |
| `model_b__adam__seed42` | `03_optimizer_study` | M3 | §3 chosen optimizer · §4 · §6 baseline |
| `model_a__adam__seed42` | `04_custom_training_evaluation` | M3 | §4 Model A vs Model B |
| `mobilenet_v2__adam__seed42` | `05_pretrained_finetuning` | M4 | §5 · §6 |
| `squeezenet1_1__adam__seed42` | `05_pretrained_finetuning` | M4 | §5 · §6 |

**Official run order:** `01` → `02` → `03` → `04` → `05` → `06` → `07`. `06` then profiles all four
architectures in one CPU session, and `07` builds every table.

Optional if compute allows: the same three-optimizer sweep on Model A.

### One decision to defend in §6

The pretrained backbones are fine-tuned at **64×64**, not upsampled to their native 224. This keeps
the memory and compute comparison honest: the point is the cost at the resolution the sensor
actually produces. The handicap is real, and it is a finding rather than a flaw. MobileNetV2's 32×
total stride leaves a 2×2 feature map at this resolution, so it cannot use the depth it was designed
around. Set `pretrained.input_resolution: 128` to run that ablation.

---

## Assignment requirements → where they live

| § | Requirement | Marks | Owner | Notebook | Library |
|---|---|---|---|---|---|
| 1 | Dataset ≤64×64, 70/15/15 splits | — | M1 | `01` | [data/](src/edgecnn/data/) |
| 2 | Model A standard CNN; Model B depthwise-separable ≤100k params; activation justification | 20 | M2 | `02` | [models/custom/](src/edgecnn/models/custom/) |
| 3 | Chosen optimizer vs SGD vs SGD+momentum | 15 | M3 | `03` | [training/](src/edgecnn/training/) |
| 4 | ≥20 epochs, loss curves, accuracy/confusion/precision/recall, A-vs-B table | 25 | M3 + M1 | `04`, `06`, `07` | [training/](src/edgecnn/training/), [evaluation/](src/edgecnn/evaluation/) |
| 5 | Fine-tune two lightweight SOTA backbones, same splits | 20 | M4 | `05` | [models/pretrained/](src/edgecnn/models/pretrained/) |
| 6 | Model B vs SOTA — accuracy / memory / compute trade-off | 20 | M4 | `06`, `07` | [evaluation/reporting.py](src/edgecnn/evaluation/reporting.py) |

Model B's 100,000-parameter cap is asserted by
[tests/test_param_budget.py](tests/test_param_budget.py), not read off a printout.

---

## Repository layout

```text
EN3150-Assignment-03-CNN/
├── README.md                  this file — split, seams, run matrix
├── CONTRIBUTING.md            the developer guide: setup, workflow, git, official runs
├── COLAB.md                   running notebooks on Colab's GPU: setup, handover, troubleshooting
├── SUBMISSION.md              how the submission folder is built and checked, and what it must not contain
├── .env.example               the per-member Colab Secrets, by name; a filled-in `.env` is git-ignored
├── pyproject.toml             package `edgecnn`; the only reason to reinstall
├── notebooks/
│   ├── 00_setup … 07_final_comparison.ipynb   one per section; the official drivers
│   └── scratch/               personal prototyping notebooks
├── configs/
│   ├── base.yaml              seed, device, paths, image size, MODE presets
│   ├── stages/                one per stage — settings + that stage's I/O declaration
│   ├── experiments/           one file per run: its model and optimizer
│   └── contracts/             JSON Schemas — the on-disk seam formats
├── data/{raw,processed,splits}/
├── src/edgecnn/
│   ├── contracts/             FROZEN — types, protocols, paths, schema
│   ├── config/                base ← stage ← experiment ← mode
│   ├── data/                  M1
│   ├── models/                registry.py + custom/ (M2) + pretrained/ (M4)
│   ├── training/              M3
│   ├── evaluation/            M1 metrics+benchmark · M3 curves · M4 reporting
│   └── utils/                 M1 — seeding, device, logging, Colab
├── tests/{contracts,fixtures}/
├── artifacts/{checkpoints,exports}/
├── results/{figures,metrics,tables}/
├── report/figures/
├── submission/                the code we upload: generated by tools/build_submission.py
└── tools/                     building, checking and running the submission
```

Every folder has a `README.md` naming its owner, its inputs and its outputs.

## Reproducibility rules

- Only `MODE = "official"` writes results. Official notebooks are committed after one clean
  Restart & Run All.
- The test set stays sealed until final evaluation: the trainer never constructs a loader over it.
- Preprocessing statistics are fitted on the training split only. The schema pins `fitted_on` to
  `"train"`, so the mistake cannot be committed silently.
- The split definition and its seed live in `data/splits/` and are committed.
- Every committed run uses `seed: 42`. Models are seeded when they are built, so results do not
  depend on which notebook cells ran first.
- Parameter counts, model sizes, latencies and the hardware are measured for every model by one code
  path, in one session, and recorded in `resources.json`.
- Commit regularly — the assignment grades sustained development history.
