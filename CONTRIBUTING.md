# Working on this repository

Four people, one pipeline, one comparison table at the end. This document covers setup, how to pull
in each other's work without breaking your own, and the rules that keep the final numbers
defensible.

---

## 1. First-time setup

Once per member, per machine.

```powershell
git clone https://github.com/ThejithaR/EN3150-Assignment-03-CNN.git
cd EN3150-Assignment-03-CNN

python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
# .venv\Scripts\activate.bat          # Windows cmd
# source .venv/bin/activate           # macOS / Linux

pip install --upgrade pip
pip install -e ".[dev]"
```

Verify:

```powershell
python -c "import edgecnn; print(edgecnn.__version__)"     # -> 0.1.0
pytest tests/contracts -q                                  # -> all green
```

### What `pip install -e .` actually does, and why

Python can only `import edgecnn` if it knows where `edgecnn` lives. This command writes that
location into your virtual environment, permanently.

`-e` means **editable**. It records a *pointer to your source folder*, not a copy. So:

- Edit a `.py` file → the change is live on the next run. **No rebuild, no reinstall.**
- Add a new module, rename one, delete one → picked up automatically.
- `import edgecnn.training` works from `scripts/`, from `tests/`, from a notebook in
  `notebooks/`, and from any directory on your machine.

Without it you would be relying on Python's fallback of searching the current working directory,
which breaks the moment you open a notebook — a notebook's working directory is `notebooks/`, not
the repo root — and the usual fix (`sys.path.append('..')`) gets committed and then breaks for
whoever's folder depth differs.

### Re-running the install

**Only when `pyproject.toml` dependencies change.** That is the single trigger.

If an import fails after a pull, check you are in the right virtual environment *before*
reinstalling — an inactive `.venv` is the more common cause.

```powershell
python -c "import sys; print(sys.prefix)"      # should end in ...\EN3150-Assignment-03-CNN\.venv
```

### GPU note

The default `pip install` gives you CPU-only PyTorch. For CUDA, install torch first from the
official index, then install this package:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

Everything runs on CPU — slower, but nothing is GPU-only. **Inference latency must be measured on
CPU for every model regardless**, because the §6 argument is about edge devices and edge devices
have no GPU.

---

## 2. Getting other people's changes

The routine after someone merges to `main`.

```powershell
# 1. update main
git checkout main
git pull --rebase origin main

# 2. ONLY if pyproject.toml changed in that pull
pip install -e ".[dev]"

# 3. did their change break a seam you depend on?
pytest tests/contracts -q

# 4. bring your in-progress branch up to date
git checkout m3/my-feature
git rebase main
```

### Did `pyproject.toml` change?

```powershell
git diff HEAD@{1} --name-only | Select-String pyproject.toml
```

Nothing printed → no reinstall needed.

### If `pytest tests/contracts` fails after a pull

That is the suite doing its job: **someone changed an interface you depend on.**

1. Find out who and what: `git log --oneline -5 -- src/edgecnn/contracts/ configs/contracts/`
2. **Tell them.** Do not silently adapt your code to a mutated schema — if the change was
   unintentional, quietly working around it means the break surfaces again later, in the numbers.
3. If the change was agreed, update your side and move on.

The suite is designed to be **green at all times, even on unimplemented stubs**. A test whose
subject still raises `NotImplementedError` reports as *skipped*, and turns into a real assertion
automatically the moment that stub is filled in. So red genuinely means something broke — which is
only useful if it is normally green.

### Rebase, not merge

`git pull --rebase` keeps the history linear and readable. The assignment grades sustained
development history, and a log full of "Merge branch 'main' into..." commits obscures it.

---

## 3. Branches and reviews

```
m<n>/<topic>          m1/eurosat-loader   m2/model-b   m3/trainer   m4/mobilenet
```

Work on your own branch, open a PR into `main`.

### Files you may edit

Your own, per the ownership table in the [README](README.md#how-the-work-is-split). The split is
designed so **no two members edit the same file**, which means merge conflicts should be close to
zero. If you need to change something you do not own, ask the owner — do not edit it on your branch.

### Shared files — PR + 1 review, always

| File | Custodian |
|---|---|
| `src/edgecnn/contracts/**` | Member 1 |
| `configs/contracts/*.schema.json` | Member 1 |
| `src/edgecnn/models/registry.py` | Member 2 (Member 4 appends pretrained entries) |
| `configs/base.yaml` | Member 1 |
| `pyproject.toml` | Member 1 |
| `README.md` | whole team |

A change to any of these affects work already in flight for three other people. Post in the group
chat as well as opening the PR — a review notification is easy to miss, and a silently changed
contract is the single most expensive failure mode in this project.

### Commits

Commit regularly and in small pieces. The assignment explicitly grades development history over
time, and a single large commit the night before submission is visible and counts against you.

```
m2: implement depthwise separable block
m2: Model B under budget at 94,312 params
m1: stratified split + manifest writer
```

---

## 4. What is committed and what is not

| Committed | Ignored |
|---|---|
| `configs/**`, `src/**`, `scripts/**`, `tests/**` | `.venv/`, `__pycache__/` |
| `data/splits/*.csv`, `data/splits/*.json` | `data/raw/**`, `data/processed/**` |
| `results/metrics/**/*.json` | `artifacts/checkpoints/**` |
| `results/figures/**`, `results/tables/**` | `*.pt`, `*.pth` |

### Two of these are load-bearing, not preferences

**Metrics JSON is committed.** Member 4 cannot build the §4, §5 or §6 tables without every other
member's `test_metrics.json` and `resources.json`. If those lived only on the machine that produced
them, assembling the report would mean re-running every experiment — on a laptop, the week it is
due. Committing them means `scripts/build_report_assets.py` runs in seconds, anywhere, with no GPU.

**The split definition is committed.** `data/splits/split_manifest.csv` is what guarantees Model A,
Model B, MobileNetV2 and SqueezeNet saw byte-identical data. `split_meta.json` carries its SHA-256,
so a silently regenerated split is detectable.

Checkpoints are excluded because they are large and fully regenerable. Raw and processed images are
excluded because they are a download plus a deterministic transform.

---

## 5. Running things

```powershell
# Member 1 — once, then commit the three files it writes
python scripts/prepare_data.py --config configs/stages/data.yaml

# the synthetic fixture — no download, seconds to run, unblocks M3 and M4
python scripts/prepare_data.py --config configs/stages/data.yaml --synthetic

# Member 2
python scripts/summarize_models.py --check-budget

# Member 3
python scripts/train.py --config configs/experiments/model_b__adam.yaml
python scripts/run_optimizer_study.py

# Member 1 — for every model, so all rows come from one code path
python scripts/evaluate.py --config configs/experiments/model_b__adam.yaml

# Member 4 — after runs land
python scripts/build_report_assets.py
```

### Fast iteration

```powershell
python scripts/train.py --config configs/experiments/model_b__adam.yaml `
    --epochs 2 --subset-fraction 0.05
```

Results from a run with `subset_fraction < 1.0` or fewer than 20 epochs **must not be committed**.
`scripts/train.py` refuses to write them, and `history.schema.json` rejects a run with fewer than
20 epochs anyway.

---

## 6. Rules that protect the final numbers

These are not style preferences. Each one, if broken, makes a number in the report wrong in a way
that is hard to detect afterwards.

1. **Never read the test split during training.** The trainer must not construct a loader over
   `DataBundle.test`, not even to print a number. Member 1's `scripts/evaluate.py` is the only
   thing that touches it.
2. **Never redraw the split.** It is committed. `scripts/prepare_data.py` requires `--force` to
   overwrite, and doing so invalidates every result already in `results/metrics/`.
3. **Fit preprocessing on train only.** The schema pins `fitted_on: "train"`.
4. **One seed — 42 — for every committed run.** Different seeds across runs make the comparison
   table incoherent.
5. **Never hand-build an artifact path.** Use `edgecnn.contracts.paths`.
6. **Never compute a metric yourself for the report.** Use Member 1's `compute_metrics` and
   `profile_model`. Four implementations of "precision" differ in averaging and zero-division
   handling, and the resulting table looks perfectly plausible while being wrong.
7. **Models return raw logits.** No softmax inside a model.
8. **Measure inference latency on CPU**, for every model, even on a CUDA machine.

---

## 7. If you are stuck

- **"What am I supposed to build?"** — the `README.md` in the folder you own. Every one names its
  owner, inputs and outputs.
- **"What shape does X arrive in?"** — [`src/edgecnn/contracts/types.py`](src/edgecnn/contracts/types.py)
  and the `contract:` block in your `configs/stages/*.yaml`.
- **"Where do I write my output?"** — [`src/edgecnn/contracts/paths.py`](src/edgecnn/contracts/paths.py).
- **"Is my output correct?"** — write it through `schema.write_json`; it validates before writing.
- **"Am I blocked on someone?"** — you should not be after Phase 0. Use
  `dataset.name: synthetic` and Member 2's model stubs.
