# Developer guide

Four people, one pipeline, one comparison table at the end. Read this once before you start. After
that you will mostly need [§3 Your daily loop](#3-your-daily-loop) and
[§11 Troubleshooting](#11-troubleshooting).

1. [Setup](#1-setup-once)
2. [How development happens](#2-how-development-happens)
3. [Your daily loop](#3-your-daily-loop)
4. [Worked example — Member 2 builds Model B](#4-worked-example--member-2-builds-model-b)
5. [How work moves between members](#5-how-work-moves-between-members)
6. [Git with notebooks](#6-git-with-notebooks)
7. [Official runs](#7-official-runs)
8. [Colab fallback](#8-colab-fallback)
9. [What is and is not committed](#9-what-is-and-is-not-committed)
10. [Rules that protect the numbers](#10-rules-that-protect-the-numbers)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Setup (once)

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

**In VS Code:** open the repository folder, then

1. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`;
2. open any notebook → **Select Kernel** (top right) → **Python Environments** → `.venv`.

Then open [`notebooks/00_setup.ipynb`](notebooks/00_setup.ipynb) and run it top to bottom. Every cell
should pass, except the device cell, which waits for Member 1's Phase 0 work.

### What `pip install -e .` does, and why

Python can only `import edgecnn` if it knows where `edgecnn` lives. This command records that
location in your virtual environment, permanently. `-e` means **editable**: it records a *pointer to
your source folder*, not a copy. So:

- edit a `.py` file → the change is live on the next run, with no rebuild or reinstall;
- add, rename or delete a module → picked up automatically;
- `import edgecnn` works in every notebook, in the tests, and from any directory.

That is why no notebook ever needs `sys.path.append('..')`.

**Re-run the install only when `pyproject.toml` dependencies change.** That is the single trigger. If
an import fails after a pull, check you are in the right environment *before* reinstalling:

```powershell
python -c "import sys; print(sys.prefix)"      # should end in ...\EN3150-Assignment-03-CNN\.venv
```

### GPU note

The default install gives you CPU-only PyTorch, and every model trains on CPU — just more slowly.
For CUDA, install torch from the official index first, then the package:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

---

## 2. How development happens

Code lives in three places, each with one job:

| | `notebooks/scratch/` | `src/edgecnn/*.py` | `notebooks/0X_*.ipynb` |
|---|---|---|---|
| **What** | your experiments and prototypes | the shared library | the eight section notebooks |
| **Why** | try ideas fast, see results inline | one implementation everyone imports | run each step, see every result, produce the report's numbers |
| **Produces report results?** | never | no — notebooks call it | **yes**, in `MODE = "official"` |

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

### The promote rule

> Prototype in a scratch cell → move it into its stub in `src/edgecnn/` the moment it works →
> import it from then on.

Move it **immediately** when any of these is true:

1. someone else needs it — nobody can import a function from your notebook;
2. a second notebook needs it;
3. it is a `Dataset` or transform used with `num_workers > 0` — on Windows, DataLoader workers
   cannot load a class defined inside a notebook;
4. a contract test covers it (the parameter budget, the logits shape) — tests cannot import notebooks;
5. you are about to do an official run.

**Notebooks are never "converted to .py at the end".** They stay as the drivers and are submitted as
code. Only functions move, one at a time, as soon as they work. Converting everything in report week
would mean the reported numbers came from code that no longer exists in that form. It would also
mean four copies of the training loop drifting apart until the comparison stops being fair.

Moving code costs you nothing: every notebook runs `%autoreload 2`, so an edit to `model_b.py` takes
effect the next time you run a cell.

### Every stub already tells you what to build

Each function you own exists as a stub whose docstring states what it takes, what it returns, what it
writes, and the traps to avoid. **Keep the stub's signature.** Other members' notebooks are already
calling it with those arguments.

When you implement a stub, **rewrite its docstring for a reader**. The submitted code contains
none of our internal notes — no member details, no team process. Anything only the team needs goes
under a `Team notes:` line or a `# team:` comment. What the submission must never contain is in
[SUBMISSION.md §4](SUBMISSION.md#4-what-must-never-appear-in-it).

### `MODE` — one switch in every notebook

| `MODE` | Data | Epochs | Writes to `results/`? | Use it for |
|---|---|---|---|---|
| `"synthetic"` | generated fixture | 2 | no | before EuroSAT exists — runs in seconds |
| `"debug"` | 5% of EuroSAT | 2 | no | finding bugs step by step |
| `"official"` | full EuroSAT | 30 | **yes** | the results in the report |

The presets live in [`configs/base.yaml`](configs/base.yaml). Every library function **returns its
result in every mode** — so debug runs still show every plot — but **writes files only when the run
is official** (`cfg.is_official`). A debug run cannot overwrite a committed result.

---

## 3. Your daily loop

1. **Sync:**
   ```powershell
   git checkout main
   git pull --rebase origin main
   pytest tests/contracts
   ```
2. **Branch:** `git checkout -b m<n>/<topic>`, e.g. `m2/model-b`.
3. **Prototype** in `notebooks/scratch/m<n>_<topic>.ipynb`.
4. **Promote:** when it works, move it into its stub in `src/edgecnn/`, keeping the signature.
5. **Run it in your section notebook**, first with `MODE = "synthetic"`, then `"debug"`. Look at
   each step's output before running the next — that is the point of the notebook.
6. **Test:** `pytest`. Tests covering your function switch from *skipped* to real on their own, with
   no test edits needed.
7. **Commit** the library code and the notebook (outputs included), push, and open a PR. Get one
   review if you touched a shared file ([§6](#6-git-with-notebooks)).

---

## 4. Worked example — Member 2 builds Model B

**1 · Prototype.** In `notebooks/scratch/m2_model_b.ipynb`:

```python
import torch
from torch import nn

class ModelB(nn.Module):
    ...                                   # try an architecture

model = ModelB(num_classes=10)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))   # under 100,000?
model(torch.randn(2, 3, 64, 64)).shape                                  # (2, 10)?
```

Iterate until the parameter count and shapes are right.

**2 · Promote.** Move the class into [`src/edgecnn/models/custom/model_b.py`](src/edgecnn/models/custom/model_b.py)
and make `build_model_b(num_classes, input_shape, **overrides)` return it. `overrides` holds the
`model_b` section of [`configs/stages/models.yaml`](configs/stages/models.yaml) — `stem`, `blocks`,
`head`, `activation` — so widths come from the config, not from constants in the code.

**3 · Run it in the section notebook.** Open `notebooks/02_custom_architectures.ipynb` with
`MODE = "synthetic"` and run it. It builds the model with
`build_model_from_config(cfg_b, NUM_CLASSES)`, checks the output shape, prints the per-layer table
and asserts the budget.

**4 · Iterate without restarting.** Change a width in `model_b.py`, re-run the build cell. Autoreload
picks up the edit.

**5 · Test.** `pytest` — `tests/test_param_budget.py` and the Seam 2 tests in
`tests/contracts/test_seams.py` go from *skipped* to *passing*.

**6 · Commit and hand off.**

```powershell
git add src/edgecnn/models/custom/model_b.py notebooks/02_custom_architectures.ipynb
git commit -m "Implement Model B"
git push -u origin m2/model-b
```

Open a PR. Once it is merged, Member 3's notebook `03` trains the real Model B instead of the stub.

---

## 5. How work moves between members

Nobody reads anyone else's code. Work moves through five **seams**: fixed objects and files whose
formats are frozen in [`src/edgecnn/contracts/`](src/edgecnn/contracts/) and
[`configs/contracts/`](configs/contracts/).
The [development guide](DEVELOPMENT.md#the-five-seams) describes them.

| When this lands | Published by | It unblocks | Until then, others use |
|---|---|---|---|
| synthetic fixture + synthetic loader + plot style | M1 | every notebook in `synthetic` | nothing — this lands first (Phase 0) |
| shape-only `model_a`, `model_b` | M2 | `03`, `04` | nothing — this lands first (Phase 0) |
| real split `data/splits/*` (from `01`, official) | M1 | `debug` and `official` everywhere | `MODE = "synthetic"` |
| `Trainer` | M3 | training cells in `05` | the stub's contract; check your builders in synthetic |
| `evaluate_run`, `profile_all` | M1 | evaluation cells in `03`–`05`; notebook `06` | skip those cells for now |
| plot functions in `reporting.py` | M4 | confusion matrices in `04`, `05` | quick plots in your scratch notebook |
| official `history` / `test_metrics` JSON | M3, M4 | `06`, `07` | debug results in memory |

**You know something has landed** when its PR is merged and announced in the group chat. After
`git pull`, the related contract tests stop skipping, and the notebook cells that called the stub
start working.

**Nobody waits.** Until a piece lands, `MODE = "synthetic"` plus each stub's documented contract is
enough to build against.

---

## 6. Git with notebooks

### Branches and ownership

- Branch naming: `m<n>/<topic>` — `m1/eurosat-loader`, `m3/trainer`.
- **Edit only your own files and notebooks** ([development guide → work split](DEVELOPMENT.md#how-the-work-is-split)).
  Each notebook has one owner, so notebook merge conflicts should not happen.
- **One open branch per notebook at a time.** Two branches editing the same notebook is the one way
  to get a notebook conflict.

### Shared files — PR + 1 review, always

| File | Custodian |
|---|---|
| `src/edgecnn/contracts/**`, `configs/contracts/*.schema.json` | Member 1 |
| `src/edgecnn/models/registry.py` | Member 2 (Member 4 appends pretrained entries) |
| `configs/base.yaml`, `pyproject.toml` | Member 1 |
| `README.md`, `CONTRIBUTING.md` | whole team |

A change to one of these affects work already in progress for three other people. Post in the group
chat as well as opening the PR.

### Notebooks are committed *with* their outputs

That way everyone — including the lecturer — sees plots and metrics on GitHub without running
anything. Keep them small:

- inline figures render at ~100 DPI (set by `apply_style()`); the high-resolution PNG for the report is
  saved to `results/figures/` separately;
- never print in bulk — no whole DataFrames, no per-batch logs; training shows one `tqdm` line;
- print repo-relative paths, never `C:\Users\<you>\...`.

Before committing a notebook's **official** run: Kernel → **Restart & Run All**, and check that it
finished without errors.

### Reviewing

Review the `.py` diff as normal. For a notebook, open it on GitHub, which renders cells and outputs.
Its raw JSON diff is unreadable.

### If a notebook conflict happens anyway

Never hand-merge notebook JSON. Keep the owner's version and re-run it:

```powershell
git checkout --theirs notebooks/03_optimizer_study.ipynb   # or --ours, whichever is the owner's
git add notebooks/03_optimizer_study.ipynb
```

### Pulling in someone else's work

```powershell
git checkout main
git pull --rebase origin main
pip install -e ".[dev]"          # ONLY if pyproject.toml changed
pytest tests/contracts           # did their change break an interface you depend on?
git checkout m3/my-branch
git rebase main                  # bring your branch up to date
```

To check whether `pyproject.toml` changed in that pull:

```powershell
git diff HEAD@{1} --name-only | Select-String pyproject.toml
```

**If `pytest tests/contracts` fails after a pull,** someone changed an interface. Find out who with
`git log --oneline -5 -- src/edgecnn/contracts/ configs/`, then **tell them**. Do not quietly adapt
your code to it. If the change was agreed, update your side.

Use `git pull --rebase`, not a plain merge. The assignment grades the development history, and a log
full of merge commits obscures it. **Commit regularly** for the same reason.

---

## 7. Official runs

Work moves through stages. Each opens only when the previous gate is met:

| Stage | What | Gate to move on |
|---|---|---|
| 2 · Phase 0 | synthetic fixture, shape-only models, trainer on synthetic data | every section notebook runs top to bottom in `synthetic` |
| 3 · Build | scratch → promote → section notebook in `debug` | your notebook runs in `debug` on real EuroSAT |
| 4 · Dry run | the whole chain `01` → `07` in `debug`, on one machine, together | it passes; real epoch times show who needs Colab |
| 5 · Official runs | each owner runs their notebooks in `official` | all six runs' JSON committed and schema-valid |
| 6 · Report | everyone writes their section | — |

**Official run order:** `01` → `02` → `03` → `04` → `05` → `06` → `07`.
`04` reuses Model B's run from `03`; `06` needs every run's `history.json`; `07` needs everything.

**Hardware rules** — the comparison tables are only fair if these hold:

- `03` and `04` run on the **same machine or runtime**, because their epoch times share a table;
- `05`'s two backbones run on the same runtime;
- `06` runs **once, in one session, on a CPU**, after `03`–`05` are committed, so every latency is
  measured on the same hardware. Use a local machine and name its CPU in the report.

**Unattended runs:** to run an official notebook without keeping VS Code open, use

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/03_optimizer_study.ipynb
```

This runs it headless and saves the outputs into the notebook, just like Restart & Run All.

Each notebook ends with an **official-run checklist**. After a run, commit the notebook together with
every file it wrote, then announce it.

---

## 8. Colab fallback

Everything runs locally by default. If a run is too slow on your laptop — the Stage 4 dry run shows
you — run the **same notebook** on Colab's free GPU. Its first cell sets Colab up and does nothing on
your laptop.

- **Development runs:** VS Code with the Google Colab extension. The notebook stays local; only the
  kernel moves.
- **Official runs:** Colab in the browser, so the last cell can push the results with your token.

The full step-by-step guide is **[COLAB.md](COLAB.md)**: one-time setup, both paths, which notebooks
belong on Colab, and troubleshooting. Three rules to remember:

- Colab only sees what you have **pushed**.
- **Never type or print your GitHub token** in a notebook. Outputs are committed, and the repository
  is public.
- Your fork and branch go in your Colab Secrets (`A03_REPO` / `A03_BRANCH`), never in the notebook.
  If you edit cell 1's defaults for a VS Code run, put them back before committing.

Each member uses their own Colab — nothing is shared except the repository.

---

## 9. What is and is not committed

| Committed | Ignored |
|---|---|
| `src/**`, `configs/**`, `tests/**` | `.venv/`, `__pycache__/` |
| `notebooks/**/*.ipynb` **with outputs** | `data/raw/**`, `data/processed/**` |
| `data/splits/*.csv`, `data/splits/*.json` | `artifacts/checkpoints/**` (including `_debug/`) |
| `results/metrics/**/*.json`, `results/figures/**`, `results/tables/**` | `tests/fixtures/synthetic/` (generated) |

Two of these matter for more than tidiness:

- **Metrics JSON is committed.** Member 4 builds the §3, §4 and §6 tables from everyone's
  `test_metrics.json` and `resources.json`. If those existed only on the machine that produced them,
  the report would need every experiment re-run in its final week.
- **The split is committed.** `data/splits/split_manifest.csv` guarantees all four models saw
  byte-identical data. `split_meta.json` stores its SHA-256, so a regenerated split is detectable.

---

## 10. Rules that protect the numbers

Break one of these and a number in the report is wrong in a way that is hard to spot later.

1. **Only `MODE = "official"` writes results.** Never hand-edit a file in `results/`, and never
   commit output from a synthetic or debug run.
2. **Restart & Run All** before committing an official notebook. The committed outputs must come from
   one clean top-to-bottom run.
3. **Hyperparameters live in `configs/`**, never typed into a notebook cell. Each setting has exactly
   one home, and an optimizer is defined only in its experiment file.
4. **Build models with `build_model_from_config`.** It seeds, and it guarantees the model trained in
   `03`–`05` is the same architecture profiled in `06`.
5. **Never read the test split during training.** Only `evaluate_run` touches it, once, after training.
6. **Never redraw the split.** It is committed. `prepare_dataset` refuses to overwrite it without
   `force=True`.
7. **Never compute a metric yourself** for the report. Use `evaluate_run` and `profile_all`, so every
   model is measured by one code path.
8. **Models return raw logits** — no softmax inside a model.
9. **Latency is measured on CPU**, for every model, in one session (notebook `06`).

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'edgecnn'` | the notebook kernel is not your `.venv` | **Select Kernel** → `.venv`. Check with `import sys; sys.prefix` |
| an edit to a `.py` file seems ignored | the old object is still in memory | re-run the cell that *builds* the object; if it persists, restart the kernel |
| `Can't get attribute 'X' on <module '__main__'>` (Windows) | a `Dataset` or transform is defined in a notebook cell | promote it into `src/edgecnn/` ([§2](#the-promote-rule)) |
| `NotImplementedError: Member N: implement ...` | that member's part has not landed yet | use `MODE = "synthetic"`, skip the cell, or ask them |
| `ContractViolation` | an artifact does not match its schema — an interface broke | tell the owner of the stage that produced it ([§6](#pulling-in-someone-elses-work)) |
| `pytest tests/contracts` fails right after a pull | someone changed a shared contract | see who with `git log`, and tell them |
| a notebook merge conflict | two branches edited the same notebook | keep the owner's version and re-run it ([§6](#if-a-notebook-conflict-happens-anyway)) |
| training is far too slow | laptop CPU | use `debug` for development; see [§8 Colab](#8-colab-fallback) for official runs |
