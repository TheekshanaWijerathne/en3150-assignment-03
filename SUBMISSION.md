# Final submission — the code notebook

The plan for turning this repository into the single notebook we upload as our code. Nothing here is
built yet. The build happens at the end (Stage 6, after the official runs), but **the conventions in
[§6](#6-how-internal-material-gets-removed) apply from today**, because following them while you
implement is what keeps the final step cheap.

1. [What we submit](#1-what-we-submit)
2. [Why a notebook works, and how](#2-why-a-notebook-works-and-how)
3. [Structure of the final notebook](#3-structure-of-the-final-notebook)
4. [What goes in, and what stays out](#4-what-goes-in-and-what-stays-out)
5. [What must never appear in the final notebook](#5-what-must-never-appear-in-the-final-notebook)
6. [How internal material gets removed](#6-how-internal-material-gets-removed)
7. [The build script](#7-the-build-script)
8. [Checks before uploading](#8-checks-before-uploading)
9. [Timeline](#9-timeline)
10. [Fallback](#10-fallback)

---

## 1. What we submit

The assignment asks for two uploads to Moodle, both named `<GroupNo>_A03_EN3150`:

| Upload | What it is | Graded? |
|---|---|---|
| **Report** (PDF) | the write-up. Carries every member's name and index number, and the repository link | **yes** — "I will use only the pdf for grading" |
| **Code** | "complete, well-documented (commented), and ready to run" | no, but it must meet that bar |

For the code, we submit **one self-contained notebook, `<GroupNo>_A03_EN3150.ipynb`**, that:

- contains **all** source code and configuration, as readable cells;
- shows the **official results inline** — the same numbers and figures as the report;
- **runs top to bottom in an empty folder**, with no repository and no `.py` files present;
- contains **nothing internal**: no member details, no team-workflow comments ([§5](#5-what-must-never-appear-in-the-final-notebook)).

The repository stays exactly as it is — our workspace, with every README, member role and test. We
don't upload it; the report links to it so the lecturer can see our commit history.

**The final notebook is generated, never edited by hand.** A fix goes into the repository, and the
notebook is rebuilt.

## 2. Why a notebook works, and how

Our structure makes this a mechanical step. The library in `src/edgecnn/` holds all the logic, and
the section notebooks only call it. So the final file is just **the library, the configs, and the
section notebooks' cells**, assembled in that order by a script.

**Self-extracting cells.** Each module and config file goes into the notebook as a `%%writefile`
cell holding its source:

```python
%%writefile src/edgecnn/data/loaders.py
"""Build the train, validation and test DataLoaders."""
...
```

Running these cells recreates `src/edgecnn/` and `configs/` inside a working folder, after which the
experiment cells import `edgecnn` exactly as they do today. Four properties make this safe:

- **Imports are unchanged,** so the submitted code is exactly the code we tested.
- **Paths still work.** Every path is derived from the package's own location, so recreating the
  repository's layout makes each one resolve.
- **The data regenerates.** The split is deterministic (fixed seed, stratified). A check compares the
  regenerated manifest's SHA-256 with the committed one, which proves it is the same split — so we
  never ship a 27,000-row CSV.
- **Outputs are copied, not recomputed.** Each experiment cell keeps its output from our official
  run, so the notebook shows exactly the numbers in the report. Re-running on a GPU could shift them
  slightly, which is why we don't.

**The alternatives:**

| Alternative | Problem |
|---|---|
| Results-only notebooks, no `.py` | they show results but cannot run, so they fail "ready to run" |
| Pasting code as plain cells | means rewriting imports across ~30 modules, with name clashes (`paths`, `schema`); the submitted code would no longer be the tested code |
| One notebook per section | works (a shared `library.ipynb` plus `%run` in each), but it's nine files and nine chances to go stale; one file is simpler to grade |

## 3. Structure of the final notebook

The experiments mirror the report's sections, so a reader can put the two side by side.

```
Title — EN3150 Assignment 03, group number
Summary — what the study does, in one paragraph
How to run — requirements, the MODE switch, expected runtime; outputs shown are from the reported runs

Part A · Setup (runs first)
  A1  Dependencies     %pip install, pinned to the versions used for the official runs
  A2  Workspace        creates ./a03_workspace and moves into it; sets MODE
  A3  Source code      one %%writefile cell per module, grouped by package
  A4  Configuration    one %%writefile cell per YAML file and JSON schema
  A5  Environment      import check, library versions, device

Part B · Experiments
  §1  Data preparation              from 01 — plus the split hash check
  §2  Custom architectures          from 02
  §3  Optimizer selection           from 03
  §4  Training and evaluation       from 04
  §5  Fine-tuning SOTA backbones    from 05
  §6  Resource benchmark            from 06
  §6  Final comparison              from 07
```

**Run modes in the final file:** `MODE = "official"` by default. Re-running it that way reproduces
the whole study: it downloads EuroSAT (~90 MB) and retrains everything, which takes hours on a CPU.
`MODE = "synthetic"` is a smoke test that finishes in about a minute.

**Merging is safe.** Part B runs every section in one kernel, but each section notebook already runs
on its own during our official runs. So every section defines everything it uses, and no section can
silently depend on a leftover variable from another.

## 4. What goes in, and what stays out

| In the final notebook (after cleanup) | Stays in the repository only |
|---|---|
| `src/edgecnn/**/*.py`, except `utils/colab.py` | `utils/colab.py` — it pushes results from Colab to GitHub, a team workflow tool. Its one import line in `utils/__init__.py` is dropped |
| `configs/**/*.yaml` and `configs/contracts/*.json`, with every `stages:` block removed — no code reads them; they document team ownership | `tests/` |
| the step cells of notebooks `01`–`07`, with their official outputs | `notebooks/00_setup` (reduced to the A5 environment cell), `notebooks/scratch/` |
| one clean title, summary and how-to-run | every README, `CONTRIBUTING.md`, `COLAB.md`, this file |
| | `pyproject.toml`, `.gitignore`, `data/`, `results/`, `artifacts/`, `report/` — regenerated when the notebook runs, or not needed |

**Dropped from every section notebook:** the title cell with its owner table; cell 1 (Colab
bootstrap); cell 2 (setup and `MODE`, replaced by Part A); `%autoreload`; the official-run checklist;
the discussion prompts; the Colab push cell.

## 5. What must never appear in the final notebook

All of this is fine in the repository. **None of it may appear in the final file — in code, comments,
docstrings, markdown or cell outputs:**

| Category | Examples |
|---|---|
| **Member details** | "Member 2", "M3", `member_1`, names, index numbers, GitHub usernames, emails, fork URLs |
| **Ownership and process** | "Owner:", "custodian", "shared file", "PR + 1 review", "group chat", "tell the team", "Phase 0", stages and gates, "promote", "scratch" |
| **Pointers to internal docs** | `CONTRIBUTING.md`, `COLAB.md`, `README.md`, `SUBMISSION.md`, `tests/…`, "notebook 03" |
| **Workflow machinery** | the Colab bootstrap, `push_results`, `IN_COLAB`, `google.colab`, `GH_TOKEN`, `A03_REPO` / `A03_BRANCH` |
| **Stub leftovers** | `NotImplementedError`, "implement …", "TODO", "stub" |
| **Team-only guidance** | discussion prompts, checklists, "Implementation notes for …", "seam" |

**What stays is reader-facing documentation:** what each function does, its arguments and return
values, and *why* each design choice was made — why macro averaging, why ReLU6, why 64×64. The code
must be "well-documented (commented)", so we remove the **internal part** of the documentation,
not the documentation itself.

## 6. How internal material gets removed

Three layers: **conventions** you follow while implementing, **automatic removal** of anything marked
internal, and **a gate** that refuses to build the file while anything from [§5](#5-what-must-never-appear-in-the-final-notebook)
remains.

### 6.1 Code — from the moment you implement a stub

- **Rewrite the stub's docstring for a reader.** The stubs' current docstrings are notes for us
  ("Implementation notes for …", who calls it, traps to avoid). When you implement a function, replace
  that with a summary, its `Args` / `Returns` / `Raises`, and the reasoning a reader needs. Keep the
  technical *why* — for example "measure a copy, so the caller's model keeps its train / eval
  mode" — and drop the *who*.
- **Team-only text you want to keep in the repository** goes at the **end** of the docstring, after a
  line that says exactly `Team notes:`. The build deletes everything from that line to the end of the
  docstring.
- **Comments explain why, never who.** A comment meant only for the team starts with `# team:`, and
  the build deletes the line.

```python
def compute_norm_stats(cfg):                      # today: a stub
    """Compute per-channel mean and std over the TRAIN split only.

    Returns:
        {"mean": [r, g, b], "std": [r, g, b]} over pixels scaled to [0, 1].

    Fitting on validation or test data would leak held-out information into
    training, so only the train split is used.

    Team notes:
    Member 1 owns this. The schema pins fitted_on to "train" - see configs/contracts.
    """
```

The final notebook keeps everything above `Team notes:`.

### 6.2 Configs

- Every `stages:` block is removed automatically.
- Team-only comment lines start with `# team:` and are removed.
- The header comment of each stage file (owner, notebook, code location) is team material. It gets
  rewritten, or marked with `# team:`, in the cleanup pass ([§9](#9-timeline)).
- JSON schema `description` fields cannot hold comments. Any that mention members or notebooks are
  rewritten in the cleanup pass.

### 6.3 Notebooks

- Cells meant only for the team get the **cell tag `internal`** — in VS Code, *Add Cell Tag* from the
  cell's `…` menu. The build drops them. The skeletons' existing internal cells are listed in
  [§4](#4-what-goes-in-and-what-stays-out) and get tagged once.
- Write step markdown for a reader: "the Model B run from §3", not "notebook 03" or "Member 3's run".

### 6.4 The gate

After assembly, the build scans the **whole** notebook — code, markdown **and outputs** — against the
[§5](#5-what-must-never-appear-in-the-final-notebook) list, plus a denylist file holding our names,
index numbers and GitHub usernames. **If anything matches, it prints every hit and refuses to write
the file.** The conventions keep the gate quiet; the gate catches what they miss.

**Scope today:** about 280 internal lines — 198 in code, 43 in configs, 40 in notebooks. Most sit in
stub docstrings and disappear when those stubs are implemented following [§6.1](#61-code--from-the-moment-you-implement-a-stub).
The rest are in the already-implemented plumbing (`contracts/`, `config/loader.py`,
`models/registry.py`, the package `__init__` files, the JSON schemas and stage-file headers), which
gets one cleanup pass before the build.

## 7. The build script

Planned as `tools/build_submission.py`. It is **not written yet** — that happens at Stage 6. What it
does:

1. **Checks preconditions** — a clean working tree, on the commit whose results the report uses; all
   six runs' official JSON present; every section notebook last run with `MODE = "official"`; no
   `NotImplementedError` left in `src/`.
2. **Collects the library** — every module under `src/edgecnn/` except `utils/colab.py`, with that
   module's import removed from `utils/__init__.py`.
3. **Cleans the source** — removes `Team notes:` sections from docstrings, parsing the code rather
   than using text search, so real code is never touched; removes `# team:` comment lines.
4. **Cleans the configs** — removes `stages:` blocks and `# team:` lines.
5. **Builds Part A** — dependencies pinned to the versions recorded during the official runs; the
   workspace folder; the `%%writefile` cells; the environment cell.
6. **Builds Part B** from notebooks `01`–`07`:
   - drops the cells listed in [§4](#4-what-goes-in-and-what-stays-out) and every cell tagged
     `internal`;
   - keeps each remaining cell's official output, and clears execution counts (they come from
     separate sessions);
   - removes `%autoreload`;
   - inserts the split hash check after §1, using the hash recorded in `data/splits/split_meta.json`.
7. **Adds** the title, summary and how-to-run cells, and the notebook metadata.
8. **Runs the gate** ([§6.4](#64-the-gate)).
9. **Writes** `dist/<GroupNo>_A03_EN3150.ipynb`. `dist/` is git-ignored: the file is a build product,
   rebuilt from the repository whenever anything changes.

## 8. Checks before uploading

- [ ] **The gate passes:** no member details or internal material anywhere, including cell outputs.
- [ ] **It is self-contained:** copy *only* the `.ipynb` into an empty folder and run it headless with
      `MODE = "synthetic"` (`jupyter nbconvert --to notebook --execute`). It must finish.
- [ ] **Every result cell shows its official output,** and every figure renders.
- [ ] **The numbers match:** accuracies and table values in the notebook equal
      `results/metrics/**/*.json` and the report's tables. The build can cross-check this.
- [ ] **The split hash check is present,** and it passed in the official run.
- [ ] **It opens cleanly** in Jupyter, VS Code and Colab.
- [ ] **The size is reasonable** — inline figures at ~100 DPI keep it to a few MB.
- [ ] **The dependency pins** match the versions `00_setup` printed during the official runs.
- [ ] **The filename is `<GroupNo>_A03_EN3150.ipynb`**, with the group number filled in.
- [ ] **Someone who didn't build it reads it top to bottom** and finds no team jargon ("seam",
      "stage", "gate", "promote").

## 9. Timeline

| When | What | Who |
|---|---|---|
| **From now** | follow [§6.1](#61-code--from-the-moment-you-implement-a-stub) as each stub is implemented; write step markdown for a reader | everyone, in their own files |
| **Once, early** | tag the internal cells in notebooks `00`–`07` as `internal` | each notebook's owner |
| **Official runs (Stage 5)** | keep `00_setup`'s printed library versions — they become the pins | Member 1 |
| **Stage 6** | cleanup pass on the plumbing, configs and schemas; write the build script; run the gate and fix every hit; run the self-containment check | one member builds; another reviews |
| **Before upload** | work through [§8](#8-checks-before-uploading); upload together with the report | whole group |

## 10. Fallback

If building the notebook runs short on time, upload the repository instead, as a zip of the final
commit. It is complete and ready to run, and its README explains how.

It would include our internal docs and member roles, though. If that's unacceptable, zip only
`src/`, `configs/` and notebooks `01`–`07`, plus a short run guide, and run the same gate over those
files first.
