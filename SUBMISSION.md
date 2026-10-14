# Final submission: the code folder

How the code we upload is built, checked and run. The report is a separate upload.

1. [What we submit](#1-what-we-submit)
2. [What the folder holds](#2-what-the-folder-holds)
3. [How it is built](#3-how-it-is-built)
4. [What must never appear in it](#4-what-must-never-appear-in-it)
5. [The official run](#5-the-official-run)
6. [Checks before uploading](#6-checks-before-uploading)

---

## 1. What we submit

The assignment asks for two uploads to Moodle, both named after the group. Ours is named, not
numbered: **Misclassified**.

| Upload | What it is | Graded? |
|---|---|---|
| **Report**: `Misclassified_A03_EN3150.pdf` | the write-up, with every member's name and index number and the repository link | **yes**: "I will use only the pdf for grading" |
| **Code**: `Misclassified_A03_EN3150.zip` | the folder `submission/Misclassified_A03_EN3150/`, zipped | no, but it must be "complete, well-documented (commented), and ready to run" |

The repository stays as it is: our workspace, with every README, member role and test. The
submission's README links to it.

## 2. What the folder holds

```
submission/Misclassified_A03_EN3150/
├── README.md               run order, where to run, run times, results, reproducibility, references
├── requirements.txt
├── 01_data_preparation.ipynb ... 07_final_comparison.ipynb
└── results/                written by the notebooks, committed: the numbers in the report
```

Each notebook contains **all the code it runs**, as readable, commented cells: nothing is imported
from `edgecnn` or from any file. It has none of the development machinery: no `MODE`, no synthetic
or debug data, no Colab cells, no YAML configs or JSON schemas (the settings are plain values in the
code), no member details. The notebooks pass results to each other through `results/`; the training
notebooks prepare the data themselves, so each can also run on its own.

`data/` (the dataset and a pixel cache) and `checkpoints/` (trained weights) are written when the
notebooks run and are git-ignored.

## 3. How it is built

**The notebooks are generated, never edited by hand.**

| Source | What it holds |
|---|---|
| `tools/submission/parts/*.md` | the implementation, one part per topic (data, models, training, evaluation, benchmark, tables), as Markdown with Python code blocks |
| `tools/submission/notebooks/*.md` | the seven notebooks: their text, their experiment steps, and `<!-- include: part -->` lines |
| `tools/submission/README.md`, `requirements.txt` | the folder's README (its environment lines are filled in from the executed notebooks) |
| `tools/submission/denylist.txt` | member names, usernames and emails, for the gate; never copied |

```powershell
python tools/build_submission.py            # write the notebooks, README.md and requirements.txt
python tools/build_submission.py --docs     # rewrite only README.md and requirements.txt
python tools/build_submission.py --check    # the gate and run status, outputs included
python tools/check_submission.py            # the parts compute exactly what edgecnn computed
```

A notebook that already has outputs is only overwritten with `--force`, so an executed run is never
lost by accident. Every notebook that uses a part gets the same copy of it, so the copies cannot
drift apart.

**The parts reproduce `edgecnn` exactly.** `tools/check_submission.py` runs the parts, as the
notebooks do, next to the library that produced the reported results, on the real dataset, and
compares: the split fingerprint, every decoded pixel, the normalisation statistics bit for bit, the
first batches with augmentation, every model's initial weights and frozen layers, a short training
run weight for weight, every metric, every cost measurement, and all five tables character for
character. Run it after changing a part.

One property of the reported runs is kept on purpose: in `03` and `05`, the runs share one set of
data loaders, so the training loader's shuffling continues from one run to the next. The runs start
from identical weights with the same seed and schedule, but **do not see their batches in the same
order**; the report must not claim they do.

## 4. What must never appear in it

`tools/build_submission.py` scans every text in the folder (code, Markdown, printed outputs, tables,
JSON; not embedded images) and refuses to pass while it finds any of these:

| Category | Examples |
|---|---|
| **Banned characters** | the em-dash, the section sign, the middle dot |
| **Member details** | names, GitHub usernames, emails, index numbers (6 digits and a letter) |
| **Development features** | `MODE`, synthetic data, `edgecnn`, Colab cells and secrets, autoreload, "Member 2", "Team notes", `TODO`, `NotImplementedError` |
| **Machine paths** | drive letters, `/home/`, `/Users/`, `/content/`, `/root/` |

The repository link in the README is the one allowed exception: it contains a username.

## 5. The official run

The training notebooks run on a Colab T4, like the reported runs, in **one session**, so they share
their files; the benchmark and the tables run on the laptop.

1. **Colab:** open `tools/colab_run_submission.ipynb` from your fork, choose a T4 GPU, Run all. It
   clones the repository, runs notebooks `01` to `05` in place with `tools/run_submission.py`, then
   pushes the executed notebooks and `results/` back. About 40 minutes.
2. **Laptop:** `git pull`, then, on a quiet machine:

   ```powershell
   .\.venv\Scripts\python tools\run_submission.py 06 07
   .\.venv\Scripts\python tools\build_submission.py --docs
   ```

   `--docs` fills in the README's environment lines and runs the gate.
3. Check that every accuracy equals the development results in `results/metrics/`: on the same GPU
   and versions, the training is exactly repeatable.

`python tools/run_submission.py --smoke DIR` runs every notebook in a copy in `DIR` with one epoch
per training run: a quick check that the notebooks run end to end, without touching the folder.

## 6. Checks before uploading

- [ ] `python tools/build_submission.py --check` passes, and every notebook shows `ran`.
- [ ] The accuracies in the notebooks equal those in the report's tables.
- [ ] The README's environment lines name the T4 and the laptop CPU.
- [ ] The README's repository link opens, and the repository is public.
- [ ] Someone who did not build it opens each notebook and finds no team jargon.
- [ ] The zip holds the folder without `data/` and `checkpoints/`, and is named
      `Misclassified_A03_EN3150.zip`.
