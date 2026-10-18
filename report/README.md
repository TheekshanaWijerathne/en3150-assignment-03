# `report/`

The submitted document. The filename format is mandated: **`<GroupNo>_A03_EN3150.pdf`**; our group is named, not
numbered, so ours is **`Misclassified_A03_EN3150.pdf`**.

The code and the report are uploaded **separately** to Moodle, once per group. The code submission is
the repository — the section notebooks, committed with their outputs, are its readable,
ready-to-run record.

## Where each section's material comes from

Report sections follow notebook ownership, so each member writes up the work they did and defends
their own marks.

| Section | Marks | Author | Notebook | Material it produces |
|---|---|---|---|---|
| Dataset, splits, preprocessing, reproducibility | — | M1 | `01` | split table, class distribution, sample grid, `split_meta.json` |
| §2 Custom architecture design | **20** | M2 | `02` | `tables/custom_architectures.md`, `param_breakdown.md`, activation notes |
| §3 Optimizer selection and tuning | **15** | M3 | `03`, `07` | per-run curves, `figures/optimizer_overlay.png`, `tables/optimizer_comparison.md` |
| §4 Training and evaluation | **25** | M3 + M1 | `03`, `04`, `06`, `07` | curves, confusion matrices, `tables/custom_model_comparison.md` |
| §5 SOTA fine-tuning | **20** | M4 | `05`, `06` | curves, confusion matrices, both parameter counts, size in MB |
| §6 Final comparison and trade-offs | **20** | M4 | `06`, `07` | `tables/final_comparison.md`, accuracy-vs-cost figures |

Each notebook ends with **discussion prompts** for its section — a starting point for the write-up.

The report must also include **every member's name and index number**, and the **GitHub repository
link**.

## `figures/`

Do not put anything here by hand. Notebook `07` copies `results/figures/` into it, so there is one
source of truth, and figures cannot drift out of sync with the numbers in the tables.

## What actually earns the marks

The assignment says plainly: *"The interpretation of results and the discussion are important."*
Numbers alone score poorly. Make sure the report does the following:

- **§2** — state kernel sizes, filter counts and FC widths explicitly, and **derive** the total
  trainable parameters by hand rather than pasting a summary dump. Justify the activation choice on
  hardware grounds, not on convention.
- **§3** — discuss the momentum parameter's effect on convergence **and** on final performance
  separately. They are different claims and can disagree.
- **§4** — discuss the trade-offs of *moving from standard to depthwise separable* convolutions, in
  both directions: what accuracy was lost, and what was bought.
- **§6** — balance accuracy, memory footprint and computational cost:
  - Give ratios, not just both numbers ("40× fewer parameters for 6 points of accuracy").
  - Be fair about the 64×64 handicap on the SOTA backbones. They were designed for 224×224, and
    MobileNetV2's 32× stride leaves a 2×2 feature map here. Claiming a clean win for the custom model
    without saying this makes the argument weaker, not stronger.
  - Pretrained weights matter most when data is scarce. EuroSAT's 27,000 images are enough to train
    from scratch; at 2,700 the conclusion might flip.
  - Peak **activation** memory, not parameter count, is often what actually blocks deployment on a
    microcontroller. A model can fit in flash and still fail to run.
  - "It depends on the memory budget" is a strong conclusion only **if** you identify the threshold
    where the answer flips. Declaring a winner without that is weaker.

## Plagiarism

Plagiarism is checked, with a 50% penalty on top of the mark. Copying between groups gives both
groups zero. Write your own prose, and cite anything you borrow.
