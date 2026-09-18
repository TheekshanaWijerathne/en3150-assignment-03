# `report/`

The submitted document. Filename format is mandated: **`<GroupNo>_A03_EN3150.pdf`**.

Code and report are uploaded **separately** to Moodle. Only one submission per group.

## Who writes what

Report sections follow code ownership, so each member writes up the work they did and defends
their own marks.

| Section | Marks | Author | Draws on |
|---|---|---|---|
| Dataset, splits, preprocessing, reproducibility | — | M1 | `data/splits/split_meta.json` |
| §2 Custom architecture design | **20** | M2 | `results/tables/custom_architectures.md`, `param_breakdown.md` |
| §3 Optimizer selection and tuning | **15** | M3 | `results/tables/optimizer_comparison.md`, `figures/optimizer_overlay.png` |
| §4 Training and evaluation | **25** | M3 + M1 | `figures/<run_id>/curves.png`, `tables/custom_model_comparison.md` |
| §5 SOTA fine-tuning | **20** | M4 | `results/metrics/mobilenet_v2__*`, `squeezenet1_1__*` |
| §6 Final comparison and trade-offs | **20** | M4 | `results/tables/final_comparison.md`, trade-off scatter |

Must also include: **every member's name and index number**, and the **GitHub repository link**.

## `figures/`

Do not put anything here by hand. `scripts/build_report_assets.py` mirrors
`results/figures/` into this folder, so there is exactly one source of truth and figures cannot
drift out of sync with the numbers in the tables.

## What actually earns the marks

The assignment says plainly: *"The interpretation of results and the discussion are important."*
Numbers alone score poorly. Specific things to make sure the report does:

- **§2** — state kernel sizes, filter counts and FC widths explicitly, and **derive** the total
  trainable parameters by hand rather than pasting a summary dump. Justify the activation choice on
  hardware grounds, not on convention.
- **§3** — discuss the momentum parameter's effect on convergence **and** on final performance
  separately. They are different claims and can disagree.
- **§4** — discuss the trade-offs observed *moving from standard to depthwise separable*
  convolutions, in both directions: what accuracy was lost, and what was bought.
- **§6** — balance accuracy, memory footprint and computational cost. Points worth making:
  - State the *ratio*, not just both numbers ("40× fewer parameters for 6 points of accuracy").
  - Be fair about the 64×64 handicap on the SOTA backbones — they were designed for 224×224, and
    MobileNetV2's 32× stride leaves a 2×2 feature map here. Claiming a clean win for the custom
    model without mentioning this is a weaker answer, not a stronger one.
  - Note that pretrained weights matter most when data is scarce. EuroSAT's 27,000 images are
    enough to train from scratch; at 2,700 the conclusion would likely flip.
  - Peak **activation** memory, not parameter count, is often what actually blocks MCU deployment.
    A model can fit in flash and still not run.
  - "It depends on the memory budget" is a strong conclusion **if** you identify the threshold at
    which the answer flips. Declaring a winner without that is weaker.

## Plagiarism

Checked, with a 50% penalty on top of the mark. Copying between groups gives both parties zero.
Write your own prose; cite anything borrowed.
