"""Run the three-optimizer comparison.

Owner: Member 3.   Section 3 [15 marks]

    IN   configs/stages/training.yaml -> optimizer_study
    OUT  three runs under results/metrics/:
           model_b__sgd__seed42/
           model_b__sgd_momentum__seed42/
           model_b__adam__seed42/
         results/figures/optimizer_overlay.png
         results/tables/optimizer_comparison.md

    python scripts/run_optimizer_study.py --config configs/stages/training.yaml

Equivalent to running scripts/train.py three times with different optimizer
configs, but reseeds between variants so all three start from identical
weights. Without that the comparison measures initialisation noise alongside
the optimizer.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stages/training.yaml")
    parser.add_argument("--model", default=None, help="Override the study model.")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Subset of variants, e.g. --variants sgd adam")
    args = parser.parse_args()

    # Member 3:
    #   run_optimizer_study(cfg) -> {optimizer_name: TrainResult}
    #   summarize_study(results)
    #   plot_optimizer_overlay([...run_ids...])
    #
    # Hold EVERYTHING except the optimizer fixed: same seed, same epochs, same
    # scheduler, same augmentation. Log the held-fixed settings at the start so
    # the report can state plainly that only one variable changed.
    raise NotImplementedError("Member 3: implement scripts/run_optimizer_study.py")


if __name__ == "__main__":
    raise SystemExit(main())
