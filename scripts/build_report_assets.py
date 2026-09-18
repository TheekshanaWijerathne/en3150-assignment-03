"""Aggregate every committed run into the report's tables and figures.

Owner: Member 4.   Stage: evaluation   (SEAM 5 producer)   Sections 4/5/6

    IN   results/metrics/*/history.json        (Member 3)
         results/metrics/*/test_metrics.json   (Member 1)
         results/metrics/*/resources.json      (Member 1)
    OUT  results/tables/custom_model_comparison.md    Section 4
         results/tables/optimizer_comparison.md       Section 3
         results/tables/final_comparison.md           Section 6
         results/figures/*.png  ->  report/figures/

    python scripts/build_report_assets.py --config configs/stages/evaluation.yaml

Reads only committed JSON, so it runs on a laptop with no GPU, in seconds,
without re-running anyone's training. That is why results/metrics/**/*.json is
committed while artifacts/checkpoints/ is not.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stages/evaluation.yaml")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any expected run is missing, instead of warning.")
    args = parser.parse_args()

    # Member 4:
    #   runs = load_all_runs()
    #   build_custom_comparison_table()      Section 4
    #   build_optimizer_comparison_table()   Section 3
    #   build_final_comparison_table()       Section 6
    #   plot_confusion_matrix(run_id) for each
    #   plot_tradeoff_scatter("trainable_params"); plot_tradeoff_scatter("macs")
    #   export_report_figures()
    #
    # Without --strict, warn and skip a run whose JSON is missing rather than
    # crashing: for most of the project only some of the team's runs have
    # landed, and this script should stay usable throughout. Print a summary of
    # what was found and what was skipped so gaps are visible before the
    # report is written, not after.
    raise NotImplementedError("Member 4: implement scripts/build_report_assets.py")


if __name__ == "__main__":
    raise SystemExit(main())
