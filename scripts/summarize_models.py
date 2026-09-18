"""Print and tabulate the custom architectures.

Owner: Member 2.   Stage: models   Section 2 [20 marks]

    IN   configs/stages/models.yaml
    OUT  results/tables/custom_architectures.md
         results/tables/param_breakdown.md

    python scripts/summarize_models.py --config configs/stages/models.yaml

Section 2 asks for the network parameters to be detailed explicitly and the
total trainable parameters calculated - so this produces a PER-LAYER table, not
just a total. Member 2 should also derive the same numbers by hand in the
report; this script is the check on that arithmetic, not a substitute for it.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stages/models.yaml")
    parser.add_argument("--models", nargs="*", default=["model_a", "model_b"])
    parser.add_argument("--check-budget", action="store_true",
                        help="Exit non-zero if model_b exceeds 100,000 trainable params.")
    args = parser.parse_args()

    # Member 2:
    #   for each model: build_model(name, num_classes, input_shape)
    #     - per-layer table: layer, output shape, params, MACs
    #     - totals, and the Model A / Model B ratio
    #     - assert model_b <= MODEL_B_PARAM_BUDGET
    #
    # Print the layer where each model spends the most parameters. For Model A
    # that is the flatten into the first FC layer, and naming it explicitly is
    # a stronger Section 2 answer than only comparing totals.
    raise NotImplementedError("Member 2: implement scripts/summarize_models.py")


if __name__ == "__main__":
    raise SystemExit(main())
