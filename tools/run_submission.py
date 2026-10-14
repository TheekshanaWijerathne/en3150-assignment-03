"""Run the submission notebooks in place, in order, keeping their outputs.

    python tools/run_submission.py 01 02 03 04 05     the training notebooks: on a GPU (Colab)
    python tools/run_submission.py 06 07              the benchmark and the tables: on the laptop
    python tools/run_submission.py --smoke DIR        a quick check of every notebook, in a copy
                                                      in DIR with one epoch per training run

Each notebook runs in a fresh kernel with the submission folder as its working directory, so they
share data/, checkpoints/ and results/ exactly as a reader running them in order would. What the
notebooks print, such as each training epoch, is echoed here as it happens.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "submission" / "Misclassified_A03_EN3150"


class EchoingClient(NotebookClient):
    """A notebook client that also prints the notebook's text output while it runs."""

    def output(self, outs, msg, display_id, cell_index):
        if msg.get("msg_type") == "stream":
            sys.stdout.write(msg["content"].get("text", ""))
            sys.stdout.flush()
        return super().output(outs, msg, display_id, cell_index)


def save(notebook: nbformat.NotebookNode, path: Path) -> None:
    text = nbformat.writes(notebook)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")


def run(folder: Path, prefixes: list[str], epochs: int | None = None) -> int:
    """Execute the notebooks ``prefixes`` of ``folder`` in order; 0 when every one finished."""
    for prefix in prefixes:
        matches = sorted(folder.glob(f"{prefix}_*.ipynb"))
        if len(matches) != 1:
            print(f"no single notebook starts with {prefix!r} in {folder}")
            return 1
        path = matches[0]
        notebook = nbformat.read(path, as_version=4)
        if epochs is not None:  # the quick check: shorter training, nothing else changes
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.source = cell.source.replace("EPOCHS = 30", f"EPOCHS = {epochs}")
        print(f"\n===== {path.name}", flush=True)
        started = time.time()
        client = EchoingClient(
            notebook,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(folder)}},
        )
        try:
            client.execute()
        except CellExecutionError as error:
            save(notebook, path)  # keep the outputs so far, to see what failed
            print(f"\n{path.name} FAILED:\n{error}")
            return 1
        save(notebook, path)
        print(f"===== {path.name} finished in {(time.time() - started) / 60:.1f} min", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("notebooks", nargs="*", help="notebook numbers, such as 01 03")
    parser.add_argument(
        "--smoke",
        metavar="DIR",
        type=Path,
        help="run every notebook in a copy in DIR, with one epoch per training run",
    )
    args = parser.parse_args()

    if args.smoke:
        destination = args.smoke.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for path in TARGET.glob("*.ipynb"):
            shutil.copy2(path, destination / path.name)
        prefixes = args.notebooks or [p.name[:2] for p in sorted(destination.glob("*.ipynb"))]
        return run(destination, prefixes, epochs=1)
    if not args.notebooks:
        parser.error("name the notebooks to run, such as 01 02 03 04 05")
    return run(TARGET, args.notebooks)


if __name__ == "__main__":
    sys.exit(main())
