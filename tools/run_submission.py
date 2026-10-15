"""Run the submission notebooks in place, in order, keeping their outputs.

    python tools/run_submission.py 01 02 03 04 05     the training notebooks: on a GPU (Colab)
    python tools/run_submission.py 06 07              the benchmark and the tables: on the laptop
    python tools/run_submission.py --smoke DIR        a quick check of every notebook, in a copy
                                                      in DIR with one epoch per training run
    python tools/run_submission.py --clean            only tidy the outputs of executed notebooks

Each notebook runs in a fresh kernel with the submission folder as its working directory, so they
share data/, checkpoints/ and results/ exactly as a reader running them in order would. What the
notebooks print, such as each training epoch, is echoed here as it happens.

On Colab, pandas tables come wrapped in Colab's "interactive table" button and its JavaScript.
After each run the wrapper is removed, leaving pandas' standard table, as any Jupyter shows it.
"""

from __future__ import annotations

import argparse
import re
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


#: pandas' own table HTML, inside Colab's interactive-table wrapper.
PANDAS_TABLE = re.compile(
    r"<div>\s*<style scoped>.*?</table>(?:\s*<p>[^<]*</p>)?\s*</div>", re.DOTALL
)


def unwrap_colab_tables(notebook: nbformat.NotebookNode) -> int:
    """Replace Colab's interactive-table wrapper by the pandas table it wraps; returns the count."""
    count = 0
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if html is None:
                continue
            html = "".join(html) if isinstance(html, list) else html
            if "colab-df-container" not in html:
                continue
            table = PANDAS_TABLE.search(html)
            if table is None:
                raise ValueError("a Colab table without a pandas table inside it")
            output["data"]["text/html"] = table.group(0)
            count += 1
    return count


def save(notebook: nbformat.NotebookNode, path: Path) -> None:
    unwrap_colab_tables(notebook)
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
    parser.add_argument(
        "--clean", action="store_true", help="only tidy the outputs of executed notebooks"
    )
    args = parser.parse_args()

    if args.clean:
        for path in sorted(TARGET.glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            count = unwrap_colab_tables(notebook)
            if count:
                save(notebook, path)
                print(f"{path.name}: unwrapped {count} Colab table(s)")
        return 0
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
