# `tests/` — the interface alarm

**Run after every `git pull`:**

```powershell
pytest tests/contracts        # the interfaces between members - about a second
pytest                        # everything
```

`pyproject.toml` already adds `-q`. Adding another `-q` hides pytest's pass/fail summary line.

## Green on day one, on purpose

Most of the library is still stubs, yet these tests pass. That is deliberate.

If the suite failed until every stub was filled in, nobody would look at it, and the one signal that
says *"someone's merge broke the interface you depend on"* would be lost in the noise. So a test
whose subject still raises `NotImplementedError` is reported as **skipped**, through the `pending()`
helper in `conftest.py`:

```python
with pending("Member 2"):
    model = build_model("model_b", num_classes=10)
assert count_params(model) <= 100_000     # runs for real once the stub is implemented
```

The assertion switches on by itself the moment the stub is implemented. **Nobody has to remember to
enable a test.**

## Layout

| Path | What it checks | Needs torch? |
|---|---|---|
| `contracts/test_schemas.py` | schemas are valid, accept good artifacts, reject bad ones | no |
| `contracts/test_paths_and_configs.py` | `run_id` round-trips; paths agree; debug checkpoints are kept apart | no |
| `contracts/test_config_layout.py` | one home per setting; stage I/O blocks; each run's optimizer; `MODE` and `is_official` | no |
| `contracts/test_seams.py` | Seam 1 batch shapes; Seam 2 logits; seeded builds; every experiment builds from its config | yes |
| `test_param_budget.py` | Model B stays under 100,000 trainable parameters | yes |
| `test_notebooks.py` | the eight section notebooks exist and follow the standard layout; no tokens, no local paths | no |
| `test_colab.py` | `push_results` issues the right git commands and never leaks the token | no |
| `test_data.py` | the split is exact and reproducible; manifests are byte-identical across platforms; the pixel cache rebuilds when an image changes; augmentation on train only; loader order, subsets and normalisation; EuroSAT preparation on a small fake download — only official runs write, a committed split is never redrawn, edits are detected, missing images are refetched | yes |
| `test_benchmark.py` | size equals the saved weights and matches torchvision's published sizes; latency runs on one thread and restores the setting; peak memory follows each result to its last use, exactly, on hand-worked examples; runs sharing an architecture share one measurement; official profiles need the training history | yes |
| `test_evaluation.py` | metrics match hand-computed cases; the confusion matrix stays square; `test_metrics.json` validates; MAC counts reproduce torchvision's published figures; `evaluate_run` end to end with the real Model B and trainer, refusing another run's checkpoint; loss-curve figures display once | yes |
| `test_utils.py` | seeding repeats; device errors are loud; logging does not stack; Markdown tables; plot style | yes |
| `fixtures/` | the synthetic dataset — see `fixtures/README.md` | |

## What these tests are for

They are not about coverage. Each one guards against a way the four-member split can fail
**silently**:

- a setting defined twice → an SGD run that silently trains with Adam;
- a malformed `run_id` → results written where nobody reads them;
- a softmax left inside a model → it trains, but worse, and nothing crashes;
- a debug run counted as official → a 2-epoch result in the report;
- different initial weights across the §3 runs → the comparison measures noise;
- a token printed in a notebook → published, because outputs are committed and the repo is public;
- a depthwise conv with `groups=1` → Model B silently blows its 100k budget;
- overlapping splits → every reported number is invalid.

**Notebooks are checked for structure only.** Running them needs data and trained models, so that is
done by the team in the Stage 4 dry run ([CONTRIBUTING.md §7](../CONTRIBUTING.md#7-official-runs)).

## If a test fails after you pull

Someone changed an interface. **Tell them** — do not quietly adapt your code to a changed schema. See
[CONTRIBUTING.md §6](../CONTRIBUTING.md#pulling-in-someone-elses-work).

## Adding tests

Test the *contract*, not the implementation. "Model B returns logits of the right shape" belongs here.
"Model B's third conv layer has 64 filters" does not: Member 2 must stay free to retune the
architecture without breaking someone else's test.
