# `tests/` — the interface alarm

**Run after every `git pull`:**

```powershell
pytest tests/contracts -q
```

## The suite is green on day one, on purpose

Most of the repository is unimplemented stubs, yet these tests pass. That is deliberate.

If the suite were red while stubs were unfilled, nobody would look at it — and the one signal that
tells you *"someone else's merge broke the interface you depend on"* would be lost in noise.

So a test whose subject still raises `NotImplementedError` reports as **skipped**, via the
`pending()` helper in `conftest.py`:

```python
with pending("Member 2"):
    model = build_model("model_b", num_classes=10)
assert count_params(model) <= 100_000     # runs for real once the stub is filled
```

The assertion activates automatically the moment that stub is implemented. **Nobody has to
remember to enable a test.**

## Layout

| Path | What it checks | Needs torch? |
|---|---|---|
| `contracts/test_schemas.py` | Schemas are valid, accept good artifacts, reject bad ones | no |
| `contracts/test_paths_and_configs.py` | `run_id` round-trips; all six configs compose; stages declare their I/O | no |
| `contracts/test_seams.py` | Seam 1 batch shapes; Seam 2 logits contract, for all four models | yes |
| `test_param_budget.py` | Model B stays under 100,000 trainable parameters | yes |
| `fixtures/` | The synthetic dataset — see `fixtures/README.md` |  |

The schema and config tests need neither torch nor a dataset, so they run in about a second on any
machine. That is the set to run constantly.

## What these tests are actually for

They are not about code coverage. Each one guards a way the four-member split can fail **silently**:

- A malformed `run_id` → Member 3 writes where nobody reads. A glob matching nothing raises nothing.
- A softmax left inside a model → trains on a double-softmaxed signal, does not crash, quietly
  underperforms.
- A percentage written where a fraction was expected → a plausible-looking wrong number in the table.
- `fitted_on: "all"` → preprocessing leaks held-out data into training.
- Fewer than 20 epochs → violates the assignment, discovered at report time.
- A depthwise conv with `groups=1` → Model B silently blows its 100k budget.
- Overlapping splits → every reported number is invalid.

## If a test fails after you pull

Someone changed an interface. **Tell them** — do not quietly adapt your code to a mutated schema.
See [CONTRIBUTING.md](../CONTRIBUTING.md#if-pytest-testscontracts-fails-after-a-pull).

## Adding tests

Test the *contract*, not the implementation. "Model B returns logits of the right shape" belongs
here; "Model B's third conv layer has 64 filters" does not — Member 2 must stay free to retune the
architecture without breaking someone else's test.
