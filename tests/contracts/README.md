# `tests/contracts/` — the cross-member seam tests

**Run after every `git pull`:**

```powershell
pytest tests/contracts -q
```

| File | Checks | Needs torch? |
|---|---|---|
| `test_schemas.py` | schemas are valid, accept good artifacts, reject bad ones | no |
| `test_paths_and_configs.py` | `run_id` round-trips; all six configs compose; stages declare their I/O | no |
| `test_seams.py` | Seam 1 batch shapes; Seam 2 logits contract, for all four models | yes |

The first two need neither torch nor a dataset, so they run in about a second. That is the set to
run constantly.

## Green by design

Tests whose subject is still a stub **skip** rather than fail, via the `pending()` helper in
[`../conftest.py`](../conftest.py), and turn into real assertions the moment that stub is
implemented. So red genuinely means something broke — which is only useful if the suite is normally
green. See [../README.md](../README.md).

## What belongs here

Tests of the **contract**, not of an implementation. "Model B returns logits of the right shape"
belongs here; "Model B's third conv layer has 64 filters" does not — Member 2 must stay free to
retune the architecture without breaking someone else's test.
