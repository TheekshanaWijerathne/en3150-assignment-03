# `config/` — layered YAML loading

**Owner: Member 1.** Used by every script and every member.

## Inputs / outputs

**Input:** a path to `configs/experiments/<run>.yaml`
**Output:** a `ResolvedConfig` — the merged mapping plus the handful of fields that appear in
signatures everywhere (`run_id`, `model_name`, `optimizer_name`, `seed`, `device`).

```python
from edgecnn.config import load_config

cfg = load_config("configs/experiments/model_b__adam.yaml")
cfg.run_id                              # "model_b__adam__seed42"
cfg.section("training")["epochs"]       # 30
```

Those fields are hoisted out of the raw mapping so nobody writes `cfg["experiment"]["seed"]` and
typoes the key path — a missing nested key returns `None` rather than raising, which is exactly the
kind of failure that surfaces ten functions later.

## The merge

```
base.yaml  <-  stages/*.yaml (via `extends:`)  <-  experiments/<run>.yaml  <-  CLI overrides
```

Mappings merge recursively. **Lists are replaced wholesale** — an experiment setting
`augmentation: [random_flip]` means exactly that, not "the base list plus this".

## Two guarantees worth knowing

**Paths are repo-relative.** `resolve_path` anchors them to `paths.REPO_ROOT`, which is derived
from `__file__`. So the same YAML works from `scripts/`, `tests/` and `notebooks/` regardless of
the working directory.

**`run_id` cannot disagree with itself.** If a config states `run_id:` explicitly and it does not
match what `model`, `optimizer` and `seed` imply, loading raises `ConfigError`. Without that check,
artifacts would be written to one directory and read from another — and a glob that matches nothing
raises nothing, so it would fail silently, at report time.
