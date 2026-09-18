# `utils/` — shared helpers

**Owner: Member 1.** Used by all four members. Dependency-light and side-effect free on import.

| File | Purpose |
|---|---|
| `seed.py` | `seed_everything`, `worker_init_fn` |
| `device.py` | `resolve_device`, `describe_device` |
| `logging.py` | `setup_logging`, `get_logger` |
| `io.py` | `sha256_file`, `ensure_dir`, `write_markdown_table` |

## Inputs / outputs

**Inputs:** plain arguments — no config objects, no dataset, no model.
**Outputs:** helpers imported by every other package.

## Why seeding is load-bearing here

The assignment asks for random seeds to be recorded. The four-way split makes it matter for a
second reason: if Member 3's Adam run and SGD run do not start from identical initial weights, the
Section 3 comparison measures initialisation noise alongside the optimizer, and the conclusion
about momentum is not supported by the evidence.

So `seed_everything` is called at the top of every script, before the model is constructed, and
again before each optimizer variant in the study.

`worker_init_fn` matters too: without it, DataLoader workers inherit non-deterministic seeds and
augmentation differs run to run even when `seed_everything` was called.

## Why `describe_device` exists

Section 4 requires the evaluation hardware to be reported, and an epoch-time or latency column
without a device string is meaningless. This produces the string that goes into `resources.json`
and into the report table, for example `cpu (11th Gen Intel Core i7-11800H)`.

## For JSON that crosses a member boundary

Use `edgecnn.contracts.schema.read_json` / `write_json` instead of anything here — those validate
against the contract. `io.py` is for everything else.
