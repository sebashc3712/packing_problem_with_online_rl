# Legacy / Archived Files

These files are archival snapshots of earlier development attempts. They are **not imported by any
active code** and are preserved here only for research history and reproducibility of older
experiments.

## Contents

| File | What it was |
|------|-------------|
| `experiments.py` | Original experiment driver, superseded by `experiments_v2.py`. Ran experiments 1–4 against the base module. |
| `vec_env.py` | Vectorized environment used by the original `experiments.py` driver (non-versioned). |
| `parallel_train.py` | Parallel training loop used by the original `experiments.py` driver (non-versioned). |
| `experiments_v1_sequential.py` | Sequential (non-parallel) debug variant of the experiment driver for V1. |
| `oskp_rl_up_buffer_experiments_v1_sequential.py` | V1 module variant used by the sequential driver above. |
| `experiments_v2_sequential.py` | Sequential debug variant of the V2 experiment driver. |
| `oskp_rl_up_buffer_experiments_v2_sequential.py` | V2 module variant used by the sequential V2 driver. |

## Why "sequential" variants exist

The `*_sequential.py` files were non-parallel debug versions created during development of the
parallel training pipeline. They exist solely to make it easier to trace execution without
multiprocessing, and were superseded once the parallel versions were validated.

## Running `experiments.py` (historical reference)

`experiments.py` imports `oskp_rl_up_buffer_experiments` by bare module name. That module lives
at the **repo root** and was deliberately kept there (it is still imported by `experiments_v2.py`
at line 993). To run `experiments.py` from this `legacy/` subdirectory you would need to set
`PYTHONPATH` explicitly:

```bash
# From repo root
PYTHONPATH=. python legacy/experiments.py
```

However, these older experiments are unlikely to be useful to re-run; the results they produced
are superseded by `experiments_results_refactored/`.
