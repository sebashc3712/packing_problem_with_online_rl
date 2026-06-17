# Version Timeline

This is the authoritative reference for which version of the codebase corresponds to which
experiment, paper submission, and conceptual development stage.

---

## Overview

The codebase grew incrementally. Each version is a **self-contained stack** of three files:

```
oskp_rl_up_buffer_experiments_v{N}.py  — DQN agent, environment, heuristics
vec_env_v{N}.py                        — vectorized (subprocess) environment wrapper
parallel_train_v{N}.py                 — parallel training loop
```

All stacks share a single experiment driver: **`experiments_v2.py`** (entry point via
`--exp N` flag).

---

## Version-by-Version Summary

### base / v0
**Module:** `oskp_rl_up_buffer_experiments.py` · `vec_env.py`* · `parallel_train.py`*
**Exp #:** 1–4 (via legacy `experiments.py`)
**Paper relationship:** Pre-paper exploration. Not referenced in `main.tex`.

The original 5-heuristic agent. The module file (`oskp_rl_up_buffer_experiments.py`) is still
imported by `experiments_v2.py` (line 993) for backward-compatible evaluation functions, so it
stays at the repo root even though `vec_env.py` and `parallel_train.py` have been archived.

\* `vec_env.py` and `parallel_train.py` are archived in `legacy/` — they were only used by the
now-archived `legacy/experiments.py` driver.

---

### V2
**Module:** `oskp_rl_up_buffer_experiments_v2.py` · `vec_env_v2.py` · `parallel_train_v2.py`
**Exp #:** 5–10 (via `experiments_v2.py`)
**Paper relationship:** Development / ablation baseline.

Key changes from base:
- Refactored to support the new parallel training pipeline (`parallel_train_v2.py`).
- Buffer is **external** — the agent always places the current box, then optionally buffers it
  before stream end; buffer size is a hyperparameter passed from outside.
- Exp 10 is the primary V2 entry point: `--exp 10 --max-epochs 30 --num-envs 8`.

Results in: `experiments_results_refactored/exp10_*`

---

### V3
**Module:** `oskp_rl_up_buffer_experiments_v3.py` · `vec_env_v3.py` · `parallel_train_v3.py`
**Exp #:** 11
**Paper relationship:** ORIGINALLY SUBMITTED PAPER VERSION.

Key changes from V2:
- Buffer box dimensions are added as input features to the state representation ("buffer
  vision"), giving the agent explicit awareness of what is deferred.
- Still uses the external-buffer mechanism (buffer is not an action).

Results in: `experiments_results_refactored/exp11_*`

---

### V4 — current `main.tex` methodology
**Module:** `oskp_rl_up_buffer_experiments_v4.py` · `vec_env_v4.py` · `parallel_train_v4.py`
**Exp #:** 12
**Paper relationship:** METHODOLOGY IN THE CURRENT `main.tex`. This is the primary version.

Key changes from V3:
- **Buffer deferral becomes an explicit action (action index 5)**, expanding the action space
  from 5 to 6. The agent directly chooses whether to defer to the buffer rather than having the
  buffer managed externally.
- This design aligns deferral incentives with the Q-learning objective.

Results in: `experiments_results_refactored/exp12_buffer_as_action_size_{1,2,3,4}/`

Key utilization numbers (avg validation on cut1/cut2/rs benchmarks):

| Buffer K | cut1  | cut2  | rs    | avg   |
|----------|-------|-------|-------|-------|
| 0 (5-action) | 62.3 | 62.3 | 52.4 | 59.0 |
| 1        | 66.4  | 66.5  | 58.2  | 63.7  |
| 2        | 71.4  | 70.2  | 62.1  | 67.9  |
| 3        | 74.4  | 72.1  | 63.8  | 70.1  |

---

### V5
**Module:** `oskp_rl_up_buffer_experiments_v5.py` · `vec_env_v5.py` · `parallel_train_v5.py`
**Exp #:** 13
**Paper relationship:** Later exploration, not the primary result.

Key changes from V4:
- Improved multiprocessing in evaluation (separate process pool for validation).
- Further tuning of the buffer-as-action design.

Results in: `experiments_results_refactored/exp13_improved_buffer_action_size_{2,3}/`

---

### V6
**Module:** `oskp_rl_up_buffer_experiments_v6.py` · `vec_env_v6.py` · `parallel_train_v6.py`
**Exp #:** 14
**Paper relationship:** Ablation / later exploration.

Key changes from V5:
- Reduces the heuristic set to **Corner-Based only** (1 action, no buffer).
- Intended as an ablation to isolate the contribution of heuristic diversity.

Results in: `experiments_results_refactored/exp14_corner_only/`

---

## Quick Reference Table

| Version | Module file | Exp # | Action space | Paper relationship |
|---------|-------------|-------|--------------|-------------------|
| base/v0 | `oskp_rl_up_buffer_experiments.py` | 1–4 | 5 heuristics, external buffer | Pre-paper exploration |
| v2 | `oskp_rl_up_buffer_experiments_v2.py` | 5–10 | 5 heuristics, external buffer | Dev / ablation baseline |
| **v3** | `oskp_rl_up_buffer_experiments_v3.py` | 11 | 5 heuristics + buffer-vision input | **Originally submitted paper** |
| **v4** | `oskp_rl_up_buffer_experiments_v4.py` | 12 | 6 actions (5 heuristics + BUFFER) | **Current `main.tex` methodology** |
| v5 | `oskp_rl_up_buffer_experiments_v5.py` | 13 | 6 actions | Later exploration |
| v6 | `oskp_rl_up_buffer_experiments_v6.py` | 14 | 1 action (Corner only) | Later ablation |
