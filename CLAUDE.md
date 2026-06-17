# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Research project implementing a Dueling DQN agent that selects among 5 packing heuristics (Stacking, Best-Fit, Semi-Perfect-Fit, Corner-Based, Complex Fit) to solve the Online 3D Bin Packing Problem with a deferral buffer. The paper is in `main.tex`.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run epoch-based parallel training (main experiment)
python experiments_v2.py --exp 10 --max-epochs 30 --num-envs 8

# Quick debug run
python experiments_v2.py --exp 10 --max-epochs 2 --num-envs 2

# Run buffer-as-action experiment (v4, exp 12)
python experiments_v2.py --exp 12 --max-epochs 30 --num-envs 8 --buffer-size 3

# Run corner-only experiment (v6, exp 14, no buffer)
python experiments_v2.py --exp 14 --max-epochs 10 --num-envs 8

# Generate GA-mixed training instances
python ga_instance_mixer.py
```

## Architecture

The codebase has multiple versioned files (base/v0, v2–v6). Each version is a self-contained stack of 3 files + the shared experiment driver. See `docs/VERSIONS.md` for the full version narrative and paper relationships.

### Versioned Stacks

| Version | Module | VecEnv | ParallelTrain | Experiment | Description |
|---------|--------|--------|---------------|------------|-------------|
| base/v0 | `oskp_rl_up_buffer_experiments.py` | (archived) | (archived) | exp 1–4 | Original 5-heuristic agent; module kept at root — imported by driver at line 993 |
| v2 | `oskp_rl_up_buffer_experiments_v2.py` | `vec_env_v2.py` | `parallel_train_v2.py` | exp 10 | 5 heuristics, configurable buffer (external) |
| v3 | `oskp_rl_up_buffer_experiments_v3.py` | `vec_env_v3.py` | `parallel_train_v3.py` | exp 11 | Buffer box dimensions as input (originally submitted paper version) |
| **v4** | `oskp_rl_up_buffer_experiments_v4.py` | `vec_env_v4.py` | `parallel_train_v4.py` | **exp 12** | **Primary.** 6 actions (5 heuristics + buffer defer as action); methodology in current `main.tex` |
| v5 | `oskp_rl_up_buffer_experiments_v5.py` | `vec_env_v5.py` | `parallel_train_v5.py` | exp 13 | Improved buffer-as-action with multiprocessing eval |
| v6 | `oskp_rl_up_buffer_experiments_v6.py` | `vec_env_v6.py` | `parallel_train_v6.py` | exp 14 | Corner-only heuristic (1 action, no buffer) |

### Shared Files
- **`experiments_v2.py`** -- Experiment driver. Entry point for all runs (exp 5-14). Contains evaluate/train/visualize functions for each version.
- **`ga_instance_mixer.py`** -- Genetic algorithm to evolve synthetic training datasets from source instances.
- **`approachesO3DKP/`** -- Dataset files: `cut_1.pt`, `cut_2.pt`, `rs.pt` (benchmark validation), `ga_mixed.pt` / `ga_mixed_large.pt` (GA-generated training data). These are pickled lists of episodes (each episode = list of [l,w,h] boxes).

### Documentation (`docs/`)
- **`docs/VERSIONS.md`** -- Authoritative version timeline: what changed at each step, exp-number mapping, paper relationships (V3 = originally submitted, V4/exp12 = current `main.tex`, V5/V6 = later exploration).
- **`docs/RESULTS.md`** -- `experiments_results_refactored/` naming convention, CSV column schema for `epoch_training_results.csv`, and key results table.
- **`docs/experiments_documentation.md`** -- Deep-dive on the experiment workflow (moved from repo root).
- **`docs/ga_mixer_documentation.md`** -- Deep-dive on the GA instance mixer (moved from repo root).

### Legacy (`legacy/`)
- Superseded non-parallel files: old driver `experiments.py`, base-version `vec_env.py` / `parallel_train.py`, and `*_sequential.py` debug variants. Not imported by any active code. See `legacy/README.md`.

### Experiment Numbers

| Exp | Function | Version | Description |
|-----|----------|---------|-------------|
| 5 | `run_buffer_comparison_v2` | v2 | Buffer size comparison |
| 6 | `run_experiment_6` | v2 | Buffer with LR 0.005 |
| 7 | `run_experiment_7` | v2 | LR test, no buffer |
| 8 | `run_final_best_config_experiment` | v2 | Final best config |
| 9 | `grid_search` | v2 | Grid search |
| 10 | `run_epoch_training` | v2 | Epoch-based parallel, configurable buffer |
| 11 | `run_epoch_training_v3` | v3 | Buffer box dims as input |
| 12 | `run_epoch_training_v4` | v4 | Buffer-as-action (6 actions) |
| 13 | `run_epoch_training_v5` | v5 | Improved buffer-as-action |
| 14 | `run_epoch_training_corner_only` | v6 | Corner-only heuristic, no buffer |

### Key Design Patterns
- **Heuristic action space**: Agent picks a heuristic index (0-4), not raw coordinates. Each heuristic deterministically places the box.
- **Buffer mechanism**: Agent can defer a box to a k-size buffer. After the stream ends, buffered boxes are attempted in order. (v4+ has buffer as an explicit action; v6 has no buffer.)
- **Feasibility mask**: Auxiliary neural network head predicts valid placements with flatness-bias supervision. Used by `proxy_scores_for_heuristics()` to score each heuristic.
- **Pallet**: Default 10x10x10 grid. Height map (10x10 array) is the core state representation.

### Experiment Results
- Outputs go to `experiments_results_refactored/` (gitignored except CSVs/PNGs).
- Model checkpoints are `.pt` files (gitignored).
- Benchmark instances: cut1, cut2, rs (fixed single orientation).

### Best Practices
- for every implementation in training and evaluation we use multi-processing
- every change needs to be test with an experiment to see the errors in real enviorentment
- Besides technical checks in a test, we check that: 1. utilization numbers are feasible (lower than 100%). 2. the agent is not "addicted" to one heuristic, 3. the loss function is changing with the epochs
