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

# Generate GA-mixed training instances
python ga_instance_mixer.py
```

## Architecture

The codebase has multiple versioned files (v1-v4). **V4 is the primary/current version.**

### Core Files (V4 stack)
- **`oskp_rl_up_buffer_experiments_v4.py`** — All-in-one module: `BoxPilingEnv` (environment), `DQNAgent` (Dueling DQN with mask prediction head), `train()` loop, 5 heuristic placement functions, `proxy_scores_for_heuristics()`, and ground-truth feasibility mask computation (`compute_gt_mask`).
- **`experiments_v2.py`** — Experiment driver. Imports from the V4 module. Contains `evaluate()` for greedy assessment and experiment configs (exp 5/7/10). Entry point for all runs.
- **`parallel_train_v4.py`** — `parallel_train_one_epoch()`: orchestrates multi-env training with stream-then-buffer logic, scaled batch size (32 × n_envs).
- **`vec_env_v4.py`** — `SubprocVecEnv`: multiprocessing-based vectorized environment wrapper with custom commands (new_box_arrival, get_proxy_scores, choose_action_by_heuristic).

### Supporting Files
- **`ga_instance_mixer.py`** — Genetic algorithm to evolve synthetic training datasets from source instances.
- **`approachesO3DKP/`** — Dataset files: `cut_1.pt`, `cut_2.pt`, `rs.pt` (benchmark validation), `ga_mixed.pt` / `ga_mixed_large.pt` (GA-generated training data). These are pickled lists of episodes (each episode = list of [l,w,h] boxes).

### Key Design Patterns
- **Heuristic action space**: Agent picks a heuristic index (0-4), not raw coordinates. Each heuristic deterministically places the box.
- **Buffer mechanism**: Agent can defer a box to a k-size buffer. After the stream ends, buffered boxes are attempted in order.
- **Feasibility mask**: Auxiliary neural network head predicts valid placements with flatness-bias supervision. Used by `proxy_scores_for_heuristics()` to score each heuristic.
- **Pallet**: Default 10×10×10 grid. Height map (10×10 array) is the core state representation.

### Experiment Results
- Outputs go to `experiments_results_refactored/` (gitignored except CSVs/PNGs).
- Model checkpoints are `.pt` files (gitignored).
- Benchmark instances: cut1, cut2, rs (fixed single orientation).

### Best Practices
- for every implementation in training and evaluation we use multi-processing
- every change needs to be test with an experiment to see the errors in real enviorentment
- Besides technical checks in a test, we check that: 1. utilization numbers are feasible (lower than 100%). 2. the agent is not "addicted" to one heuristic, 3. the loss function is changing with the epochs
