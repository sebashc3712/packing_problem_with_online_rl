# Heuristic-Based Agent to Solve The Online Three-Dimensional Container Loading Problem

**Authors:** Juan Sebastián Herrera-Cobo & David Álvarez-Martínez

**Venue:** Submitted to the "Learning to Optimize: RL as an Optimizer" special session.

**Contribution:** A Dueling DQN agent that selects among five domain-specific packing heuristics
(plus a learnable buffer deferral action) to solve the Online 3D Bin Packing Problem, outperforming
the best single-heuristic baseline across all tested buffer sizes.

---

## Problem

The **Online 3D Bin Packing Problem (Online 3D-BPP)** packs a stream of boxes — arriving one at a
time without lookahead — into a container. This implementation uses:

- **Pallet:** 10×10×10 discrete grid.
- **Stability constraint:** Each placed box must have at least 60% of its base area supported by
  the pallet or a previously placed box.
- **Buffer deferral:** A size-K buffer allows the agent to defer up to K boxes from the stream for
  placement after all other boxes have arrived.
- **Utilization metric:** `U = packed_volume / container_volume` (reported as a percentage).
- **Benchmarks:** Three fixed-orientation instance sets — `cut1`, `cut2`, `rs`.

---

## Method

The agent uses a **Dueling DQN** with the following components:

1. **Heuristic action space** — instead of raw 3D coordinates, the agent selects one of five
   deterministic packing heuristics:
   - Stacking
   - Best-Fit
   - Semi-Perfect-Fit
   - Corner-Based
   - Complex-Fit

2. **Buffer-as-action (V4 / exp12)** — a sixth action defers the current box to the buffer,
   expanding the action space from 5 to 6 actions when K > 0.

3. **Feasibility-masking auxiliary head** — a secondary network head trained with flatness-bias
   supervision predicts valid placement locations and is used by `proxy_scores_for_heuristics()`
   to score each heuristic candidate.

4. **State representation** — a 10×10 height map of the pallet, augmented with the current box
   dimensions and buffer occupancy features.

The paper source is in [`main.tex`](main.tex). Reference PDFs are in [`papers/`](papers/).

---

## Version Timeline

Each version is a self-contained stack (`oskp_rl_up_buffer_experiments_v{N}.py` +
`vec_env_v{N}.py` + `parallel_train_v{N}.py`). All stacks share the single driver
`experiments_v2.py`. See [docs/VERSIONS.md](docs/VERSIONS.md) for the full narrative.

| Version | Module file | Exp # | What changed | Paper relationship |
|---------|-------------|-------|--------------|-------------------|
| base/v0 | `oskp_rl_up_buffer_experiments.py` | 1–4 | Original 5-heuristic agent, external buffer | Pre-paper exploration |
| v2 | `oskp_rl_up_buffer_experiments_v2.py` | 5–10 | Parallel training pipeline, refactored buffer | Dev / ablation baseline |
| **v3** | `oskp_rl_up_buffer_experiments_v3.py` | 11 | Buffer box dimensions as state input ("buffer vision") | **Originally submitted paper** |
| **v4** | `oskp_rl_up_buffer_experiments_v4.py` | 12 | Buffer deferral as an explicit action (6 actions) | **Current `main.tex` methodology** |
| v5 | `oskp_rl_up_buffer_experiments_v5.py` | 13 | Multiprocessing evaluation improvements | Later exploration |
| v6 | `oskp_rl_up_buffer_experiments_v6.py` | 14 | Corner-only ablation (1 action, no buffer) | Ablation |

---

## Repository Layout

```
packing_problem_with_online_rl/
├── experiments_v2.py                        # Shared experiment driver (entry point)
├── ga_instance_mixer.py                     # GA tool to generate synthetic training data
├── main.tex                                 # Paper source
├── requirements.txt
├── LICENSE                                  # MIT
│
├── oskp_rl_up_buffer_experiments.py         # base/v0 module (kept at root — imported by driver)
├── oskp_rl_up_buffer_experiments_v2.py      # v2 module
├── vec_env_v2.py  /  parallel_train_v2.py
├── oskp_rl_up_buffer_experiments_v3.py      # v3 module
├── vec_env_v3.py  /  parallel_train_v3.py
├── oskp_rl_up_buffer_experiments_v4.py      # v4 module (primary)
├── vec_env_v4.py  /  parallel_train_v4.py
├── oskp_rl_up_buffer_experiments_v5.py      # v5 module
├── vec_env_v5.py  /  parallel_train_v5.py
├── oskp_rl_up_buffer_experiments_v6.py      # v6 module
├── vec_env_v6.py  /  parallel_train_v6.py
│
├── approachesO3DKP/                         # Datasets (see Datasets section)
│   ├── cut_1.pt  /  cut_2.pt  /  rs.pt     # Benchmark validation instances
│   ├── ga_mixed.pt                          # GA-generated training instances
│   └── ga_mixed_large.pt
│
├── docs/
│   ├── VERSIONS.md                          # Authoritative version timeline
│   ├── RESULTS.md                           # Results directory schema & column docs
│   ├── experiments_documentation.md         # Deep-dive on the experiment workflow
│   └── ga_mixer_documentation.md            # Deep-dive on the GA instance mixer
│
├── legacy/                                  # Archived non-active files
│   ├── README.md                            # Explains what is here
│   ├── experiments.py                       # Old driver (superseded)
│   ├── vec_env.py  /  parallel_train.py     # Old support files
│   └── *_sequential.py                      # Non-parallel debug variants
│
├── papers/                                  # Reference PDFs
├── results Huertas/                         # Baseline comparison data
└── experiments_results_refactored/          # Training outputs (gitignored model files)
```

> **Note on active file locations:** All active Python stacks must stay at the repo root because
> versioned modules import each other by bare module name and `vec_env_v*` workers use
> `multiprocessing` spawn, which re-imports by name. Do not move them into subfolders.

---

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `numpy>=1.24.0`, `torch>=2.0.0`, `matplotlib>=3.7.0`, `pandas>=2.0.0`.

---

## Quick Start / Reproduce

All experiments run through the shared driver. General syntax:

```bash
python experiments_v2.py --exp <N> --max-epochs <M> --num-envs <E> [--buffer-size <K>]
```

### Reproduce the primary results (V4 / exp12, buffer-as-action)

```bash
# K=1 buffer
python experiments_v2.py --exp 12 --max-epochs 30 --num-envs 8 --buffer-size 1

# K=2 buffer
python experiments_v2.py --exp 12 --max-epochs 30 --num-envs 8 --buffer-size 2

# K=3 buffer (best reported result)
python experiments_v2.py --exp 12 --max-epochs 30 --num-envs 8 --buffer-size 3
```

### Other versions

```bash
# V2 epoch training (5 heuristics, external buffer)
python experiments_v2.py --exp 10 --max-epochs 30 --num-envs 8

# V3 buffer-vision (originally submitted paper)
python experiments_v2.py --exp 11 --max-epochs 30 --num-envs 8 --buffer-size 2

# V6 corner-only ablation
python experiments_v2.py --exp 14 --max-epochs 10 --num-envs 8
```

### Quick debug run (single epoch, 2 envs)

```bash
python experiments_v2.py --exp 12 --max-epochs 1 --num-envs 2 --buffer-size 3
```

### Full experiment number table

| Exp # | Version | Function | Description |
|-------|---------|----------|-------------|
| 5 | v2 | `run_buffer_comparison_v2` | Buffer size comparison |
| 6 | v2 | `run_experiment_6` | Buffer with LR 0.005 |
| 7 | v2 | `run_experiment_7` | LR test, no buffer |
| 8 | v2 | `run_final_best_config_experiment` | Final best config |
| 9 | v2 | `grid_search` | Grid search |
| 10 | v2 | `run_epoch_training` | Epoch-based parallel, configurable buffer |
| 11 | v3 | `run_epoch_training_v3` | Buffer box dims as input |
| **12** | **v4** | `run_epoch_training_v4` | **Buffer-as-action (6 actions) — primary** |
| 13 | v5 | `run_epoch_training_v5` | Improved buffer-as-action |
| 14 | v6 | `run_epoch_training_corner_only` | Corner-only heuristic, no buffer |

---

## Datasets

All datasets live in `approachesO3DKP/` as pickled Python objects (`.pt` files, loaded with
`torch.load` / `pickle.load`). Each dataset is a list of *episodes*, where each episode is a
list of `[l, w, h]` box dimension tuples.

| File | Role | Description |
|------|------|-------------|
| `cut_1.pt` | Validation | cut1 benchmark instances (fixed single orientation) |
| `cut_2.pt` | Validation | cut2 benchmark instances (fixed single orientation) |
| `rs.pt` | Validation | rs benchmark instances (fixed single orientation) |
| `ga_mixed.pt` | Training | GA-generated synthetic mix (small) |
| `ga_mixed_large.pt` | Training | GA-generated synthetic mix (large) |

To generate new synthetic training data:

```bash
python ga_instance_mixer.py
```

See [docs/ga_mixer_documentation.md](docs/ga_mixer_documentation.md) for details on the
genetic algorithm and target fill-rate objective.

---

## Results

See [docs/RESULTS.md](docs/RESULTS.md) for the full directory schema, CSV column definitions,
and guidance on reading `training_summary_plots.png`.

### Key utilization numbers (V4 / exp12, averaged across cut1/cut2/rs)

| Buffer K | cut1 | cut2 | rs   | avg  |
|----------|------|------|------|------|
| 0 (V2, 5-action) | 62.3 | 62.3 | 52.4 | 59.0 |
| 1 | 66.4 | 66.5 | 58.2 | 63.7 |
| 2 | 71.4 | 70.2 | 62.1 | 67.9 |
| 3 | 74.4 | 72.1 | 63.8 | 70.1 |

All values are utilization percentages (higher is better; max theoretical = 100%).

---

## Baseline Comparison

[`results Huertas/`](results%20Huertas/) contains the single-heuristic baseline data from the
Huertas et al. comparison (R=0, single orientation). The best single heuristic per buffer size:

| K | Heuristic | cut1  | cut2  | rs    | avg   |
|---|-----------|-------|-------|-------|-------|
| 0 | RF        | 57.25 | 57.73 | 45.05 | 53.34 |
| 1 | RF        | 63.47 | 63.56 | 55.02 | 60.7  |
| 2 | RF        | 67.98 | 67.96 | 62.00 | 66.0  |
| 3 | CF        | 70.49 | 70.26 | 66.09 | 69.0  |

---

## Legacy

Earlier, non-parallel, and superseded files are archived in [`legacy/`](legacy/). See
[`legacy/README.md`](legacy/README.md) for details.

---

## Citation

If you use this code, please cite the paper (BibTeX entry will be added upon publication).

---

## License

MIT — see [`LICENSE`](LICENSE).
