# Experiment Results Reference

All training outputs are stored in `experiments_results_refactored/` (gitignored at the
model/checkpoint level; CSVs and PNGs are tracked).

---

## Directory Naming Convention

```
experiments_results_refactored/
  exp{N}_{description}[_size_{K}]/
```

- `N` — experiment number (matches `--exp N` flag in `experiments_v2.py`)
- `description` — short human label
- `size_{K}` — buffer size K (only for buffer experiments)

---

## Subdirectories

| Directory | Exp | Description |
|-----------|-----|-------------|
| `exp10_epoch_training` | 10 | V2 epoch training (early run) |
| `exp10_epoch_training_early_patience` | 10 | V2 with early stopping |
| `exp10_epoch_training_no_up` | 10 | V2 without utilization-penalty |
| `exp10_epoch_training_parallel` | 10 | V2 parallel training |
| `exp10_epoch_training_parallel_failed` | 10 | V2 parallel (aborted run) |
| `exp10_epoch_training_parallel_v2` | 10 | V2 parallel refinement |
| `exp10_epoch_training_parallel_v2_buffer_size_1` | 10 | V2, external buffer K=1 |
| `exp10_epoch_training_parallel_v2_buffer_size_2` | 10 | V2, external buffer K=2 |
| `exp10_epoch_training_parallel_v2_buffer_size_3` | 10 | V2, external buffer K=3 |
| `exp10_epoch_training_parallel_v2_buffer_size_4` | 10 | V2, external buffer K=4 |
| `exp10_epoch_training_seq_v2` | 10 | V2 sequential (debug) |
| `exp10_epoch_training_suboptimal` | 10 | V2 suboptimal config |
| `exp11_epoch_training_buffer_dims_size_1` | 11 | V3 buffer-vision, K=1 |
| `exp11_epoch_training_buffer_dims_size_2` | 11 | V3 buffer-vision, K=2 |
| `exp12_buffer_as_action_size_1` | 12 | **V4 (primary), K=1** |
| `exp12_buffer_as_action_size_2` | 12 | **V4 (primary), K=2** |
| `exp12_buffer_as_action_size_3` | 12 | **V4 (primary), K=3** |
| `exp12_buffer_as_action_size_4` | 12 | **V4 (primary), K=4** |
| `exp13_improved_buffer_action_size_2` | 13 | V5 improved, K=2 |
| `exp13_improved_buffer_action_size_3` | 13 | V5 improved, K=3 |
| `exp14_corner_only` | 14 | V6 corner-only ablation |

---

## `epoch_training_results.csv` Column Schema

Each row corresponds to one training epoch. Columns (in order):

| Column | Description |
|--------|-------------|
| `epoch` | Epoch index (0-based) |
| `train_util` | Mean training utilization (fraction of pallet volume packed) |
| `train_q_loss` | DQN Q-network training loss |
| `train_mask_loss` | Feasibility-mask auxiliary head training loss |
| `val_cut1_util` | Validation utilization on cut1 benchmark |
| `val_cut2_util` | Validation utilization on cut2 benchmark |
| `val_rs_util` | Validation utilization on rs benchmark |
| `avg_val_util` | Average of the three validation utilizations |
| `invalid_learned` | Count of invalid placements predicted by the learned mask |
| `invalid_attempted` | Count of invalid placements attempted |
| `epsilon` | Current epsilon (exploration rate) |
| `lr` | Current learning rate |
| `buffer_defer_count` | Number of times the BUFFER action was chosen during training |
| `buffer_place_after_count` | Number of buffered boxes successfully placed after stream end |
| `train_h_stacking` | Training heuristic usage: Stacking |
| `train_h_best_fit` | Training heuristic usage: Best-Fit |
| `train_h_semi_perfect_fit` | Training heuristic usage: Semi-Perfect-Fit |
| `train_h_corner` | Training heuristic usage: Corner-Based |
| `train_h_complex_fit` | Training heuristic usage: Complex-Fit |
| `train_h_BUFFER` | Training heuristic usage: BUFFER defer action |
| `val_cut1_h_stacking` | Validation (cut1) heuristic usage: Stacking |
| `val_cut1_h_best_fit` | Validation (cut1) heuristic usage: Best-Fit |
| `val_cut1_h_semi_perfect_fit` | Validation (cut1) heuristic usage: Semi-Perfect-Fit |
| `val_cut1_h_corner` | Validation (cut1) heuristic usage: Corner-Based |
| `val_cut1_h_complex_fit` | Validation (cut1) heuristic usage: Complex-Fit |
| `val_cut1_h_BUFFER` | Validation (cut1) heuristic usage: BUFFER defer action |
| `val_cut2_h_stacking` | Validation (cut2) heuristic usage: Stacking |
| `val_cut2_h_best_fit` | Validation (cut2) heuristic usage: Best-Fit |
| `val_cut2_h_semi_perfect_fit` | Validation (cut2) heuristic usage: Semi-Perfect-Fit |
| `val_cut2_h_corner` | Validation (cut2) heuristic usage: Corner-Based |
| `val_cut2_h_complex_fit` | Validation (cut2) heuristic usage: Complex-Fit |
| `val_cut2_h_BUFFER` | Validation (cut2) heuristic usage: BUFFER defer action |
| `val_rs_h_stacking` | Validation (rs) heuristic usage: Stacking |
| `val_rs_h_best_fit` | Validation (rs) heuristic usage: Best-Fit |
| `val_rs_h_semi_perfect_fit` | Validation (rs) heuristic usage: Semi-Perfect-Fit |
| `val_rs_h_corner` | Validation (rs) heuristic usage: Corner-Based |
| `val_rs_h_complex_fit` | Validation (rs) heuristic usage: Complex-Fit |
| `val_rs_h_BUFFER` | Validation (rs) heuristic usage: BUFFER defer action |

---

## `training_summary_plots.png`

Each experiment directory also contains `training_summary_plots.png`, a multi-panel figure
showing per-epoch curves for: utilization (train vs. val), Q-loss, mask loss, epsilon, and
per-heuristic action usage fractions. This is the primary visual diagnostic for checking whether
the agent is learning and whether it diversifies heuristic usage.

---

## Key Results Summary (V4 / exp12)

The paper reports utilization averaged across cut1, cut2, and rs benchmarks:

| Buffer K | cut1 | cut2 | rs   | avg  | Best epoch |
|----------|------|------|------|------|-----------|
| 0 (K=0, 5-action V2) | 62.3 | 62.3 | 52.4 | 59.0 | — |
| 1        | 66.4 | 66.5 | 58.2 | 63.7 | 13 |
| 2        | 71.4 | 70.2 | 62.1 | 67.9 | 14 |
| 3        | 74.4 | 72.1 | 63.8 | 70.1 | 27 |

All utilization values are in percentage points (0–100 scale).
