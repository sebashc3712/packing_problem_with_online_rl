# Experiments Workflow - Technical Documentation

This document explains the technical implementation and workflow of the experimental suite (`experiments.py`) used to validate the 3D Bin Packing RL model.

---

## 1. Overview

The `experiments.py` script is the main driver for standardized model validation. It manages:
- Data loading from synthetic and real-world-style datasets.
- Multi-run training cycles with hyperparameter variations.
- Greedy evaluation against held-out validation sets.
- Results aggregation and visualization.

---

## 2. Core Components

### 2.1 Data Loading (`load_instances`)
Loads `.pt` (PyTorch/Pickle) files containing sequences of boxes.
- **Training Source**: `ga_mixed.pt` (GA-evolved episodes).
- **Validation Sources**: `cut_1.pt` (Regular patterns), `cut_2.pt` (Randomly sized), `rs.pt` (Fixed stream).

### 2.2 Evaluation Logic (`evaluate`)
Performs a greedy assessment of a trained agent.
- **Epsilon**: Set to `0.0` (fully greedy).
- **Process**:
    1. Iterates through the box stream.
    2. Uses `act_with_mask_bias` to select the best packing heuristic.
    3. Replicates the "Stream-then-Buffer" logic used in training.
    4. Calculates the average volume utilization across all validation episodes.

---

## 3. Experimental Setups

The script supports three distinct experiments to answer specific research questions:

### Experiment 1: Learning Strategy Confirmation
*   **Goal**: Prove the agent learns a robust strategy on artificial data that generalizes to other distributions.
*   **Process**:
    - Shuffles `ga_mixed.pt`.
    - Trains for 2100 episodes ($LR=0.001$, Buffer=$k$).
    - Evaluates on all valid datasets (CUT-1, CUT-2, RS).

### Experiment 2: Buffer Size Comparison ($k$)
*   **Goal**: Quantify the impact of the deferral buffer size on final packing efficiency.
*   **Process**:
    - Iterates through buffer sizes $k \in \{1, 2, 3, 4\}$.
    - Performs 4 separate training and validation cycles.
    - Saves results to `buffer_comparison.csv`.

### Experiment 3: Learning Rate Comparison ($LR$)
*   **Goal**: Find the optimal learning speed/stability trade-off.
*   **Process**:
    - Iterates through $LR \in \{0.0001, 0.001, 0.005\}$.
    - Performs 3 separate training and validation cycles (unlimited buffer).
    - Saves results to `lr_comparison.csv`.

---

## 4. Training Loop Workflow

Each training run within an experiment follows these steps:
1.  **Shuffle Data**: Eliminates the fitness-based ordering bias from the GA generator.
2.  **Environment Setup**: Initializes `BoxPilingEnv` with specific pallet and buffer constraints.
3.  **Optimization**: 
    - Gradient step after **every placement**.
    - Target network update **once per episode**.
    - Monitoring of Q-loss and Mask-loss.
4.  **Artifact Generation**: Saves visualizations (`episode_N_results.png`) and trend graphs (`utilization_trend.png`, `loss_trend.png`).

---

## 5. Directory & Output Structure

All outputs are versioned in `experiments_results_v2/` to ensure reproducibility:

```text
experiments_results_v2/
├── exp1/
│   ├── train_ga_mixed/
│   └── validation_results.csv
├── exp2/
│   ├── train_buffer_1/ ... train_buffer_4/
│   └── buffer_comparison.csv
└── exp3/
    ├── train_lr_0.0001/ ... train_lr_0.005/
    └── lr_comparison.csv
```

---

## 6. Execution Commands

### Run Everything
```bash
python experiments.py --all
```

### Run Specific Test (e.g., Learning Rate)
```bash
python experiments.py --exp 3 --episodes 2100
```

### Quick Debug Run
```bash
python experiments.py --exp 1 --episodes 5 --val-episodes 2
```
