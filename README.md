# Packing Problem with Online RL

This project implements a Deep Reinforcement Learning (DRL) solution for the **3D Bin Packing Problem (3D-BPP)**, specifically designed for sequences of items arriving online. The model optimizes container utilization while adhering to stability constraints using a heuristic-based action space and Dueling DQN architecture.

## Project Structure

- `main.py`: Entry point for simple training/visualization runs.
- `experiments_v2.py`: Main script to execute standardized experiments (Buffer size comparisons, LR tuning, parallel training, etc.).
- `oskp_rl_up_buffer_experiments_v2.py`: The core DRL model implementation (V2), featuring "Buffer Vision" and optimized feasibility mask.
- `parallel_train_v2.py`: Implementation of parallelized training logic using multiple environments.
- `ga_instance_mixer.py`: A Genetic Algorithm (GA) utility to generate high-quality synthetic training datasets.
- `approachesO3DKP/`: Directory containing source and synthetic datasets (`.pt` files).

## Key Features

- **Heuristic-Driven Action Space**: Instead of raw coordinates, the agent selects from specialized heuristics (Stacking, Best-Fit, Corner, etc.).
- **Buffer Vision**: Maintains a $k$-size buffer for deferring boxes, with buffer occupancy as a state feature.
- **Feasibility Mask Supervision**: Auxiliary head trained with **flatness-bias** to predict valid placement locations.
- **Parallel Training**: Supports multi-threaded environment sampling for accelerated learning.

## Getting Started

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments (V2)
The V2 suite supports various experiments with parallelization:

```bash
# Run Epoch-Based Training (Exp 10) with 8 environments
python experiments_v2.py --exp 10 --max-epochs 30 --num-envs 8
```

| Exp ID | Description |
|--------|-------------|
| 5 | Buffer vs No Buffer (Optimized) |
| 7 | Learning Rate Comparison (No Buffer) |
| 10| Epoch-Based Training (Parallel) |

## Results
Experiment outputs are saved to:
- `experiments_results_refactored/` (Latest V2 results)

---
*For a detailed mathematical deep-dive, see [model_documentation.md](model_documentation.md).*
