# Packing Problem with Online RL

This project implements a Deep Reinforcement Learning (DRL) solution for the **3D Bin Packing Problem (3D-BPP)**, specifically designed for sequences of items arriving online. The model optimizes container utilization while adhering to stability constraints using a heuristic-based action space and Dueling DQN architecture.

## Project Structure

- `main.py`: Entry point for simple training/visualization runs.
- `experiments.py`: Main script to execute standardized experiments (Buffer size comparisons, LR tuning, etc.).
- `oskp_rl_up_with_buffer_with_mask.py`: The core DRL model implementation, featuring "Buffer Vision" and feasibility mask supervision.
- `ga_instance_mixer.py`: A Genetic Algorithm (GA) utility to generate high-quality synthetic training datasets.
- `analysis.py`: Tools for processing experiment results and generating summary metrics.
- `model_documentation.md`: Full mathematical detail of the MDP formulation, stability rules, and network architecture.
- `approachesO3DKP/`: Directory containing source and synthetic datasets (`.pt` files).
- `papers/`: Research material and literature context.

## Key Features

- **Heuristic-Driven Action Space**: Instead of raw coordinates, the agent selects from specialized heuristics (Stacking, Best-Fit, Corner, etc.) to ensure robust placements.
- **Buffer Vision**: The model maintains a $k$-size buffer for deferring boxes and uses the current buffer occupancy as an input feature for strategic decision-making.
- **Feasibility Mask Supervision**: The NN has a dedicated head to predict valid placement locations, which is trained via multi-task loss to accelerate learning of physical constraints.
- **60% Vertical Support Rule**: Implements a realistic stability requirement where boxes must have at least 60% base support.

## Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments
To run the full suite of experiments:
```bash
python experiments.py --all
```

To run a specific experiment (e.g., Buffer Comparison):
```bash
python experiments.py --exp 2 --episodes 2100
```

### Data Generation
If you need to regenerate the "GA-Mixed" artificial instances:
```bash
python ga_instance_mixer.py
```

## Results
Experiment outputs, including 3D pallet visualizations and utilization trends, are saved to:
- `experiments_results/` (Legacy/Baseline results)
- `experiments_results_v2/` (Current model results with improvements)

---
*For a detailed mathematical deep-dive, see [model_documentation.md](model_documentation.md).*
