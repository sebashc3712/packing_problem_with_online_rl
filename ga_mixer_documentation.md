# GA Mixed Instance Generation - Technical Documentation

This document provides a comprehensive explanation of how the synthetic training instances (`ga_mixed.pt`) are generated using a Genetic Algorithm (GA).

---

## 1. Overview

The GA Instance Mixer (`ga_instance_mixer.py`) evolves a population of "episodes" (sequences of box dimensions) to create high-quality training data for the 3D Bin Packing RL model. The goal is to produce diverse, challenging box sequences that target a specific container fill rate.

The V2 implementation uses a significantly larger scale for population and generation count to ensure extreme strategy robustness.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  cut_1.pt   │     │  cut_2.pt   │     │   rs.pt     │
│ (2100 eps)  │     │ (2100 eps)  │     │ (2100 eps)  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   ▼
        ┌──────────────────────┐
        │   Genetic Algorithm  │
        │  (100 generations)   │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │ ga_mixed_large.pt    │
        │  (50,000 eps req)    │
        └──────────────────────┘
```

---

## 2. Source Datasets

The GA mixer loads episodes from three source files:

| Dataset   | Episodes | Boxes/Episode | Box Dims   |
|-----------|----------|---------------|------------|
| `cut_1.pt`| 2,100    | 11-45         | 2-5        |
| `cut_2.pt`| 2,100    | 11-48         | 2-10       |
| `rs.pt`   | 2,100    | 100           | 2-5        |

---

## 3. Fitness Function

Each episode is scored using a multi-objective fitness function:

$$\text{Fitness}(ep) = \underbrace{(1 - |fill - target\_fill|)}_{\text{Fill Score}} + \underbrace{\alpha \cdot diversity(ep)}_{\text{Diversity Bonus}}$$

Where:
- **Fill Score**: Deviation from target fill (default 85%).
- **Diversity**: Ratio of unique box types to total boxes.
- **α (alpha_diversity)**: Weight for diversity (0.12).

---

## 4. Genetic Operators

### 4.1 Mutation Operators

| Operator | Probability | Description |
|----------|-------------|-------------|
| **Reorder** | 15% of boxes | Swap positions of two random boxes within an episode |
| **Switch** | 10% of population | Swap a box between two different episodes |
| **Dimension** | 8% of boxes | Modify a dimension (L, W, or H) by ±1 |
| **Hard Mutation** | 10% of dim mut | Occasionally jump to larger random dimensions (3-7) |

---

## 5. Evolution Process

1. **Score & Rank**: All episodes are scored by fitness.
2. **Elitism**: Top 10% survive unchanged.
3. **Reproduction**: Crossover + Reorder/Dimension Mutation.
4. **Switching**: Cross-episode mutation applied to the entire population.

---

## 6. Configuration Parameters (V2 Large-Scale)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `generations` | 100 | Number of evolution cycles |
| `population_size` | 10,000 | Episodes in each generation |
| `elite_frac` | 0.10 | Top 10% preserved unchanged |
| `target_fill` | 0.85 | Target container fill rate |
| `alpha_diversity` | 0.12 | Weight for diversity in fitness |
| `n_output_episodes`| 50,000 | Requested output size (clamped by population) |

---

## 7. Running the Generator

```bash
python ga_instance_mixer.py
```

This will save the evolved instances to `approachesO3DKP/ga_mixed_large.pt`.
