# GA Mixed Instance Generation - Technical Documentation

This document provides a comprehensive explanation of how the synthetic training instances (`ga_mixed.pt`) are generated using a Genetic Algorithm (GA).

---

## 1. Overview

The GA Instance Mixer (`ga_instance_mixer.py`) evolves a population of "episodes" (sequences of box dimensions) to create high-quality training data for the 3D Bin Packing RL model. The goal is to produce diverse, challenging box sequences that target a specific container fill rate.

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
        │   (60 generations)   │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │     ga_mixed.pt      │
        │     (3000 eps)       │
        └──────────────────────┘
```

---

## 2. Source Datasets

The GA mixer loads episodes from three source files:

| Dataset   | Episodes | Boxes/Episode | Box Dims   | Avg Volume |
|-----------|----------|---------------|------------|------------|
| `cut_1.pt`| 2,100    | 11-45         | 2-5        | 1,000      |
| `cut_2.pt`| 2,100    | 11-48         | 2-10       | 2,000      |
| `rs.pt`   | 2,100    | 100           | 2-5        | 4,287      |

Each episode is a list of boxes: `[[l, w, h], [l, w, h], ...]`

---

## 3. Fitness Function

Each episode is scored using a multi-objective fitness function:

$$\text{Fitness}(ep) = \underbrace{(1 - |fill - target\_fill|)}_{\text{Fill Score}} + \underbrace{\alpha \cdot diversity(ep)}_{\text{Diversity Bonus}}$$

Where:
- **Fill Score**: How close the episode's total volume is to the target (85% of container capacity = 850 units for a 10×10×10 container)
- **Diversity**: Ratio of unique box types to total boxes (encourages variety)
- **α (alpha_diversity)**: Weight for diversity component (0.12)

### Example Calculation
```
Episode: [[3,3,3], [4,4,4], [3,3,3], [2,2,2]]
Total Volume: 27 + 64 + 27 + 8 = 126
Fill Rate: 126 / 1000 = 0.126
Fill Score: 1 - |0.126 - 0.85| = 0.276

Unique boxes: {(3,3,3), (4,4,4), (2,2,2)} = 3
Diversity: 3 / 4 = 0.75
Diversity Bonus: 0.12 × 0.75 = 0.09

Final Fitness: 0.276 + 0.09 = 0.366
```

---

## 4. Genetic Operators

### 4.1 Selection: Tournament (k=3)
- Randomly sample 3 episodes from the population
- Select the one with the highest fitness
- Used to choose parents for crossover

### 4.2 Crossover: Interleaved Merge
Combines two parent episodes by alternating boxes:

```
Parent A: [A1, A2, A3, A4]
Parent B: [B1, B2, B3]
Target Length: 4 (randomly chosen from parents)

Child: [A1, B1, A2, B2]  (interleaved)
```

### 4.3 Mutation Operators

| Operator | Probability | Description |
|----------|-------------|-------------|
| **Reorder** | 15% of boxes | Swap positions of two random boxes within an episode |
| **Switch** | 10% of population | Swap a box between two different episodes |
| **Dimension** | 8% of boxes | Modify a dimension (L, W, or H) by ±1 |
| **Hard Mutation** | 0.8% of boxes | Jump to a random large dimension (3-7) |

#### Dimension Mutation Logic
```python
if random.random() < 0.1:  # 10% chance of "hard" mutation
    # Jump to large dimension
    dim = random.randint(3, 7)  # for L or W
    dim = random.randint(2, 5)  # for H
else:
    # Small ±1 mutation
    dim = dim ± 1
```

---

## 5. Evolution Process

```
for generation in range(60):
    1. Score all episodes by fitness
    2. Keep top 10% as "elite" (unchanged)
    3. Fill remaining population via:
       - Tournament selection of parents
       - Crossover to create child
       - Apply reorder mutation
       - Apply dimension mutation
    4. Apply switch-between-episodes mutation
```

### Population Flow Per Generation
```
Population (2000 episodes)
    │
    ▼
┌───────────────────┐
│  Score & Rank     │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐   ┌───────────────┐
│ Elite │   │ New Children  │
│ (200) │   │   (1800)      │
└───┬───┘   └───────┬───────┘
    │               │
    └───────┬───────┘
            ▼
┌───────────────────┐
│ Switch Mutation   │
│ (10% of pairs)    │
└─────────┬─────────┘
          ▼
    New Population
```

---

## 6. Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `generations` | 60 | Number of evolution cycles |
| `population_size` | 2000 | Episodes in each generation |
| `elite_frac` | 0.10 | Top 10% preserved unchanged |
| `mutation_reorder_p` | 0.15 | Probability of box position swap |
| `mutation_switch_p` | 0.10 | Probability of cross-episode swap |
| `mutation_dim_p` | 0.08 | Probability of dimension change |
| `target_fill` | 0.85 | Target container fill rate |
| `alpha_diversity` | 0.12 | Weight for diversity in fitness |
| `n_output_episodes` | 3000 | Final output size |
| `random_seed` | 42 | Reproducibility |

---

## 7. Output Summary

After running the GA, the output `ga_mixed.pt` contains:

| Metric | Value |
|--------|-------|
| Episodes | 2000 (top-ranked) |
| Boxes/Episode | 18-21 |
| Avg Boxes/Episode | 19.1 |
| Min Dimension | 2 |
| Max Dimension | 9 |
| Avg Episode Volume | 816.7 |

> **Note**: Episodes are output **sorted by descending fitness**, meaning early episodes are "easier" (closer to target fill) and later episodes are "harder". The training code shuffles this data before use.

---

## 8. Running the Generator

```bash
python ga_instance_mixer.py
```

This will:
1. Load source datasets from `approachesO3DKP/`
2. Run the GA for 60 generations
3. Save top 3000 episodes to `approachesO3DKP/ga_mixed.pt`
4. Print a summary of the generated data
