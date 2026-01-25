# Deep Reinforcement Learning for the Online 3D Bin Packing Problem with Stability Constraints

## 1. Problem Definition

The problem addressed is a variant of the **Online 3D Bin Packing Problem (3D-BPP)**, specifically modeled for industrial palletization. We define a bin (pallet) with dimensions $(L, W, H)$. A stream of items (boxes) arrives sequentially, where each box $i$ has dimensions $b_i = (l_i, w_i, h_i)$. The objective is to maximize the volume utilization while adhering to physical stability constraints.

### 1.1 State Representation
The state $s_t$ at time $t$ is represented by a 2.5D height map $M \in \mathbb{R}^{L \times W}$, where $M_{x,y}$ denotes the current maximum height at coordinate $(x,y)$.

The state vector passed to the neural network is:
$$s = \left\{ \frac{1}{H}M, \frac{b_i}{\text{max\_dims}} \right\}$$
where $M$ is the height map and $b_i$ is the current box dimensions.

---

## 2. Stability and Feasibility

A placement $(x, y, z, rot)$ is feasible if it satisfies the following mathematical conditions based on the height map $M$:

### 2.1 Vertical Support
A box must have at least 60% of its footprint supported by the level immediately below. Let $A = [x, x+l) \times [y, y+w)$ be the box's footprint. The base height $z$ is:
$$z = \max_{(x,y) \in A} M_{x,y}$$
For a placement to be stable, the support ratio $S_r$ must satisfy:
$$S_r = \frac{|\{(x',y') \in A : M_{x',y'} = z\}|}{|A|} \ge 0.60$$

### 2.2 Height Constraint
The resulting top of the box must not exceed the maximum height $H$:
$$z + h \le H$$

---

## 3. Reinforcement Learning Framework

The problem is modeled as a Markov Decision Process (MDP) where the agent chooses **heuristics** rather than raw coordinates.

### 3.1 Action Space
The action space $\mathcal{A}$ consists of 5 discrete heuristics:
1.  **Stacking**: Maximizes placement height (z) to build dense columns.
2.  **Best Fit**: Minimizes the residual gap $(H - (z+h))$.
3.  **Semi-Perfect Fit**: Minimizes a combination of support waste and vertical gap.
4.  **Random Fit**: A stochastic mix of Stacking and Semi-Perfect Fit.
5.  **Corner**: Maximizes the number of touching faces (walls or other boxes).

### 3.2 Network Architecture
We use a **Dueling Deep Q-Network (DQN)** with a custom **Mask Head**:

1.  **Shared CNN**: Extracts features from the height map.
2.  **Value Stream $V(s)$**: Estimates the state value.
3.  **Advantage Stream $A(s, a)$**: Estimates the relative advantage of each heuristic.
4.  **Mask Head $M(s)$**: A secondary linear head that predicts a feasibility mask $\mathcal{M} \in \{0, 1\}^{2 \times L \times W}$ for all possible $(x, y, rot)$ coordinates, filtered by the stability and flatness criteria.

The Q-value for heuristic $a$ is:
$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

### 3.3 Loss Function
The model is trained with a multi-task loss:
$$\mathcal{L} = \text{MSE}(Q_{target}, Q_{curr}) + \lambda \text{BCE}(\mathcal{M}_{pred}, \mathcal{M}_{GT})$$
where $\lambda \approx 0.2$ and $\text{BCE}$ is the Binary Cross Entropy for the feasibility mask.

---

## 4. Reward Structure

The reward $R$ favors both immediate volume gain and pallet "flatness" to facilitate future placements:

1.  **Volume Reward**: $R_{vol} = \frac{l \times w \times h}{L \times W \times H} \times 1000$.
2.  **Flatness Reward (Maximal Flat Area)**: Calculated using the height map $M$. It rewards placements that result in larger contiguous rectangular areas at the same height level.
3.  **Invalid Action Penalty**: A large negative reward (e.g., $-2000$) is applied if a heuristic selects an impossible placement.

---

## 5. Inference Logic with Buffer

The system incorporates a **k-size Buffer** to handle online arrival constraints. If the agent determines that the current box cannot be placed optimally (or at all), the box can be deferred to a buffer of size $k$. Deferred boxes are re-evaluated once the state changes, allowing for look-ahead-like behavior in a purely online setting.
