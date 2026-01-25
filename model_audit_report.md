# Audit Report: Model Architecture and Learning Process

This document provides a "hard check" on the current RL model and its training on artificial instances.

## 1. Neural Network Architecture Audit

### Strengths
*   **Dueling DQN**: The separation of Value ($V(s)$) and Advantage ($A(s,a)$) heads is a robust choice for discrete action spaces (heuristics), as it helps the model learn which states are valuable independently of the action taken.
*   **Convolutional Feature Extraction**: Using a 3-layer CNN for the height map is appropriate for capturing spatial dependencies and "flatness" patterns.
*   **Multi-Task Mask Head**: Supervising the model with a ground-truth feasibility mask (BCE loss) is an excellent way to force the network to learn the physics of the environment quickly, even before Q-values converge.

### Critical Concerns
*   **Input Context**: The model receives the height map and the dimensions of the *current* box. However, it does not see the "buffer" or the queue. This limits the agent's ability to learn "deferral" strategies effectively (it can't know what it's deferring for).
*   **Mask Head Integration**: The current implementation uses a "soft bias" where the predicted mask influences the action selection. If the mask head is not yet accurate, it can "poison" the Q-learning exploration in early stages.

---

## 2. Training and Learning Signal Audit

### Data: Artificial Instances (GA Mixed)
*   **The GA Mixer** (`ga_instance_mixer.py`) creates "elite" episodes by combining boxes from different distributions.
*   **Risk**: The GA optimizes for `target_fill` and `diversity`. If the artificial instances are "too easy" (e.g., very small boxes that fit anywhere), the model may fail to learn complex spatial reasoning required for real-world constraints.
*   **Learning Signal**: The reward structure combines volume utilization and "Maximal Flat Area". This is a competitive signal.

### The "Voxel-to-Height-Map" Reversion
*   We recently reverted from 3D voxels to a 2.5D height map.
*   **Audit Result**: This significantly speeds up training and makes the state space less sparse. However, it removes the ability to learn "bridging" which was a more advanced (though harder to learn) feature. The 60% support rule is a good middle ground.

---

## 3. Conclusions and Recommendations

### Is the model "really learning"?
**Yes**, the model shows clear learning behavior (ε-decay and utilization trends). However, it is likely learning the **Ground-Truth Mask** faster than it is learning the **Strategic Heuristic Choice**.

### Recommendations
1.  **Buffer Vision**: Pass the number of items in the buffer (or their average volume) as an additional input to the `DQNAgent`. This allows the model to learn when to "wait" strategically.
2.  **Weighted Multi-Task Loss**: Monitor the ratio between `q_loss` and `mask_loss`. If `mask_loss` dominates too early, the model might become a "validity checker" rather than an "optimizer".
3.  **Artificial Difficulty**: Ensure the GA mixer includes "pathological" cases (large boxes, irregular streams) to prevent the model from over-fitting to easy packing scenarios.
