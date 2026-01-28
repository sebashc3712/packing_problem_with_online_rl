# Deep Analysis: Persistent Utilization Decline

## Problem
The utilization decline persists even after removing the LR scheduler. The MA(50) drops from ~0.65 to ~0.58 after episode 1500.

![Utilization Trend](/Users/sebastian.herrera/.gemini/antigravity/brain/498c5f3a-e5f2-4b7b-bbdd-f75556ba09db/uploaded_image_1769427978707.png)

---

## Root Cause #1: Target Network Updated Too Frequently

**Current Code:**
```python
agent.update_target_model()  # Called after EVERY placement
```

This is called **15+ times per episode** (once per box placement). In standard DQN:
- The target network provides **stable Q-targets** for learning.
- It should be updated **every 1,000-10,000 gradient steps** or using **soft updates (Polyak averaging)**.

**Impact:** Updating the target after every step causes the Q-targets to oscillate wildly, destabilizing learning and causing the agent to "forget" good policies.

---

## Root Cause #2: Replay Buffer Dynamics

**Current Settings:**
- `memory` size: 10,000 experiences
- ~15 placements per episode → buffer fills after ~700 episodes
- After that, old experiences are discarded FIFO

**Problem:** Early experiences (from high-epsilon exploration) may be suboptimal, but by the time the buffer is cycling, the agent is already in exploitation mode. The mix of old/new experiences can confuse the policy.

---

## Root Cause #3: Epsilon Decay Timeline

| Episode | ε           | Behavior        |
|---------|-------------|-----------------|
| 0       | 1.00        | 100% random     |
| 460     | ~0.10       | 90% greedy      |
| 920     | ~0.01       | 99% greedy (floor) |
| 1500+   | 0.01        | 99% greedy      |

After ~920 episodes, the agent is almost fully exploiting. If the learned policy has subtle flaws, it will keep making the same mistakes without correcting them (no exploration to discover better strategies).

---

## Root Cause #4: Training Data Ordering Bias (CRITICAL)

The GA mixer outputs episodes **sorted by descending fitness**:
```python
ranked = sorted(..., reverse=True)  # Highest fitness first
```

**Impact:**
- **Episodes 0-500**: Highest fitness → easiest to pack near 85% target
- **Episodes 1500-2000**: Lowest fitness → hardest to pack

This creates a **systematic difficulty curve** that makes later episodes appear as "declining performance" when it's actually just harder data!

---

## Proposed Fixes

### Fix 1: Update Target Network Less Frequently (CRITICAL)
Update the target network **once per episode** instead of every step:

```python
# In train(), OUTSIDE try_place_one_box, at end of episode:
agent.update_target_model()
```

And remove the calls inside `try_place_one_box()`.

### Fix 2: Use Soft (Polyak) Updates for Target Network
Instead of hard copies, blend the networks:
```python
def update_target_model(self, tau=0.005):
    for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### Fix 3: Increase Epsilon Floor
Raise `epsilon_min` from `0.01` to `0.05` to maintain 5% exploration throughout training:
```python
self.epsilon_min = 0.05
```

---

## Recommended Implementation
Implement **Fix 1** (most critical) and **Fix 3** (safety net for exploration).
