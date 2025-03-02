import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd  # For exporting metrics to CSV/DataFrame

# ---------------------
# Environment Class
# ---------------------
class BoxPilingEnv:
    def __init__(self, pallet_size=(10, 10), max_height=10):
        self.pallet_size = pallet_size
        self.max_height = max_height
        self.current_height_map = np.zeros(pallet_size)
        self.current_box = None
        self.placed_boxes = []

        # Track invalid actions:
        self.invalid_actions_learned = 0      # When no valid action exists in choose_action
        self.invalid_actions_attempted = 0    # When an attempted action fails in step

    def reset(self):
        self.current_height_map = np.zeros(self.pallet_size)
        self.placed_boxes = []
        self.invalid_actions_learned = 0
        self.invalid_actions_attempted = 0
        self.current_box = None
        return self._get_state()

    def _get_state(self):
        box_dims = self.current_box if self.current_box is not None else np.zeros(3)
        return {'height_map': self.current_height_map.copy(), 'box_dims': box_dims}

    def new_box_arrival(self, box_dims):
        self.current_box = box_dims
        return self._get_state()

    def get_rotated_box_dims(self, box, rotation):
        valid_rotations = [
            (0, 1, 2),  # Original (L, W, H)
            (1, 0, 2),  # Swap width & depth
            (0, 2, 1),  # Swap depth & height
            (2, 1, 0),  # Swap width & height
            (1, 2, 0),  # Rotate all
            (2, 0, 1)   # Rotate all
        ]
        return tuple(box[i] for i in valid_rotations[rotation])

    def _is_valid_placement(self, x, y, w, d, h):
        # 1) Boundary check
        if (x + w > self.pallet_size[0]) or (y + d > self.pallet_size[1]):
            return False

        # 2) Height limit: placing box must not exceed max height
        placement_area = self.current_height_map[x:x+w, y:y+d]
        base_height = np.max(placement_area)
        if base_height + h > self.max_height:
            return False

        # 3) Require that the entire region is exactly at base_height (no partial bridging)
        if not np.all(placement_area == base_height):
            return False

        # 4) Full support: the entire area must be supported
        support_count = np.sum(placement_area == base_height)
        if support_count < (w * d):
            return False

        return True

    def _update_height_map(self, x, y, w, d, h):
        placement_area = self.current_height_map[x:x+w, y:y+d]
        base_z = np.max(placement_area)  # base height where the box sits
        self.current_height_map[x:x+w, y:y+d] = base_z + h
        return base_z

    def _calculate_maximal_flat_area(self, height_map):
        max_area = 0
        unique_heights = np.unique(height_map)
        for level in unique_heights:
            binary_matrix = (height_map == level).astype(int)
            current_max = self._maximal_rectangle(binary_matrix)
            max_area = max(max_area, current_max)
        return max_area

    def _maximal_rectangle(self, matrix):
        if matrix.size == 0:
            return 0
        rows, cols = matrix.shape
        max_area = 0
        heights = np.zeros(cols, dtype=int)
        for row in matrix:
            heights = np.where(row == 1, heights + 1, 0)
            stack = []
            for i in range(cols + 1):
                while stack and (i == cols or heights[i] < heights[stack[-1]]):
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, h * w)
                stack.append(i)
        return max_area

    def _is_terminal(self):
        return np.all(self.current_height_map >= self.max_height)

    def get_valid_actions(self, box_dims):
        valid_actions = []
        for rotation in range(6):
            w, d, h = self.get_rotated_box_dims(box_dims, rotation)
            for xx in range(self.pallet_size[0] - w + 1):
                for yy in range(self.pallet_size[1] - d + 1):
                    if self._is_valid_placement(xx, yy, w, d, h):
                        action = xx * self.pallet_size[1] * 6 + yy * 6 + rotation
                        valid_actions.append(action)
        return valid_actions

    # ---------------------
    # Heuristics
    # ---------------------
    def heuristic_stacking(self, valid_actions):
        best_action = None
        best_support = -np.inf
        for action in valid_actions:
            rotation = action % 6
            remaining = action // 6
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            region = self.current_height_map[xx:xx+w, yy:yy+d]
            support_level = np.max(region)
            if support_level > best_support:
                best_support = support_level
                best_action = action
        return best_action

    def heuristic_best_fit(self, valid_actions):
        best_action = None
        best_gap = np.inf
        for action in valid_actions:
            rotation = action % 6
            remaining = action // 6
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            region = self.current_height_map[xx:xx+w, yy:yy+d]
            support_level = np.max(region)
            gap = self.max_height - (support_level + h)
            if gap < best_gap:
                best_gap = gap
                best_action = action
        return best_action

    def heuristic_semi_perfect_fit(self, valid_actions):
        best_action = None
        best_score = np.inf
        for action in valid_actions:
            rotation = action % 6
            remaining = action // 6
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            region = self.current_height_map[xx:xx+w, yy:yy+d]
            support_level = np.max(region)
            support_count = np.sum(region == support_level)
            waste = (w * d - support_count)
            gap = self.max_height - (support_level + h)
            score = waste + gap
            if score < best_score:
                best_score = score
                best_action = action
        return best_action

    def heuristic_random_fit(self, valid_actions):
        placed_volume = sum(b[2] * b[3] * b[4] for b in self.placed_boxes)
        pallet_volume = self.pallet_size[0] * self.pallet_size[1] * self.max_height
        utilization = placed_volume / pallet_volume if pallet_volume > 0 else 0
        threshold = 0.10
        p_stack = 0.66 if utilization < threshold else 0.33
        if np.random.rand() < p_stack:
            return self.heuristic_stacking(valid_actions)
        else:
            return self.heuristic_semi_perfect_fit(valid_actions)

    def choose_action_by_heuristic(self, heuristic_name):
        valid_actions = self.get_valid_actions(self.current_box)
        if not valid_actions:
            self.invalid_actions_learned += 1
            return None

        if heuristic_name == 'stacking':
            action = self.heuristic_stacking(valid_actions)
        elif heuristic_name == 'best_fit':
            action = self.heuristic_best_fit(valid_actions)
        elif heuristic_name == 'semi_perfect_fit':
            action = self.heuristic_semi_perfect_fit(valid_actions)
        elif heuristic_name == 'random_fit':
            action = self.heuristic_random_fit(valid_actions)
        else:
            raise ValueError("Unknown heuristic")

        if action not in valid_actions:
            action = random.choice(valid_actions)
        return action

    def step(self, action):
        rotation = action % 6
        remaining = action // 6
        yy = remaining % self.pallet_size[1]
        xx = remaining // self.pallet_size[1]

        w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
        if not self._is_valid_placement(xx, yy, w, d, h):
            self.invalid_actions_attempted += 1
            reward = -2000
            return self._get_state(), reward, False, {"invalid": True}

        base_z = self._update_height_map(xx, yy, w, d, h)
        self.placed_boxes.append((xx, yy, w, d, h, base_z))

        original_max_space = self._calculate_maximal_flat_area(self.current_height_map)
        temp_height_map = self.current_height_map.copy()
        temp_height_map[xx:xx+w, yy:yy+d] += h
        new_max_space = self._calculate_maximal_flat_area(temp_height_map)
        reward = (new_max_space / original_max_space) * 100 if original_max_space > 0 else 0

        self.current_box = None
        done = self._is_terminal()
        return self._get_state(), reward, done, {}

    def visualize_pallet(self, episode_num, boxes_attempted, utilization,
                         invalid_learned, invalid_attempted, output_dir=""):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, (xx, yy, w, d, h, base_z) in enumerate(self.placed_boxes):
            color = plt.cm.tab20(i % 20)
            ax.bar3d(xx, yy, base_z, w, d, h, shade=True,
                     color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_title(
            f"Episode {episode_num} - 3D Box Visualization\n"
            f"Utilization: {utilization:.1%} | "
            f"Invalid Learned: {invalid_learned} | Invalid Attempted: {invalid_attempted}"
        )
        ax.set_xlim(0, self.pallet_size[0])
        ax.set_ylim(0, self.pallet_size[1])
        ax.set_zlim(0, self.max_height)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Height')

        invalid_mask = self.current_height_map == 0
        ax.scatter(*np.where(invalid_mask), color='red', s=10, label='Unusable Space')
        plt.legend()
        
        # Save in the provided output directory.
        filename = os.path.join(output_dir, f'episode_{episode_num}_results.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()


# ---------------------
# DQN Agent
# ---------------------
class DQNAgent:
    def __init__(self, state_dims, action_size=4):
        self.state_dims = state_dims
        self.action_size = action_size  # Four heuristics
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _build_model(self):
        class CustomNetwork(nn.Module):
            def __init__(self, pallet_size, action_size):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Flatten()
                )
                with torch.no_grad():
                    dummy_input = torch.zeros(1, 1, *pallet_size)
                    conv_out = self.conv(dummy_input).shape[1]
                self.fc = nn.Sequential(
                    nn.Linear(conv_out + 3, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_size)
                )
            def forward(self, height_map, box_dims):
                conv_features = self.conv(height_map)
                combined = torch.cat([conv_features, box_dims], dim=1)
                return self.fc(combined)
        return CustomNetwork(self.state_dims['height_map'], self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        height_map = torch.FloatTensor(state['height_map']).unsqueeze(0).unsqueeze(0)
        box_dims = torch.FloatTensor(state['box_dims']).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(height_map, box_dims)
        return torch.argmax(q_values, dim=1).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_height_maps = np.array([s['height_map'] for s in states], dtype=np.float32)
        states_box_dims = np.array([s['box_dims'] for s in states], dtype=np.float32)
        next_states_height_maps = np.array([s['height_map'] for s in next_states], dtype=np.float32)
        next_states_box_dims = np.array([s['box_dims'] for s in next_states], dtype=np.float32)

        height_maps = torch.FloatTensor(states_height_maps).unsqueeze(1)
        box_dims = torch.FloatTensor(states_box_dims)
        next_height_maps = torch.FloatTensor(next_states_height_maps).unsqueeze(1)
        next_box_dims = torch.FloatTensor(next_states_box_dims)

        current_q = self.model(height_maps, box_dims).gather(1, torch.LongTensor(actions).unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_model(next_height_maps, next_box_dims).max(1)[0]
        dones = torch.FloatTensor([1 if d else 0 for d in dones])
        target_q = torch.FloatTensor(rewards) + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ---------------------
# Training Loop
# ---------------------
def train(episodes_boxes, output_dir):
    """
    Train the RL agent using the provided episodes_boxes.
    Saves all visualization images (pallet plots and trend graph) in output_dir.
    Returns a pandas DataFrame with final metrics.
    """
    env = BoxPilingEnv()
    agent = DQNAgent(
        state_dims={'height_map': env.pallet_size, 'box_dims': 3},
        action_size=4  # Four heuristics
    )
    
    total_episodes = 2100
    total_utilization = 0.0
    all_metrics = []
    heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'random_fit'}
    
    for episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        boxes = episodes_boxes[episode]
        box_idx = 0
        
        while not done and box_idx < len(boxes):
            box_dims = boxes[box_idx]
            box_idx += 1
            state = env.new_box_arrival(box_dims)
            
            heuristic_index = agent.act(state)
            heuristic = heuristic_map[heuristic_index]
            action = env.choose_action_by_heuristic(heuristic)
            
            if action is None:
                # Penalize and keep same box if no valid action is available.
                reward = -2000
                continue

            next_state, reward, done, info = env.step(action)
            agent.remember(state, heuristic_index, reward, next_state, done)
            agent.replay()
            agent.update_target_model()
            state = next_state
            episode_reward += reward

        # Compute container utilization
        pallet_volume = env.pallet_size[0] * env.pallet_size[1] * env.max_height
        placed_volume = sum(b[2] * b[3] * b[4] for b in env.placed_boxes)
        utilization = placed_volume / pallet_volume if pallet_volume > 0 else 0
        total_utilization += utilization
        
        episode_metrics = {
            'episode': episode + 1,
            'utilization': utilization,
            'invalid_learned': env.invalid_actions_learned,
            'invalid_attempted': env.invalid_actions_attempted,
            'boxes_attempted': box_idx,
            'placed_boxes': len(env.placed_boxes),
            'max_height': np.max(env.current_height_map),
            'avg_height': np.mean(env.current_height_map)
        }
        all_metrics.append(episode_metrics)
        
        # Visualization every 100 episodes
        if (episode + 1) % 100 == 0:
            env.visualize_pallet(
                episode_num=episode + 1,
                boxes_attempted=box_idx,
                utilization=utilization,
                invalid_learned=env.invalid_actions_learned,
                invalid_attempted=env.invalid_actions_attempted,
                output_dir=output_dir
            )
        
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        print(f"Episode: {episode+1:04d} | Util: {utilization:.1%} | "
              f"Invalid Learned: {env.invalid_actions_learned:02d} | "
              f"Invalid Attempted: {env.invalid_actions_attempted:02d} | "
              f"Boxes: {len(env.placed_boxes):02d}/{box_idx:02d} | ε: {agent.epsilon:.3f}")

    avg_utilization = total_utilization / total_episodes
    total_invalid_learned = sum(m['invalid_learned'] for m in all_metrics)
    total_invalid_attempted = sum(m['invalid_attempted'] for m in all_metrics)
    print("\nTraining Summary:")
    print(f"Average Utilization: {avg_utilization:.2%}")
    print(f"Total Invalid Learned: {total_invalid_learned}")
    print(f"Total Invalid Attempted: {total_invalid_attempted}")

    # Plot utilization trend and save it in the output_dir
    plt.figure(figsize=(12, 6))
    episodes = [m['episode'] for m in all_metrics]
    utilizations = [m['utilization'] for m in all_metrics]
    window_size = 100
    moving_avg = np.convolve(utilizations, np.ones(window_size)/window_size, mode='valid')
    plt.plot(episodes, utilizations, 'b', alpha=0.3, label='Episode Utilization')
    plt.plot(episodes[window_size-1:], moving_avg, 'r', label=f'{window_size}-Episode Moving Avg')
    plt.xlabel('Episode')
    plt.ylabel('Utilization')
    plt.title('Learning Progress')
    plt.legend()
    trend_file = os.path.join(output_dir, 'utilization_trend.png')
    plt.savefig(trend_file)
    plt.close()

    # Convert final metrics to a pandas DataFrame and return it.
    final_metrics_df = pd.DataFrame(all_metrics)
    return final_metrics_df

# -------------
# Usage Example
# -------------
# if __name__ == "__main__":
#     # Example: generate random episodes of boxes
#     episodes_boxes = [
#         [np.random.randint(1, 5, size=3).tolist() for _ in range(random.randint(5, 15))]
#         for _ in range(2100)
#     ]
#     # Provide an output directory where all visualization files and CSV will be saved.
#     output_directory = os.path.join(os.getcwd(), "training_output")
#     os.makedirs(output_directory, exist_ok=True)
#     final_metrics = train(episodes_boxes, output_dir=output_directory)
#     # Optionally, you can also save the final metrics to CSV here.
#     final_metrics.to_csv(os.path.join(output_directory, "final_metrics.csv"), index=False)
