import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd
import pickle
import argparse
import time
from tqdm import tqdm

# ================================
# Utility: longest run of False
# ================================
def _max_zero_run(bool_1d):
    """Return the maximum number of consecutive False in a 1D boolean array."""
    max_run = run = 0
    for v in bool_1d:
        if v:
            run = 0
        else:
            run += 1
            if run > max_run:
                max_run = run
    return max_run


# ================================
# Utility: Maximal Rectangle
# ================================
def _maximal_rectangle(matrix):
    if matrix.size == 0:
        return 0
    _, cols = matrix.shape
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

def calculate_average_flat_area(height_map):
    areas = []
    for level in np.unique(height_map):
        binary_matrix = (height_map == level).astype(int)
        current_max = _maximal_rectangle(binary_matrix)
        areas.append(current_max)
    return np.mean(areas) if areas else 0


# =========================================================
# Ground-truth feasibility mask with BRIDGING rule (GT)
# =========================================================
def compute_gt_mask(
    height_map,
    box_dims,
    pallet_size=(10, 10),
    max_height=10,
):
    L, W = pallet_size
    mask = np.zeros((2, L, W), dtype=np.float32)
    rotations = [(0, 1, 2)]

    valid_positions = []

    for r_idx, rot in enumerate(rotations):
        l, w, h = (int(box_dims[rot[0]]), int(box_dims[rot[1]]), int(box_dims[rot[2]]))
        if l <= 0 or w <= 0 or h <= 0:
            continue
        max_x = L - l
        max_y = W - w
        if max_x < 0 or max_y < 0:
            continue

        for x in range(max_x + 1):
            for y in range(max_y + 1):
                region = height_map[x:x + l, y:y + w]
                base_z = np.max(region)
                if base_z + h > max_height:
                    continue
                support_count = np.sum(region == base_z)
                if (support_count / (l * w)) < 0.60:
                    continue

                temp_hm = height_map.copy()
                temp_hm[x:x+l, y:y+w] = base_z + h
                flatness = calculate_average_flat_area(temp_hm)
                valid_positions.append((r_idx, x, y, flatness))

    if len(valid_positions) > 0:
        scores = np.array([pos[3] for pos in valid_positions])
        max_score = scores.max()
        tolerance = 1e-5

        for r_idx, x, y, flatness in valid_positions:
            if flatness >= max_score - tolerance:
                 mask[r_idx, x, y] = 1.0
            else:
                 mask[r_idx, x, y] = 0.0

    return mask


# ==================
# Environment
# ==================
class BoxPilingEnv:
    def __init__(
        self,
        pallet_size=(10, 10),
        max_height=10,
    ):
        self.pallet_size = pallet_size
        self.max_height = max_height

        self.current_height_map = np.zeros(pallet_size)
        self.current_box = None
        self.placed_boxes = []

        self.invalid_actions_learned = 0
        self.invalid_actions_attempted = 0

    def reset(self):
        self.current_height_map = np.zeros(self.pallet_size)
        self.placed_boxes = []
        self.invalid_actions_learned = 0
        self.invalid_actions_attempted = 0
        self.current_box = None
        return self._get_state()

    def _get_state(self):
        box_dims = self.current_box if self.current_box is not None else np.zeros(3, dtype=np.float32)
        return {'height_map': self.current_height_map.copy(), 'box_dims': box_dims}

    def new_box_arrival(self, box_dims):
        self.current_box = np.array(box_dims, dtype=np.float32)
        return self._get_state()

    def get_rotated_box_dims(self, box, rotation):
        valid_rotations = [(0, 1, 2), (1, 0, 2)]
        return tuple(int(box[i]) for i in valid_rotations[rotation])

    def _is_valid_placement(self, x, y, w, d, h):
        L, W, H = self.pallet_size[0], self.pallet_size[1], self.max_height
        if (x + w > L) or (y + d > W):
            return False, -1

        region = self.current_height_map[x:x+w, y:y+d]
        base_z = np.max(region)

        if base_z + h > H:
            return False, -1

        support_count = np.sum(region == base_z)
        if (support_count / (w * d)) < 0.60:
             return False, -1

        return True, base_z

    def _update_height_map(self, x, y, w, d, h, z):
        self.current_height_map[x:x+w, y:y+d] = z + h
        return z

    def _calculate_maximal_flat_area(self, height_map):
        max_area = 0
        for level in np.unique(height_map):
            binary_matrix = (height_map == level).astype(int)
            current_max = self._maximal_rectangle(binary_matrix)
            max_area = max(max_area, current_max)
        return max_area

    def _maximal_rectangle(self, matrix):
        if matrix.size == 0:
            return 0
        _, cols = matrix.shape
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

    def get_valid_actions(self, box_dims, pred_mask=None):
        valid_actions = []
        mapping = {}
        L, W = self.pallet_size

        for rotation in range(1):
            w, d, h = self.get_rotated_box_dims(box_dims, rotation)

            xs = range(L - w + 1)
            ys = range(W - d + 1)
            if pred_mask is not None:
                pm = np.asarray(pred_mask, dtype=bool)
                if pm.shape == (2, L, W):
                    cand = np.argwhere(pm[rotation])
                    iter_points = [(int(xx), int(yy)) for (xx, yy) in cand
                                   if xx <= L - w and yy <= W - d]
                else:
                    iter_points = [(xx, yy) for xx in xs for yy in ys]
            else:
                iter_points = [(xx, yy) for xx in xs for yy in ys]

            for (xx, yy) in iter_points:
                valid, z = self._is_valid_placement(xx, yy, w, d, h)
                if valid:
                    action = xx * W * 2 + yy * 2 + rotation
                    valid_actions.append(action)
                    mapping[str(action)] = str((xx, yy, w, d, h, z))
        return valid_actions, mapping

    def can_place_box(self, box_dims):
        acts, _ = self.get_valid_actions(box_dims)
        return len(acts) > 0

    # =========================================================
    # Heuristics
    # =========================================================

    def _get_contact_score(self, xx, yy, w, d, z):
        L, W = self.pallet_size
        perimeter_contact = 0

        if xx == 0:
            perimeter_contact += d
        else:
            neighbor_h = self.current_height_map[xx-1, yy:yy+d]
            perimeter_contact += np.sum(neighbor_h >= z)

        if xx + w == L:
            perimeter_contact += d
        else:
            neighbor_h = self.current_height_map[xx+w, yy:yy+d]
            perimeter_contact += np.sum(neighbor_h >= z)

        if yy == 0:
            perimeter_contact += w
        else:
            neighbor_h = self.current_height_map[xx:xx+w, yy-1]
            perimeter_contact += np.sum(neighbor_h >= z)

        if yy + d == W:
            perimeter_contact += w
        else:
            neighbor_h = self.current_height_map[xx:xx+w, yy+d]
            perimeter_contact += np.sum(neighbor_h >= z)

        return perimeter_contact

    def heuristic_stacking(self, valid_actions):
        best_action, best_score = None, np.inf

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)

            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                region = self.current_height_map[xx:xx+w, yy:yy+d]
                support_area = np.sum(region == z)
                support_ratio = support_area / (w * d)
                score = z - (support_ratio * 0.1)

                if score < best_score:
                    best_score, best_action = score, action
        return best_action

    def heuristic_best_fit(self, valid_actions):
        best_action, best_score = None, -np.inf

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)

            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                contact = self._get_contact_score(xx, yy, w, d, z)
                score = contact - (z * 0.01)

                if score > best_score:
                    best_score, best_action = score, action
        return best_action

    def heuristic_semi_perfect_fit(self, valid_actions):
        best_action, best_score = None, -np.inf

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)

            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                west_touch = (xx == 0) or np.any(self.current_height_map[xx-1, yy:yy+d] >= z)
                east_touch = (xx + w == self.pallet_size[0]) or np.any(self.current_height_map[xx+w, yy:yy+d] >= z)
                x_match = 1 if (west_touch and east_touch) else 0

                north_touch = (yy == 0) or np.any(self.current_height_map[xx:xx+w, yy-1] >= z)
                south_touch = (yy + d == self.pallet_size[1]) or np.any(self.current_height_map[xx:xx+w, yy+d] >= z)
                y_match = 1 if (north_touch and south_touch) else 0

                region = self.current_height_map[xx:xx+w, yy:yy+d]
                support_area = np.sum(region == z)
                waste = (w * d) - support_area
                waste_penalty = waste * 10

                score = (x_match * 100) + (y_match * 100) - waste_penalty - z

                if score > best_score:
                    best_score, best_action = score, action
        return best_action

    def heuristic_corner(self, valid_actions):
        best_action, best_score = None, -np.inf
        L, W = self.pallet_size

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)

            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                walls_hit = 0
                if xx == 0: walls_hit += 1
                if xx + w == L: walls_hit += 1
                if yy == 0: walls_hit += 1
                if yy + d == W: walls_hit += 1

                contact = self._get_contact_score(xx, yy, w, d, z)
                score = (walls_hit * 1000) + (contact * 10) - z

                if score > best_score:
                    best_score, best_action = score, action
        return best_action

    def heuristic_complex_fit(self, valid_actions):
        best_action, best_score = None, -np.inf
        L, W = self.pallet_size

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)

            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                gap_x = 0
                for ix in range(xx + w, L):
                    if np.any(self.current_height_map[ix, yy:yy+d] > z): break
                    gap_x += 1
                else: gap_x = L - (xx + w)

                gap_y = 0
                for iy in range(yy + d, W):
                    if np.any(self.current_height_map[xx:xx+w, iy] > z): break
                    gap_y += 1
                else: gap_y = W - (yy + d)

                full_gap_x = gap_x + w
                full_gap_y = gap_y + d

                residue_x = full_gap_x % w
                residue_y = full_gap_y % d

                score = (1.0 if residue_x == 0 else 0.0) * 1000 + \
                        (1.0 if residue_y == 0 else 0.0) * 1000

                contact = self._get_contact_score(xx, yy, w, d, z)
                score += (contact * 10) - z

                if score > best_score:
                    best_score, best_action = score, action
        return best_action

    def choose_action_by_heuristic(self, heuristic_name, pred_mask=None):
        valid_actions, mapping = self.get_valid_actions(self.current_box, pred_mask=pred_mask)
        if not valid_actions:
            self.invalid_actions_learned += 1
            return None, None

        if heuristic_name == 'stacking':
            action = self.heuristic_stacking(valid_actions)
        elif heuristic_name == 'best_fit':
            action = self.heuristic_best_fit(valid_actions)
        elif heuristic_name == 'semi_perfect_fit':
            action = self.heuristic_semi_perfect_fit(valid_actions)
        elif heuristic_name == 'corner':
            action = self.heuristic_corner(valid_actions)
        elif heuristic_name == 'complex_fit':
            action = self.heuristic_complex_fit(valid_actions)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic_name}")

        if action not in valid_actions:
            action = valid_actions[0]
        return action, mapping[str(action)]

    def step(self, action, free_buffer_slots=None, max_buffer_slots=None):
        rotation = action % 2
        remaining = action // 2
        yy = remaining % self.pallet_size[1]
        xx = remaining // self.pallet_size[1]

        w, d, h = self.get_rotated_box_dims(self.current_box, rotation)

        valid, base_z = self._is_valid_placement(xx, yy, w, d, h)
        if not valid:
            self.invalid_actions_attempted += 1
            reward = -0.5
            return self._get_state(), reward, False, {"invalid": True}

        old_flatness = self._calculate_maximal_flat_area(self.current_height_map)
        self._update_height_map(xx, yy, w, d, h, base_z)
        self.placed_boxes.append((xx, yy, w, d, h, base_z))
        new_flatness = self._calculate_maximal_flat_area(self.current_height_map)

        pallet_vol = self.pallet_size[0] * self.pallet_size[1] * self.max_height
        box_vol = w * d * h

        r_vol = box_vol / pallet_vol
        r_flat = (new_flatness - old_flatness) / (self.pallet_size[0] * self.pallet_size[1])

        reward = r_vol + (r_flat * 0.1)

        # Buffer occupancy bonus: reward having free buffer space
        if free_buffer_slots is not None and max_buffer_slots is not None and max_buffer_slots > 0:
            reward += 0.02 * (free_buffer_slots / max_buffer_slots)

        self.current_box = None
        done = False
        return self._get_state(), reward, done, {}

    def visualize_pallet(self, episode_num, boxes_attempted, utilization,
                         invalid_learned, invalid_attempted, output_dir=""):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, (xx, yy, w, d, h, base_z) in enumerate(self.placed_boxes):
            color = plt.cm.tab20(i % 20)
            ax.bar3d(xx, yy, base_z, w, d, h, color=color, edgecolor='black', alpha=0.8, linewidth=0.5)

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

        filename = os.path.join(output_dir, f'episode_{episode_num}_results.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()


# =====================================================
# DQN Agent v4: 6 actions (5 heuristics + buffer defer)
# =====================================================
class DQNAgent:
    def __init__(
        self,
        state_dims,
        action_size=6,
        max_height=10,
        learning_rate=0.0001,
        max_buffer_slots=4,
    ):
        self.state_dims = state_dims
        self.pallet_size = state_dims['height_map']
        self.action_size = action_size  # 6: 5 heuristics + 1 buffer defer
        self.max_height = max_height
        self.learning_rate = learning_rate
        self.max_buffer_slots = max_buffer_slots

        # Vector input size: 3 (box dims) + max_buffer_slots*3 (buffer box dims) + 6 (proxy scores)
        self.vec_input_size = 3 + max_buffer_slots * 3 + 6

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Buffer input: {max_buffer_slots} slots x 3 dims = {max_buffer_slots * 3} features (vec_input_size={self.vec_input_size})")
        print(f"Action space: {action_size} (5 heuristics + 1 buffer defer)")

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-8)

        self.memory = []
        self.memory_size = 50000
        self.batch_size = 64
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05

        self.mask_loss_weight = 0.15
        self.optimizer_step_count = 0

    def _normalize_buffer_boxes(self, buffer_boxes):
        """Normalize buffer box dimensions into a fixed-size vector."""
        L, W = self.state_dims['height_map']
        buf_dims = []
        if buffer_boxes is not None:
            for box in buffer_boxes[:self.max_buffer_slots]:
                box = np.asarray(box, dtype=np.float32)
                buf_dims.extend([box[0] / L, box[1] / W, box[2] / self.max_height])
        while len(buf_dims) < self.max_buffer_slots * 3:
            buf_dims.append(0.0)
        return buf_dims

    def _norm(self, state, buffer_boxes=None, proxy_scores=None):
        L, W = self.state_dims['height_map']

        hm = state['height_map'] / float(self.max_height)
        hm_tensor = torch.FloatTensor(hm).unsqueeze(0).unsqueeze(0).to(self.device)

        box_dims = state['box_dims']
        if np.sum(box_dims) > 0:
            bd_norm = [box_dims[0]/L, box_dims[1]/W, box_dims[2]/self.max_height]
        else:
            bd_norm = [0, 0, 0]

        buf_dims = self._normalize_buffer_boxes(buffer_boxes)

        if proxy_scores is None:
            scores = [0.0] * 6
        else:
            scores = proxy_scores[:6]

        vec = bd_norm + buf_dims + scores
        vec_tensor = torch.FloatTensor(vec).unsqueeze(0).to(self.device)

        return hm_tensor, vec_tensor

    def _build_model(self):
        vec_input_size = self.vec_input_size

        class ProNet(nn.Module):
            def __init__(self, pallet_size, action_size, vec_in_size):
                super().__init__()
                L, W = pallet_size
                self.L, self.W = L, W

                # Branch A: Spatial (Height Map only, 1 channel)
                self.spatial = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.Flatten()
                )
                spatial_out_size = 64 * L * W

                # Branch B: Vector (Box + Buffer Dims + Scores) - wider for richer buffer representation
                self.vector = nn.Sequential(
                    nn.Linear(vec_in_size, 128), nn.ReLU(),
                    nn.Linear(128, 128), nn.ReLU()
                )

                # Fusion
                fusion_size = spatial_out_size + 128

                # Critic Head (Value)
                self.val = nn.Sequential(
                    nn.Linear(fusion_size, 512), nn.ReLU(),
                    nn.Linear(512, 1)
                )

                # Actor Head (Advantage) - 6 actions
                self.adv = nn.Sequential(
                    nn.Linear(fusion_size, 512), nn.ReLU(),
                    nn.Linear(512, action_size)
                )

                # Mask Head (Auxiliary) - Branches from Spatial only
                self.mask_head = nn.Sequential(
                    nn.Linear(spatial_out_size, 512), nn.ReLU(),
                    nn.Linear(512, 2 * L * W)
                )

            def forward(self, hm_tensor, vec_tensor):
                s_feat = self.spatial(hm_tensor)
                v_feat = self.vector(vec_tensor)

                combined = torch.cat([s_feat, v_feat], dim=1)

                v = self.val(combined)
                a = self.adv(combined)
                q = v + (a - a.mean(dim=1, keepdim=True))

                mask_logits = self.mask_head(s_feat).view(-1, 2, self.L, self.W)

                return q, mask_logits

        return ProNet(self.state_dims['height_map'], self.action_size, vec_input_size)

    def remember(self, s, a, r, ns, d, buffer_boxes, proxy_scores):
        L, W = self.state_dims['height_map']
        hm_raw = s['height_map']
        bd_raw = s['box_dims']

        # Skip GT mask computation for buffer actions (action=5) - no spatial target
        if a == 5 or np.sum(bd_raw) == 0:
            gt_mask = np.zeros((2, L, W), dtype=np.float32)
        else:
            gt_mask = compute_gt_mask(hm_raw, bd_raw, pallet_size=(L, W), max_height=self.max_height)
            gt_mask = gt_mask.astype(np.float32)

        buf_boxes_copy = [np.array(b, dtype=np.float32) for b in buffer_boxes] if buffer_boxes else []

        self.memory.append((s, a, r, ns, d, buf_boxes_copy, gt_mask, proxy_scores))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    @torch.no_grad()
    def get_action_with_prior(self, state, proxy_scores, buffer_boxes=None):
        buf_count = len(buffer_boxes) if buffer_boxes else 0
        buffer_full = buf_count >= self.max_buffer_slots

        if np.random.rand() <= self.epsilon:
            if buffer_full:
                return int(np.random.randint(0, 5))  # Only heuristic actions 0-4
            return int(np.random.randint(0, self.action_size))

        hm_t, vec_t = self._norm(state, buffer_boxes, proxy_scores)
        q, _ = self.model(hm_t, vec_t)
        if buffer_full:
            q[0, 5] = -float('inf')  # Mask out buffer action
        return int(torch.argmax(q).item())

    @torch.no_grad()
    def predict_mask(self, state, buffer_boxes=None, threshold=0.5):
        hm_t, vec_t = self._norm(state, buffer_boxes)
        _, ml = self.model(hm_t, vec_t)
        return (torch.sigmoid(ml)[0].cpu().numpy() > threshold)

    @torch.no_grad()
    def get_mask_confidence_batch(self, states, buffer_boxes_list):
        hm_list = []
        vec_list = []
        L, W = self.pallet_size

        for i, state in enumerate(states):
            hm = state['height_map'] / float(self.max_height)
            hm_list.append(hm[np.newaxis, :, :])
            bd = state['box_dims']
            if np.sum(bd) > 0:
                bd_norm = [bd[0]/L, bd[1]/W, bd[2]/self.max_height]
            else:
                bd_norm = [0, 0, 0]
            buf_dims = self._normalize_buffer_boxes(buffer_boxes_list[i])
            sc = [0.0]*6
            vec_list.append(bd_norm + buf_dims + sc)

        hm_t = torch.FloatTensor(np.array(hm_list)).to(self.device)
        vec_t = torch.FloatTensor(np.array(vec_list)).to(self.device)
        _, ml = self.model(hm_t, vec_t)
        return torch.sigmoid(ml).cpu().numpy()

    @torch.no_grad()
    def get_action_with_prior_batch(self, states, proxy_scores_list, buffer_boxes_list):
        n = len(states)
        actions = []

        hm_list = []
        vec_list = []
        L, W = self.pallet_size

        # Pre-compute buffer-full flags
        buffer_full_flags = []
        for i in range(n):
            buf_count = len(buffer_boxes_list[i]) if buffer_boxes_list[i] else 0
            buffer_full_flags.append(buf_count >= self.max_buffer_slots)

        batch_indices = []
        for i in range(n):
            if np.random.rand() <= self.epsilon:
                if buffer_full_flags[i]:
                    actions.append(int(np.random.randint(0, 5)))  # Only heuristic actions
                else:
                    actions.append(int(np.random.randint(0, self.action_size)))
            else:
                actions.append(None)
                batch_indices.append(i)

        if len(batch_indices) > 0:
            for i in batch_indices:
                state = states[i]
                hm = state['height_map'] / float(self.max_height)
                hm_list.append(hm[np.newaxis, :, :])
                bd = state['box_dims']
                if np.sum(bd) > 0:
                    bd_norm = [bd[0]/L, bd[1]/W, bd[2]/self.max_height]
                else:
                    bd_norm = [0, 0, 0]
                buf_dims = self._normalize_buffer_boxes(buffer_boxes_list[i])
                sc = proxy_scores_list[i]
                vec_list.append(bd_norm + buf_dims + sc)

            hm_t = torch.FloatTensor(np.array(hm_list)).to(self.device)
            vec_t = torch.FloatTensor(np.array(vec_list)).to(self.device)
            q, _ = self.model(hm_t, vec_t)

            # Mask out buffer action for envs with full buffers
            for local_idx, global_idx in enumerate(batch_indices):
                if buffer_full_flags[global_idx]:
                    q[local_idx, 5] = -float('inf')

            batch_actions = torch.argmax(q, dim=1).cpu().numpy()

            for idx, action in zip(batch_indices, batch_actions):
                actions[idx] = int(action)

        return actions

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, buffer_boxes_list, gt_masks_list, proxy_scores_list = zip(*batch)

        L, W = self.state_dims['height_map']

        def prepare_batch(s_list, buf_boxes_list, scores_list):
            hm_list = []
            vec_list = []
            for i, s in enumerate(s_list):
                hm = s['height_map'] / float(self.max_height)
                hm_list.append(hm[np.newaxis, :, :])

                bd = s['box_dims']
                if np.sum(bd) > 0:
                    bd_norm = [bd[0]/L, bd[1]/W, bd[2]/self.max_height]
                else:
                    bd_norm = [0, 0, 0]

                buf_dims = self._normalize_buffer_boxes(buf_boxes_list[i] if buf_boxes_list is not None else None)

                if scores_list is None:
                    sc = [0.0]*6
                else:
                    sc = scores_list[i]

                vec_list.append(bd_norm + buf_dims + sc)

            hm_tensor = torch.FloatTensor(np.array(hm_list)).to(self.device)
            vec_tensor = torch.FloatTensor(np.array(vec_list)).to(self.device)
            return hm_tensor, vec_tensor

        # 1. Current State
        hm_curr, vec_curr = prepare_batch(states, buffer_boxes_list, proxy_scores_list)
        gt_masks = torch.from_numpy(np.array(gt_masks_list, dtype=np.float32)).to(self.device)

        q_cur, mask_logits = self.model(hm_curr, vec_curr)
        q_cur = q_cur.gather(1, torch.LongTensor(actions).to(self.device).unsqueeze(1)).squeeze(1)

        # 2. Next State
        with torch.no_grad():
             hm_next, vec_next_dummy = prepare_batch(next_states, buffer_boxes_list, None)
             _, mask_logits_next = self.target_model(hm_next, vec_next_dummy)
             mask_probs_next = torch.sigmoid(mask_logits_next).cpu().numpy()

             next_scores_list = []
             for i in range(len(next_states)):
                 ns = next_states[i]
                 sc = proxy_scores_for_heuristics(
                     ns['height_map'], ns['box_dims'],
                     (L, W), self.max_height,
                     mask_probs_next[i],
                     buffer_boxes=buffer_boxes_list[i],
                     max_buffer_slots=self.max_buffer_slots
                 )
                 next_scores_list.append(sc)

             hm_next, vec_next = prepare_batch(next_states, buffer_boxes_list, next_scores_list)
             q_next, _ = self.target_model(hm_next, vec_next)
             q_next, _ = q_next.max(1)

        dones_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        tgt = rewards_t + self.gamma * q_next * (1 - dones_t)

        q_loss = nn.SmoothL1Loss()(q_cur, tgt)

        # Mask loss: exclude buffer-action samples (action=5) since they have no spatial target
        actions_t = torch.LongTensor(actions).to(self.device)
        has_box = torch.tensor([1.0 if np.sum(s['box_dims']) > 0 else 0.0 for s in states]).to(self.device)
        not_buffer_action = (actions_t != 5).float()
        mask_weight = has_box * not_buffer_action

        per_pixel_loss = nn.BCEWithLogitsLoss(reduction='none')(mask_logits, gt_masks)
        per_sample_loss = per_pixel_loss.mean(dim=(1,2,3))

        mask_loss = (per_sample_loss * mask_weight).sum() / (mask_weight.sum() + 1e-6)

        loss = q_loss + self.mask_loss_weight * mask_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer_step_count += 1

        self.last_q_loss = q_loss.item()
        self.last_mask_loss = mask_loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'max_buffer_slots': self.max_buffer_slots,
            'memory': self.memory
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.target_model.load_state_dict(self.model.state_dict())


def proxy_scores_for_heuristics(height_map, box_dims, pallet_size, max_height, pred_mask_probs,
                                 buffer_boxes=None, max_buffer_slots=4):
    """
    Returns 6 proxy scores: 5 heuristic scores + 1 buffer defer score.
    """
    scores = [0.0] * 6

    if np.sum(box_dims) == 0: return scores

    Lp, Wp = pallet_size

    def get_rotated(bd, rot):
        if rot == 0: return int(bd[0]), int(bd[1]), int(bd[2])
        return int(bd[1]), int(bd[0]), int(bd[2])

    rot_dims = [get_rotated(box_dims, 0), get_rotated(box_dims, 1)]

    best_stack = (-float('inf'), 0.0)
    best_bf = (-float('inf'), 0.0)
    best_semi = (-float('inf'), 0.0)
    best_corn = (-float('inf'), 0.0)
    best_comp = (-float('inf'), 0.0)

    found_valid = False

    for rot in [0]:
        mask_probs = pred_mask_probs[rot]
        valid_indices = np.argwhere(mask_probs > 0.01)

        w, d, h = rot_dims[rot]

        valid_mask = (valid_indices[:, 0] <= Lp - w) & (valid_indices[:, 1] <= Wp - d)
        candidates = valid_indices[valid_mask]

        if len(candidates) > 0: found_valid = True

        for i in range(len(candidates)):
            xx, yy = candidates[i]

            maximal_space_score = mask_probs[xx, yy]

            region = height_map[xx:xx + w, yy:yy + d]
            base_z = np.max(region)

            if base_z + h > max_height: continue

            gap = max_height - (base_z + h)

            m_stack = -base_z
            if m_stack > best_stack[0]:
                best_stack = (m_stack, maximal_space_score)
            elif m_stack == best_stack[0]:
                best_stack = (m_stack, max(best_stack[1], maximal_space_score))

            m_bf = -gap
            if m_bf > best_bf[0]:
                best_bf = (m_bf, maximal_space_score)
            elif m_bf == best_bf[0]:
                best_bf = (m_bf, max(best_bf[1], maximal_space_score))

            support_count = np.sum(region == base_z)
            waste = (w * d) - support_count
            m_semi = -(waste + gap)
            if m_semi > best_semi[0]:
                best_semi = (m_semi, maximal_space_score)
            elif m_semi == best_semi[0]:
                best_semi = (m_semi, max(best_semi[1], maximal_space_score))

            walls = 0
            if xx == 0: walls += 1
            if xx+w == Lp: walls += 1
            if yy == 0: walls += 1
            if yy+d == Wp: walls += 1
            m_corn = walls
            if m_corn > best_corn[0]:
                best_corn = (m_corn, maximal_space_score)
            elif m_corn == best_corn[0]:
                best_corn = (m_corn, max(best_corn[1], maximal_space_score))

            gap_x = 0
            for ix in range(xx + w, Lp):
                if np.any(height_map[ix, yy:yy+d] > base_z): break
                gap_x += 1
            else: gap_x = Lp - (xx + w)
            gap_y = 0
            for iy in range(yy + d, Wp):
                if np.any(height_map[xx:xx+w, iy] > base_z): break
                gap_y += 1
            else: gap_y = Wp - (yy + d)

            residue_x = (gap_x + w) % w
            residue_y = (gap_y + d) % d
            m_comp = (1.0 if residue_x == 0 else 0.0) + (1.0 if residue_y == 0 else 0.0)
            if m_comp > best_comp[0]:
                best_comp = (m_comp, maximal_space_score)
            elif m_comp == best_comp[0]:
                best_comp = (m_comp, max(best_comp[1], maximal_space_score))

    if not found_valid:
        # No valid placements: buffer score should be high (if buffer has space)
        buf_count = len(buffer_boxes) if buffer_boxes else 0
        buffer_fill_ratio = buf_count / max_buffer_slots if max_buffer_slots > 0 else 1.0
        scores[5] = 1.0 * (1.0 - buffer_fill_ratio)
        return scores

    scores[0] = best_stack[1]
    scores[1] = best_bf[1]
    scores[2] = best_semi[1]
    scores[3] = best_corn[1]
    scores[4] = best_comp[1]

    # Buffer defer score: sigmoid-based, sharp transition around quality threshold
    best_heuristic_score = max(scores[0:5])
    buf_count = len(buffer_boxes) if buffer_boxes else 0
    buffer_fill_ratio = buf_count / max_buffer_slots if max_buffer_slots > 0 else 1.0
    buffer_capacity = 1.0 - buffer_fill_ratio

    DEFER_THRESHOLD = 0.4
    STEEPNESS = 8.0
    quality_deficit = DEFER_THRESHOLD - best_heuristic_score
    defer_signal = 1.0 / (1.0 + np.exp(-STEEPNESS * quality_deficit))

    scores[5] = defer_signal * buffer_capacity

    return scores
