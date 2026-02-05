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
    """
    Returns a continuous mask of shape (2, L, W), one channel per rotation:
      rot 0: (l, w, h)
      rot 1: (h, w, l)
    
    Values are normalized flatness scores (0 to 1):
      - 0 = invalid placement
      - 0.5 to 1.0 = valid, scaled by relative flatness quality
    """
    L, W = pallet_size
    mask = np.zeros((2, L, W), dtype=np.float32)
    rotations = [(0, 1, 2), (2, 1, 0)]
    
    # First pass: find all valid positions and their flatness scores
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
                # 60% support requirement
                support_count = np.sum(region == base_z)
                if (support_count / (l * w)) < 0.60:
                    continue
                
                # Compute flatness score for this placement
                temp_hm = height_map.copy()
                temp_hm[x:x+l, y:y+w] = base_z + h
                flatness = calculate_average_flat_area(temp_hm)
                valid_positions.append((r_idx, x, y, flatness))

    # Second pass: normalize scores and populate mask
    if len(valid_positions) > 0:
        scores = np.array([pos[3] for pos in valid_positions])
        min_score = scores.min()
        max_score = scores.max()
        
        for r_idx, x, y, flatness in valid_positions:
            if max_score - min_score > 1e-6:
                # Normalize to [0.5, 1.0] range
                normalized = 0.5 + 0.5 * (flatness - min_score) / (max_score - min_score)
            else:
                normalized = 1.0
            mask[r_idx, x, y] = normalized

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

        # metrics
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
        valid_rotations = [(0, 1, 2), (2, 1, 0)]  # (L,W,H) and (H,W,L)
        return tuple(int(box[i]) for i in valid_rotations[rotation])

    def _is_valid_placement(self, x, y, w, d, h):
        L, W, H = self.pallet_size[0], self.pallet_size[1], self.max_height
        if (x + w > L) or (y + d > W):
            return False, -1
        
        # Standard Height Map Logic: Gravity check
        region = self.current_height_map[x:x+w, y:y+d]
        base_z = np.max(region)
        
        if base_z + h > H:
            return False, -1
            
        # 60% support requirement
        support_count = np.sum(region == base_z)
        if (support_count / (w * d)) < 0.60:
             return False, -1
             
        return True, base_z

    def _update_height_map(self, x, y, w, d, h, z):
        # Update Height Map
        self.current_height_map[x:x+w, y:y+d] = z + h
        return z

    # unchanged “maximal flat area”; reward still based on flatness.
    # You can redesign reward later to credit bridging more directly.
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
        """Uses predicted mask to prune, but final validity uses BRIDGING rule."""
        valid_actions = []
        mapping = {}
        L, W = self.pallet_size

        for rotation in range(2):
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
    # Corrected Heuristics (Paper-Aligned + Refined Corner)
    # =========================================================

    def _get_contact_score(self, xx, yy, w, d, z):
        """
        Helper: Calculates how well the box 'fits' horizontally by checking 
        contact with neighbors or walls. 
        """
        L, W = self.pallet_size
        perimeter_contact = 0
        
        # Check West Wall or Neighbor
        if xx == 0: 
            perimeter_contact += d # Wall contact
        else:
            # Check if neighbor height is >= current z (meaning we are touching it)
            neighbor_h = self.current_height_map[xx-1, yy:yy+d]
            perimeter_contact += np.sum(neighbor_h >= z)

        # Check East Wall or Neighbor
        if xx + w == L: 
            perimeter_contact += d
        else:
            neighbor_h = self.current_height_map[xx+w, yy:yy+d]
            perimeter_contact += np.sum(neighbor_h >= z)

        # Check North Wall or Neighbor
        if yy == 0: 
            perimeter_contact += w
        else:
            neighbor_h = self.current_height_map[xx:xx+w, yy-1]
            perimeter_contact += np.sum(neighbor_h >= z)

        # Check South Wall or Neighbor
        if yy + d == W: 
            perimeter_contact += w
        else:
            neighbor_h = self.current_height_map[xx:xx+w, yy+d]
            perimeter_contact += np.sum(neighbor_h >= z)
            
        return perimeter_contact

    def heuristic_stacking(self, valid_actions):
        """
        [cite_start]Paper: "Prioritizes vertical space utilization by packing items in columns"[cite: 227].
        Logic: Find the LOWEST Z. If Z is equal, prefer the spot that maintains the 'column'
        (highest support ratio), strictly penalizing overhangs to prevent towers from falling.
        """
        best_action, best_score = None, np.inf # Lower score is better

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            
            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                # Calculate support ratio (area supported / box area)
                region = self.current_height_map[xx:xx+w, yy:yy+d]
                support_area = np.sum(region == z)
                support_ratio = support_area / (w * d)

                # Score:
                # 1. Primary: Minimize Z (Build from bottom up)
                # 2. Secondary: Maximize Support (Paper implies stable columns)
                # We subtract support_ratio so "1.0 support" reduces the score (improves it).
                score = z - (support_ratio * 0.1) 
                
                if score < best_score:
                    best_score, best_action = score, action
        return best_action

    def heuristic_best_fit(self, valid_actions):
        """
        [cite_start]Paper: "Seeks to fill spaces by selecting the smallest maximal space available"[cite: 252].
        Logic: Minimize the 'gap' between the box and its surroundings.
        In a Height Map, this means maximizing the contact perimeter (touching neighbors).
        """
        best_action, best_score = None, -np.inf # Higher score is better

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            
            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                # 1. Contact Score: Proxy for "Smallest Maximal Space". 
                # High contact = Small gap between box and neighbors.
                contact = self._get_contact_score(xx, yy, w, d, z)
                
                # 2. Tie-breaker: Place as low as possible (Gravity)
                # We subtract Z * 0.01 so that among equal contact scores, lower Z wins.
                score = contact - (z * 0.01)
                
                if score > best_score:
                    best_score, best_action = score, action
        return best_action

    def heuristic_semi_perfect_fit(self, valid_actions):
        """
        Paper: "Minimizes wasted space... scenarios where the box perfectly fits in 
        [cite_start]three, two, or one dimension(s)"[cite: 258, 259].
        Logic: Check for exact dimension matches (filling a hole X or Y).
        """
        best_action, best_score = None, -np.inf # Higher score is better

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            
            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                # Dimension Matches
                # Check X-axis match (touching both West and East?)
                west_touch = (xx == 0) or np.any(self.current_height_map[xx-1, yy:yy+d] >= z)
                east_touch = (xx + w == self.pallet_size[0]) or np.any(self.current_height_map[xx+w, yy:yy+d] >= z)
                x_match = 1 if (west_touch and east_touch) else 0

                # Check Y-axis match (touching both North and South?)
                north_touch = (yy == 0) or np.any(self.current_height_map[xx:xx+w, yy-1] >= z)
                south_touch = (yy + d == self.pallet_size[1]) or np.any(self.current_height_map[xx:xx+w, yy+d] >= z)
                y_match = 1 if (north_touch and south_touch) else 0
                
                # Check Z-axis match / Waste
                # Waste = Empty space BELOW the box (lack of support).
                # The paper assumes perfect fit minimizes wasted space.
                region = self.current_height_map[xx:xx+w, yy:yy+d]
                support_area = np.sum(region == z)
                waste = (w * d) - support_area
                waste_penalty = waste * 10  # Heavily penalize holes below
                
                # Score: Perfect 2D fit > 1D fit > 0D fit.
                score = (x_match * 100) + (y_match * 100) - waste_penalty - z
                
                if score > best_score:
                    best_score, best_action = score, action
        return best_action

    def heuristic_random_fit(self, valid_actions):
        """
        [cite_start]Paper: "Initially 66.67% stacking... by end, semi-perfect fit rises to 66.67%"[cite: 311, 313].
        [cite_start]"The best shifting point for Cut-1 and RS is 10%"[cite: 318, 319].
        """
        if not valid_actions:
            return None

        # Calculate Utilization
        pallet_vol = self.pallet_size[0] * self.pallet_size[1] * self.max_height
        placed_vol = sum(b[2]*b[3]*b[4] for b in self.placed_boxes)
        utilization = placed_vol / pallet_vol if pallet_vol > 0 else 0
        
        # Paper threshold logic
        threshold = 0.10 
        
        if utilization < threshold:
            # Early game: 66% Stacking, 33% Semi-Perfect
            p_stack = 0.66
        else:
            # Late game: 33% Stacking, 66% Semi-Perfect
            p_stack = 0.33
            
        if np.random.rand() < p_stack:
            return self.heuristic_stacking(valid_actions)
        else:
            return self.heuristic_semi_perfect_fit(valid_actions)

    def heuristic_corner(self, valid_actions):
        """
        Logic: Pack from Outside -> In.
        Prioritizes:
        1. Touching 2+ Walls/Boundaries (True Corners)
        2. Touching 1 Wall/Boundary
        3. Low Z (Gravity)
        """
        best_action, best_score = None, -np.inf # Higher score is better
        L, W = self.pallet_size

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            
            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                # Count Walls Touched (Outside-In Strategy)
                walls_hit = 0
                if xx == 0: walls_hit += 1
                if xx + w == L: walls_hit += 1
                if yy == 0: walls_hit += 1
                if yy + d == W: walls_hit += 1
                
                # Count Neighbor Contact (Tying it together)
                contact = self._get_contact_score(xx, yy, w, d, z)
                
                # Score Calculation:
                # Priority 1: Walls (1000 points per wall) -> Forces outside placement
                # Priority 2: Contact (10 points per unit) -> Forces tight packing
                # Priority 3: Minimize Z (negative z) -> Gravity
                score = (walls_hit * 1000) + (contact * 10) - z
                
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
        elif heuristic_name == 'random_fit':
            action = self.heuristic_random_fit(valid_actions)
        elif heuristic_name == 'corner':
            action = self.heuristic_corner(valid_actions)
        else:
            raise ValueError("Unknown heuristic")

        if action not in valid_actions:
            action = random.choice(valid_actions)
        return action, mapping[str(action)]

    def step(self, action):
        rotation = action % 2
        remaining = action // 2
        yy = remaining % self.pallet_size[1]
        xx = remaining // self.pallet_size[1]

        w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
        
        valid, base_z = self._is_valid_placement(xx, yy, w, d, h)
        if not valid:
            self.invalid_actions_attempted += 1
            reward = -5
            return self._get_state(), reward, False, {"invalid": True}

        self._update_height_map(xx, yy, w, d, h, base_z)
        self.placed_boxes.append((xx, yy, w, d, h, base_z))

        # Reward: Volume Utilization
        # (Box Volume / Pallet Volume) * Scale
        pallet_vol = self.pallet_size[0] * self.pallet_size[1] * self.max_height
        box_vol = w * d * h
        reward = (box_vol / pallet_vol) * 10 # Scaling factor
        
        self.current_box = None
        done = False # controlled by stream
        return self._get_state(), reward, done, {}

    def visualize_pallet(self, episode_num, boxes_attempted, utilization,
                         invalid_learned, invalid_attempted, output_dir=""):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Individual Box Visualization
        for i, (xx, yy, w, d, h, base_z) in enumerate(self.placed_boxes):
            # Unique color per box using a colormap
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
# DQN Agent (dueling + mask head, Option-A soft bias)
# =====================================================
class DQNAgent:
    def __init__(
        self,
        state_dims,
        action_size=5,
        max_height=10,
        learning_rate=0.001,
        # Unused params kept for compatibility if needed
        min_support_ratio=0.60,
        require_opposite_edge_support=True,
    ):
        self.state_dims = state_dims
        self.action_size = action_size
        self.max_height = max_height
        self.learning_rate = learning_rate

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

        # FIX 1: Large Memory for Epoch Training
        self.memory = []
        self.memory_size = 50000 
        self.batch_size = 32
        self.gamma = 0.95
        
        # Epsilon managed externally in training loop, but defaults here
        self.epsilon = 1.0 
        self.epsilon_min = 0.05
        
        self.mask_loss_weight = 0.15
        self.optimizer_step_count = 0

    # ------------------------------------------------------------------
    # FIX 2: SPATIAL STACKING (The "Pro" Input Method)
    # ------------------------------------------------------------------
    def _norm(self, state, buffer_count=0):
        L, W = self.state_dims['height_map']
        
        # 1. Height Map (Channel 0)
        # Normalize to [0, 1]
        hm = state['height_map'] / float(self.max_height)
        
        # 2. Box Dimensions Stretched (Channels 1, 2, 3)
        # We "stretch" the 3 scalars to cover the whole 10x10 grid.
        # This allows the Conv layers to do pixel-wise comparisons.
        box_dims = state['box_dims']
        box_tensor = np.zeros((3, L, W), dtype=np.float32)
        
        if np.sum(box_dims) > 0:
            box_tensor[0, :, :] = box_dims[0] / L           # Normalized Length
            box_tensor[1, :, :] = box_dims[1] / W           # Normalized Width
            box_tensor[2, :, :] = box_dims[2] / self.max_height # Normalized Height
        
        # 3. Stack -> Shape (1, 4, 10, 10)
        # Channel 0: Height Map
        # Channel 1: Box Length (every pixel is l_norm)
        # Channel 2: Box Width (every pixel is w_norm)
        # Channel 3: Box Height (every pixel is h_norm)
        combined_state = np.concatenate([hm[np.newaxis, :, :], box_tensor], axis=0)
        
        # 4. Buffer feature (Scalar)
        buf = np.array([min(buffer_count, 10) / 10.0], dtype=np.float32)
        
        return (torch.FloatTensor(combined_state).unsqueeze(0).to(self.device),
                None, # Box dims are now inside the conv input!
                torch.FloatTensor(buf).unsqueeze(0).to(self.device))

    def _build_model(self):
        class ProNet(nn.Module):
            def __init__(self, pallet_size, action_size):
                super().__init__()
                L, W = pallet_size
                self.L, self.W = L, W
                
                # FIX 3: 4 Input Channels, Stride=1 (No compression)
                self.conv = nn.Sequential(
                    nn.Conv2d(4, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.Flatten()
                )
                
                # Output is 64 * 10 * 10 = 6400 features
                conv_out_size = 64 * L * W 
                
                # Critic Head (Value)
                self.val = nn.Sequential(
                    nn.Linear(conv_out_size + 1, 512), nn.ReLU(), 
                    nn.Linear(512, 1)
                )
                
                # Actor Head (Advantage)
                self.adv = nn.Sequential(
                    nn.Linear(conv_out_size + 1, 512), nn.ReLU(), 
                    nn.Linear(512, action_size)
                )
                
                # Mask Head (Auxiliary)
                self.mask_head = nn.Sequential(
                    nn.Linear(conv_out_size + 1, 512), nn.ReLU(),
                    nn.Linear(512, 2 * L * W)
                )

            def forward(self, combined_state, _, buf):
                f = self.conv(combined_state)
                # Inject buffer status into dense layers
                z = torch.cat([f, buf], dim=1)
                
                v = self.val(z)
                a = self.adv(z)
                q = v + (a - a.mean(dim=1, keepdim=True))
                mask_logits = self.mask_head(z).view(-1, 2, self.L, self.W)
                return q, mask_logits

        return ProNet(self.state_dims['height_map'], self.action_size)

    # --- Standard Methods (Updated for new input shape) ---

    def remember(self, s, a, r, ns, d, buffer_count):
        # We re-compute mask here or passed in. 
        # Assuming your old code logic for mask computation is available.
        L, W = self.state_dims['height_map']
        hm_raw = s['height_map']
        bd_raw = s['box_dims']
        
        if np.sum(bd_raw) > 0:
            gt_mask = compute_gt_mask(hm_raw, bd_raw, pallet_size=(L, W), max_height=self.max_height)
            gt_mask = gt_mask.astype(np.float32)
        else:
            gt_mask = np.zeros((2, L, W), dtype=np.float32)
            
        self.memory.append((s, a, r, ns, d, buffer_count, gt_mask))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    @torch.no_grad()
    def act_with_mask_bias(self, state, buffer_count=0, mask_bias=None, beta=0.5, eps=1e-6):
        if np.random.rand() <= self.epsilon:
            return int(np.random.randint(0, self.action_size))
        
        combined, _, buf = self._norm(state, buffer_count)
        q, _ = self.model(combined, None, buf) # Pass None for box_dims
        q = q.squeeze(0)
        
        if mask_bias is not None:
            b = torch.tensor(mask_bias, dtype=torch.float32).to(self.device)
            # Normalize bias locally
            if b.std() > 1e-6:
                b = (b - b.mean()) / (b.std())
            q = q + beta * b
            
        return int(torch.argmax(q).item())

    @torch.no_grad()
    def predict_mask(self, state, buffer_count=0, threshold=0.5):
        combined, _, buf = self._norm(state, buffer_count)
        _, ml = self.model(combined, None, buf)
        return (torch.sigmoid(ml)[0].cpu().numpy() > threshold)
    
    @torch.no_grad()
    def get_mask_confidence(self, state, buffer_count=0):
        """
        Returns the raw continuous scores (0.0 to 1.0) from the mask head.
        Used for the 'Maximal Space Passthrough' investigation.
        """
        combined, _, buf = self._norm(state, buffer_count)
        # Pass None for box_dims as they are embedded in 'combined' for ProNet
        _, ml = self.model(combined, None, buf)
        return torch.sigmoid(ml)[0].cpu().numpy()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, buffer_counts, gt_masks_list = zip(*batch)

        L, W = self.state_dims['height_map']
        
        # Helper to batch process the "Spatial Stacking"
        def process_batch_states(s_list):
            processed = []
            for s in s_list:
                hm = s['height_map'] / float(self.max_height)
                bd = s['box_dims']
                bt = np.zeros((3, L, W), dtype=np.float32)
                if np.sum(bd) > 0:
                    bt[0] = bd[0]/L
                    bt[1] = bd[1]/W
                    bt[2] = bd[2]/self.max_height
                combined = np.concatenate([hm[np.newaxis, :, :], bt], axis=0)
                processed.append(combined)
            return np.array(processed, dtype=np.float32)

        s_batch = torch.FloatTensor(process_batch_states(states)).to(self.device)
        ns_batch = torch.FloatTensor(process_batch_states(next_states)).to(self.device)
        
        bc_batch = torch.FloatTensor(np.array(buffer_counts)).unsqueeze(1).to(self.device)
        bc_batch = torch.clamp(bc_batch, 0, 10) / 10.0
        
        gt_masks = torch.from_numpy(np.array(gt_masks_list, dtype=np.float32)).to(self.device)
        
        # Forward pass (Note: None for box_dims arg)
        q_cur, mask_logits = self.model(s_batch, None, bc_batch)
        q_cur = q_cur.gather(1, torch.LongTensor(actions).to(self.device).unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            q_next, _ = self.target_model(ns_batch, None, bc_batch)
            q_next, _ = q_next.max(1)
            
        dones_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        tgt = rewards_t + self.gamma * q_next * (1 - dones_t)
        
        q_loss = nn.SmoothL1Loss()(q_cur, tgt)
        
        # Mask Loss (Only for valid boxes)
        # Check channel 1 (Box Length) > 0
        has_box = (s_batch[:, 1, 0, 0] > 0).float()
        
        mask_preds = torch.sigmoid(mask_logits)
        # Mean over pixels (dim 1,2,3 -> channels, h, w)
        per_sample_loss = ((mask_preds - gt_masks) ** 2).mean(dim=(1,2,3))
        mask_loss = (per_sample_loss * has_box).sum() / (has_box.sum() + 1e-6)
        
        loss = q_loss + self.mask_loss_weight * mask_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer_step_count += 1 # Important for target update

        self.last_q_loss = q_loss.item()
        self.last_mask_loss = mask_loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.target_model.load_state_dict(self.model.state_dict())



def proxy_scores_for_heuristics(env_obj, pred_mask_probs):
    """
    Directly maps the Heuristic's chosen position to the Mask's predicted score.
    Since the Mask is trained on 'Maximal Spaces', this score IS the Maximal Space quality.
    """
    # Index: 0=Stacking, 1=BestFit, 2=SemiPerfect, 3=Random, 4=Corner
    scores = [0.0] * 5 
    
    box = env_obj.current_box
    if box is None: return scores

    Lp, Wp = env_obj.pallet_size
    max_h = env_obj.max_height
    hm = env_obj.current_height_map
    rot_dims = [env_obj.get_rotated_box_dims(box, 0), env_obj.get_rotated_box_dims(box, 1)]

    # We want to find the specific (x,y) that each heuristic would choose,
    # then grab the mask_probability at that exact coordinate.
    
    # Init with (Metric, MaskScore)
    # Stacking Metric: -Z
    best_stack = (-float('inf'), 0.0)
    # BestFit Metric: -Gap
    best_bf = (-float('inf'), 0.0)
    # SemiPerfect Metric: -Waste
    best_semi = (-float('inf'), 0.0)
    # Corner Metric: Walls
    best_corn = (-float('inf'), 0.0)

    found_valid = False

    for rot in [0, 1]:
        # Optimization: Only look at spots the network thinks are plausible (>1%)
        mask_probs = pred_mask_probs[rot]
        valid_indices = np.argwhere(mask_probs > 0.01)
        
        w, d, h = rot_dims[rot]
        
        valid_mask = (valid_indices[:, 0] <= Lp - w) & (valid_indices[:, 1] <= Wp - d)
        candidates = valid_indices[valid_mask]
        
        if len(candidates) > 0: found_valid = True
        
        for i in range(len(candidates)):
            xx, yy = candidates[i]
            
            # THE CORE: This value represents "Maximal Space Quality" (from your GT definition)
            maximal_space_score = mask_probs[xx, yy]
            
            # --- Re-run Heuristic Logic to identify the chosen spot ---
            region = hm[xx:xx + w, yy:yy + d]
            support_z = np.max(region)
            gap = max_h - (support_z + h)
            
            # 1. Stacking (Maximize -Z)
            m_stack = -support_z
            if m_stack > best_stack[0]:
                best_stack = (m_stack, maximal_space_score)
            elif m_stack == best_stack[0]:
                # Tie-breaker: If heights are equal, prefer the better Maximal Space
                best_stack = (m_stack, max(best_stack[1], maximal_space_score))

            # 2. Best Fit (Maximize -Gap)
            m_bf = -gap
            if m_bf > best_bf[0]:
                best_bf = (m_bf, maximal_space_score)
            elif m_bf == best_bf[0]:
                best_bf = (m_bf, max(best_bf[1], maximal_space_score))

            # 3. Semi-Perfect (Maximize -Waste)
            support_count = np.sum(region == support_z)
            waste = (w * d) - support_count
            m_semi = -(waste + gap)
            if m_semi > best_semi[0]:
                best_semi = (m_semi, maximal_space_score)
            elif m_semi == best_semi[0]:
                best_semi = (m_semi, max(best_semi[1], maximal_space_score))

            # 4. Corner (Maximize Walls)
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

    if not found_valid: return scores

    # Return the Maximal Space Score associated with each heuristic's choice
    scores[0] = best_stack[1]
    scores[1] = best_bf[1]
    scores[2] = best_semi[1]
    scores[3] = 0.5 
    scores[4] = best_corn[1]
    
    return scores
# ===========================================
# Training loop (Option-A soft-bias unchanged)
# ===========================================
def train(
    episodes_boxes,
    output_dir,
    model_save_path=None,
    verbose=False,
    episode_to_show=100,
    defer_penalty=-5.0,
    mask_bias_beta=0.5,
    learning_rate=0.001,
    max_buffer_size=float('inf'),
    min_support_ratio=0.50,
    require_opposite_edge_support=True,
    return_agent=False,
):
    env = BoxPilingEnv()
    agent = DQNAgent(
        state_dims={'height_map': env.pallet_size, 'box_dims': 3},
        action_size=5,
        max_height=env.max_height,
        learning_rate=learning_rate,
        min_support_ratio=min_support_ratio,
        require_opposite_edge_support=require_opposite_edge_support,
    )

    total_episodes = len(episodes_boxes)
    total_utilization = 0.0
    all_metrics = []
    heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'random_fit', 4: 'corner'}

    # Use tqdm for progress bar
    pbar = tqdm(range(total_episodes), desc="Training", unit="ep")
    for episode in pbar:
        state = env.reset()
        done = False
        boxes = episodes_boxes[episode]
        box_idx = 0

        episode_reward = 0
        episode_heuristic_counts = {k: 0 for k in heuristic_map.values()}
        total_decisions = 0
        placements_this_episode = 0
        attempts_this_episode = 0
        buffer = []
        
        # Track losses per episode
        episode_q_losses = []
        episode_mask_losses = []

        def try_place_one_box(box_dims, current_buffer_size):
            nonlocal state, episode_reward, total_decisions, placements_this_episode, attempts_this_episode, episode_q_losses, episode_mask_losses
            state = env.new_box_arrival(box_dims)
            attempts_this_episode += 1

            pred_mask = agent.predict_mask(state, buffer_count=current_buffer_size)
            mask_bias = proxy_scores_for_heuristics(env, pred_mask)
            h_idx = agent.act_with_mask_bias(state, buffer_count=current_buffer_size, mask_bias=mask_bias, beta=mask_bias_beta)
            heuristic = heuristic_map[h_idx]
            episode_heuristic_counts[heuristic] += 1
            total_decisions += 1

            action, mapping = env.choose_action_by_heuristic(heuristic, pred_mask=pred_mask)
            if action is None:
                action, mapping = env.choose_action_by_heuristic(heuristic, pred_mask=None)
                if action is None:
                    agent.remember(state, h_idx, defer_penalty, state, False, current_buffer_size)
                    agent.replay()
                    
                    # Store step losses
                    if hasattr(agent, 'last_q_loss'): episode_q_losses.append(agent.last_q_loss)
                    if hasattr(agent, 'last_mask_loss'): episode_mask_losses.append(agent.last_mask_loss)
                    
                    # Target network updated once per episode, not here
                    return False, defer_penalty

            if verbose and episode == episode_to_show:
                print(env.current_height_map)
                print(f"Heuristic: {heuristic} | Action: {mapping}")

            next_state, reward, local_done, info = env.step(action)
            agent.remember(state, h_idx, reward, next_state, local_done, current_buffer_size)
            agent.replay()
            
            # Store step losses
            if hasattr(agent, 'last_q_loss'): episode_q_losses.append(agent.last_q_loss)
            if hasattr(agent, 'last_mask_loss'): episode_mask_losses.append(agent.last_mask_loss)
            
            # Target network updated once per episode, not here
            state = next_state
            episode_reward += reward
            placements_this_episode += 1
            return True, reward

        while not done and box_idx < len(boxes):
            box_dims = boxes[box_idx]
            box_idx += 1
            placed, _ = try_place_one_box(box_dims, len(buffer))
            if not placed:
                if len(buffer) < max_buffer_size:
                    buffer.append(box_dims)
                else:
                    # Cannot place AND cannot buffer → end episode
                    done = True
                    break
            if env._is_terminal():
                done = True
                break

        # buffer passes
        while not done and len(buffer) > 0:
            made_progress = False
            new_buffer = []
            for box_dims in buffer:
                if not env.can_place_box(box_dims):
                    new_buffer.append(box_dims); continue
                # When retrying from buffer, we pass the current buffer size minus one (conceptually)
                # or just the current length. Let's use current length for simplicity.
                placed, _ = try_place_one_box(box_dims, len(buffer))
                if placed:
                    made_progress = True
                else:
                    if len(new_buffer) < max_buffer_size:
                        new_buffer.append(box_dims)
                    else:
                        # Buffer full and cannot place → end episode
                        done = True
                        break
                if env._is_terminal():
                    done = True; break
            buffer = new_buffer
            if not made_progress:
                done = True
                break

        # metrics
        pallet_volume = env.pallet_size[0] * env.pallet_size[1] * env.max_height
        placed_volume = sum(b[2] * b[3] * b[4] for b in env.placed_boxes)
        utilization = placed_volume / pallet_volume if pallet_volume > 0 else 0
        total_utilization += utilization

        if total_decisions > 0:
            perc = {k: 100 * episode_heuristic_counts[k] / total_decisions for k in episode_heuristic_counts}
        else:
            perc = {k: 0 for k in episode_heuristic_counts}

        all_metrics.append({
            'episode': episode + 1,
            'utilization': utilization,
            'invalid_learned': env.invalid_actions_learned,
            'invalid_attempted': env.invalid_actions_attempted,
            'boxes_attempted': attempts_this_episode,
            'placed_boxes': len(env.placed_boxes),
            'max_height': np.max(env.current_height_map),
            'avg_height': np.mean(env.current_height_map),
            'perc_stacking': perc['stacking'],
            'perc_best_fit': perc['best_fit'],
            'perc_semi_perfect_fit': perc['semi_perfect_fit'],
            'perc_random_fit': perc['random_fit'],
            'perc_corner': perc['corner'],
            'avg_q_loss': np.mean(episode_q_losses) if episode_q_losses else 0.0,
            'avg_mask_loss': np.mean(episode_mask_losses) if episode_mask_losses else 0.0
        })

        if (episode + 1) % 100 == 0:
            env.visualize_pallet(
                episode_num=episode + 1,
                boxes_attempted=attempts_this_episode,
                utilization=utilization,
                invalid_learned=env.invalid_actions_learned,
                invalid_attempted=env.invalid_actions_attempted,
                output_dir=output_dir,
            )

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Update target network ONCE per episode (critical for stable Q-learning)
        agent.update_target_model()

        # Get agent losses for printing
        q_l = getattr(agent, 'last_q_loss', 0.0)
        m_l = getattr(agent, 'last_mask_loss', 0.0)

        if (episode + 1) % 10 == 0:
            # Update progress bar description with key metrics
            pbar.set_postfix({
                'Util': f"{utilization:.1%}",
                'Eps': f"{agent.epsilon:.2f}",
                'Q': f"{q_l:.1f}",
                'M': f"{m_l:.2f}"
            })
            
        if verbose and (episode + 1) % 100 == 0:
             print(f"\nEpisode: {episode + 1:04d} | Util: {utilization:.1%} | "
                   f"Inv(L/A): {env.invalid_actions_learned:02d}/{env.invalid_actions_attempted:02d} | "
                   f"Pl: {len(env.placed_boxes):02d} | ε: {agent.epsilon:.3f}")

    avg_utilization = total_utilization / total_episodes
    total_invalid_learned = sum(m['invalid_learned'] for m in all_metrics)
    total_invalid_attempted = sum(m['invalid_attempted'] for m in all_metrics)
    print("\nTraining Summary:")
    print(f"Average Utilization: {avg_utilization:.2%}")
    print(f"Total Invalid Learned: {total_invalid_learned}")
    print(f"Total Invalid Attempted: {total_invalid_attempted}")

    # summary row
    avg_perc_stacking = np.mean([m['perc_stacking'] for m in all_metrics])
    avg_perc_best_fit = np.mean([m['perc_best_fit'] for m in all_metrics])
    avg_perc_semi = np.mean([m['perc_semi_perfect_fit'] for m in all_metrics])
    avg_perc_rand = np.mean([m['perc_random_fit'] for m in all_metrics])
    avg_perc_corner = np.mean([m['perc_corner'] for m in all_metrics])

    summary_metrics = {
        'episode': 'SUMMARY',
        'utilization': avg_utilization,
        'invalid_learned': total_invalid_learned,
        'invalid_attempted': total_invalid_attempted,
        'boxes_attempted': '',
        'placed_boxes': '',
        'max_height': '',
        'avg_height': '',
        'perc_stacking': avg_perc_stacking,
        'perc_best_fit': avg_perc_best_fit,
        'perc_semi_perfect_fit': avg_perc_semi,
        'perc_random_fit': avg_perc_rand,
        'perc_corner': avg_perc_corner
    }

    final_metrics_df = pd.DataFrame(all_metrics + [summary_metrics])

    # trend
    plt.figure(figsize=(12, 6))

    # Use only the per-episode rows (skip the SUMMARY row if present)
    rows = [m for m in all_metrics if isinstance(m['episode'], int)]
    episodes_x = [r['episode'] for r in rows]
    utils_y = [r['utilization'] for r in rows]
    
    plt.plot(episodes_x, utils_y, label='Utilization', alpha=0.6)
    
    # moving average
    w = 50
    if len(utils_y) >= w:
        ma = pd.Series(utils_y).rolling(window=w).mean()
        plt.plot(episodes_x, ma, label=f'MA({w})', color='red', linewidth=2)
        
    plt.xlabel('Episode')
    plt.ylabel('Utilization')
    plt.title('Training Utilization Trend')
    plt.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'utilization_trend.png'))
    plt.close()
    
    # loss trend plot
    plt.figure(figsize=(12, 6))
    q_losses = [r['avg_q_loss'] for r in rows]
    m_losses = [r['avg_mask_loss'] for r in rows]
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    p1, = ax1.plot(episodes_x, q_losses, label='Q-Loss', color='blue', alpha=0.4)
    p2, = ax2.plot(episodes_x, m_losses, label='Mask-Loss', color='green', alpha=0.4)
    
    # moving averages
    if len(q_losses) >= w:
        q_ma = pd.Series(q_losses).rolling(window=w).mean()
        m_ma = pd.Series(m_losses).rolling(window=w).mean()
        ax1.plot(episodes_x, q_ma, color='blue', linewidth=2, label=f'Q-Loss MA({w})')
        ax2.plot(episodes_x, m_ma, color='green', linewidth=2, label=f'Mask-Loss MA({w})')
        
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Q-Loss (MSE)', color='blue')
    ax2.set_ylabel('Mask-Loss (BCE)', color='green')
    plt.title('Training Loss Evolution')
    
    # combined legend
    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right')
    
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'loss_trend.png'))
    plt.close()
    
    if model_save_path:
        agent.save_model(model_save_path)
        print(f"Model saved to: {model_save_path}")

    if return_agent:
        return final_metrics_df, agent
    return final_metrics_df

