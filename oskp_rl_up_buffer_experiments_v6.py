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
    rotations = [(0, 1, 2)] # [(0, 1, 2), (1, 0, 2)]
    
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

    # Second pass: selective binary mask
    if len(valid_positions) > 0:
        scores = np.array([pos[3] for pos in valid_positions])
        max_score = scores.max()
        
        # Define tolerance for float comparison (effectively selecting top-tier spots)
        tolerance = 1e-5
        
        for r_idx, x, y, flatness in valid_positions:
            # Binary Mask: 1.0 if it matches the best possible flatness, 0.0 otherwise
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
        valid_rotations = [(0, 1, 2), (1, 0, 2)]  # (L,W,H) and (W,L,H) - "This Side Up"
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

        for rotation in range(1): # range(2)
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

    def heuristic_complex_fit(self, valid_actions):
        """
        [cite_start]Paper (Ref): Prioritizes placements where Replication Score is high.[cite: 4.3.5]
        Logic: Look for "Gaps" (distance to nearest wall or taller box) that are perfect multiples 
        of the current box's dimensions.
        """
        best_action, best_score = None, -np.inf # Higher is better
        L, W = self.pallet_size

        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            
            valid, z = self._is_valid_placement(xx, yy, w, d, h)
            if valid:
                # 1. Measure Gap in X (from xx to nearest wall/taller-box)
                gap_x = 0
                for ix in range(xx + w, L):
                    if np.any(self.current_height_map[ix, yy:yy+d] > z): break
                    gap_x += 1
                else: gap_x = L - (xx + w) # hit wall
                
                # 2. Measure Gap in Y
                gap_y = 0
                for iy in range(yy + d, W):
                    if np.any(self.current_height_map[xx:xx+w, iy] > z): break
                    gap_y += 1
                else: gap_y = W - (yy + d) # hit wall
                
                # Full gap including current box
                full_gap_x = gap_x + w
                full_gap_y = gap_y + d
                
                # Replication Scoring: residue=0 is perfect.
                residue_x = full_gap_x % w
                residue_y = full_gap_y % d
                
                # Inverse residue score (Penalty for gaps that aren't multiples)
                # Perfect match in both = 2000.
                score = (1.0 if residue_x == 0 else 0.0) * 1000 + \
                        (1.0 if residue_y == 0 else 0.0) * 1000
                
                # Tie-breakers: Tight packing + Gravity
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
            # Fallback to FIRST valid action to be deterministic
            action = valid_actions[0]
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
            reward = -0.5
            return self._get_state(), reward, False, {"invalid": True}

        # Calculate Old Flatness
        old_flatness = self._calculate_maximal_flat_area(self.current_height_map)

        self._update_height_map(xx, yy, w, d, h, base_z)
        self.placed_boxes.append((xx, yy, w, d, h, base_z))

        # Calculate New Flatness
        new_flatness = self._calculate_maximal_flat_area(self.current_height_map)

        # Reward: Volume Utilization + Flatness Delta
        # (Box Volume / Pallet Volume) * Scale
        pallet_vol = self.pallet_size[0] * self.pallet_size[1] * self.max_height
        box_vol = w * d * h
        
        # New Reward Formula (Normalized ~ 0.0 - 1.2)
        r_vol = box_vol / pallet_vol
        # Normalized flatness delta relative to surface area
        r_flat = (new_flatness - old_flatness) / (self.pallet_size[0] * self.pallet_size[1])
        
        reward = r_vol + (r_flat * 0.1)
        
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
        action_size=1,  # Corner-only: single action
        max_height=10,
        learning_rate=0.001,
        # Unused params kept for compatibility if needed
        min_support_ratio=0.60,
        require_opposite_edge_support=True,
    ):
        self.state_dims = state_dims
        self.pallet_size = state_dims['height_map']
        self.action_size = action_size
        self.max_height = max_height
        self.learning_rate = learning_rate

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-8)

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
    def _norm(self, state, buffer_count=0, proxy_scores=None):
        L, W = self.state_dims['height_map']
        
        # 1. Height Map (Channel 0) -> Spatial Branch
        hm = state['height_map'] / float(self.max_height)
        hm_tensor = torch.FloatTensor(hm).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, 10, 10)
        
        # 2. Vector Features -> Vector Branch
        # [BoxL, BoxW, BoxH, BufCount, S1, S2, S3, S4]
        box_dims = state['box_dims']
        if np.sum(box_dims) > 0:
            bd_norm = [box_dims[0]/L, box_dims[1]/W, box_dims[2]/self.max_height]
        else:
            bd_norm = [0, 0, 0]
            
        buf_norm = [min(buffer_count, 10) / 10.0]
        
        if proxy_scores is None:
            scores = [0.0] * 1
        else:
            scores = proxy_scores[:1]

        vec = bd_norm + buf_norm + scores
        vec_tensor = torch.FloatTensor(vec).unsqueeze(0).to(self.device)
        
        return hm_tensor, vec_tensor

    def _build_model(self):
        class ProNet(nn.Module):
            def __init__(self, pallet_size, action_size):
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
                
                # Branch B: Vector (Box + Buffer + Scores = 5 dims: 3 box + 1 buf + 1 score)
                self.vector = nn.Sequential(
                    nn.Linear(5, 64), nn.ReLU(),
                    nn.Linear(64, 64), nn.ReLU()
                )
                
                # Fusion
                fusion_size = spatial_out_size + 64
                
                # Critic Head (Value)
                self.val = nn.Sequential(
                    nn.Linear(fusion_size, 512), nn.ReLU(), 
                    nn.Linear(512, 1)
                )
                
                # Actor Head (Advantage)
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
                
                # Fuse
                combined = torch.cat([s_feat, v_feat], dim=1)
                
                v = self.val(combined)
                a = self.adv(combined)
                q = v + (a - a.mean(dim=1, keepdim=True))
                
                # Mask prediction uses ONLY spatial features
                mask_logits = self.mask_head(s_feat).view(-1, 2, self.L, self.W)
                
                return q, mask_logits

        return ProNet(self.state_dims['height_map'], self.action_size)

    # --- Standard Methods (Updated for new input shape) ---

    def remember(self, s, a, r, ns, d, buffer_count, proxy_scores):
        # We re-compute mask here or passed in. 
        L, W = self.state_dims['height_map']
        hm_raw = s['height_map']
        bd_raw = s['box_dims']
        
        if np.sum(bd_raw) > 0:
            gt_mask = compute_gt_mask(hm_raw, bd_raw, pallet_size=(L, W), max_height=self.max_height)
            gt_mask = gt_mask.astype(np.float32)
        else:
            gt_mask = np.zeros((2, L, W), dtype=np.float32)
            
        self.memory.append((s, a, r, ns, d, buffer_count, gt_mask, proxy_scores))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    @torch.no_grad()
    def get_action_with_prior(self, state, proxy_scores, buffer_count=0):
        # Corner-only: always return action 0 (corner)
        return 0

    @torch.no_grad()
    def predict_mask(self, state, buffer_count=0, threshold=0.5):
        # No proxy scores needed for mask
        hm_t, vec_t = self._norm(state, buffer_count)
        _, ml = self.model(hm_t, vec_t)
        return (torch.sigmoid(ml)[0].cpu().numpy() > threshold)
    
    @torch.no_grad()
    def get_mask_confidence_batch(self, states, buffer_counts):
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
            buf_norm = [min(buffer_counts[i], 10) / 10.0]
            sc = [0.0]*1
            vec_list.append(bd_norm + buf_norm + sc)
            
        hm_t = torch.FloatTensor(np.array(hm_list)).to(self.device)
        vec_t = torch.FloatTensor(np.array(vec_list)).to(self.device)
        _, ml = self.model(hm_t, vec_t)
        return torch.sigmoid(ml).cpu().numpy()

    @torch.no_grad()
    def get_action_with_prior_batch(self, states, proxy_scores_list, buffer_counts):
        # Corner-only: always return action 0 (corner) for all environments
        return [0] * len(states)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, buffer_counts, gt_masks_list, proxy_scores_list = zip(*batch)

        L, W = self.state_dims['height_map']
        
        # Helper to batch process Inputs
        def prepare_batch(s_list, buf_list, scores_list):
            hm_list = []
            vec_list = []
            for i, s in enumerate(s_list):
                # Spatial
                hm = s['height_map'] / float(self.max_height)
                hm_list.append(hm[np.newaxis, :, :])
                
                # Vector
                bd = s['box_dims']
                if np.sum(bd) > 0:
                    bd_norm = [bd[0]/L, bd[1]/W, bd[2]/self.max_height]
                else:
                    bd_norm = [0, 0, 0]
                
                buf_norm = [min(buf_list[i], 10) / 10.0]
                
                if scores_list is None:
                    sc = [0.0]*1
                else:
                    sc = scores_list[i]
                
                vec_list.append(bd_norm + buf_norm + sc)
                
            hm_tensor = torch.FloatTensor(np.array(hm_list)).to(self.device)
            vec_tensor = torch.FloatTensor(np.array(vec_list)).to(self.device)
            return hm_tensor, vec_tensor

        # 1. Current State
        hm_curr, vec_curr = prepare_batch(states, buffer_counts, proxy_scores_list)
        gt_masks = torch.from_numpy(np.array(gt_masks_list, dtype=np.float32)).to(self.device)
        
        # Forward pass
        q_cur, mask_logits = self.model(hm_curr, vec_curr)
        q_cur = q_cur.gather(1, torch.LongTensor(actions).to(self.device).unsqueeze(1)).squeeze(1)
        
        # 2. Next State
        # Need to predict next mask to get next proxy scores
        with torch.no_grad():
             # We pass None for scores first just to get mask
             hm_next, vec_next_dummy = prepare_batch(next_states, buffer_counts, None)
             _, mask_logits_next = self.target_model(hm_next, vec_next_dummy)
             mask_probs_next = torch.sigmoid(mask_logits_next).cpu().numpy()
             
             # Compute Next Proxy Scores per sample
             next_scores_list = []
             for i in range(len(next_states)):
                 ns = next_states[i]
                 sc = proxy_scores_for_heuristics(
                     ns['height_map'], ns['box_dims'], 
                     (L, W), self.max_height, 
                     mask_probs_next[i]
                 )
                 next_scores_list.append(sc)
            
             # Now fully prepare next inputs
             hm_next, vec_next = prepare_batch(next_states, buffer_counts, next_scores_list)
             q_next, _ = self.target_model(hm_next, vec_next)
             q_next, _ = q_next.max(1)
            
        dones_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        tgt = rewards_t + self.gamma * q_next * (1 - dones_t)
        
        q_loss = nn.SmoothL1Loss()(q_cur, tgt)
        
        # Mask Loss (Only for valid boxes)
        # Check if box exists (box_dims sum > 0)
        has_box = torch.tensor([1.0 if np.sum(s['box_dims']) > 0 else 0.0 for s in states]).to(self.device)
        
        # Use BCEWithLogitsLoss for binary mask
        # reduction='none' so we can apply has_box mask
        per_pixel_loss = nn.BCEWithLogitsLoss(reduction='none')(mask_logits, gt_masks)
        per_sample_loss = per_pixel_loss.mean(dim=(1,2,3)) # Mean over C, H, W
        
        mask_loss = (per_sample_loss * has_box).sum() / (has_box.sum() + 1e-6)
        
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
            'memory': self.memory
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.target_model.load_state_dict(self.model.state_dict())



def proxy_scores_for_heuristics(height_map, box_dims, pallet_size, max_height, pred_mask_probs):
    """
    Corner-only version: Returns 1 proxy score for the corner heuristic.
    """
    scores = [0.0]

    if np.sum(box_dims) == 0: return scores

    Lp, Wp = pallet_size

    def get_rotated(bd, rot):
        if rot == 0: return int(bd[0]), int(bd[1]), int(bd[2])
        return int(bd[1]), int(bd[0]), int(bd[2])

    rot_dims = [get_rotated(box_dims, 0), get_rotated(box_dims, 1)]

    best_corn = (-float('inf'), 0.0)
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

            # Corner (Maximize Walls)
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

    scores[0] = best_corn[1]

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
    defer_penalty=-0.5,
    mask_bias_beta=0.5,
    learning_rate=0.001,
    max_buffer_size=float('inf'),
    min_support_ratio=0.60,
    require_opposite_edge_support=True,
    return_agent=False,
):
    env = BoxPilingEnv()
    agent = DQNAgent(
        state_dims={'height_map': env.pallet_size, 'box_dims': 3},
        action_size=1,  # Corner-only
        max_height=env.max_height,
        learning_rate=learning_rate,
        min_support_ratio=min_support_ratio,
        require_opposite_edge_support=require_opposite_edge_support,
    )

    total_episodes = len(episodes_boxes)
    total_utilization = 0.0
    all_metrics = []
    heuristic_map = {0: 'corner'}

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
                    agent.remember(state, h_idx, defer_penalty, state, True, current_buffer_size)
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

