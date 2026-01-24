import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pandas as pd


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


# =========================================================
# Ground-truth feasibility mask with BRIDGING rule (GT)
# =========================================================
def compute_gt_mask(
    height_map,
    box_dims,
    pallet_size=(10, 10),
    max_height=10,
    min_support_ratio=0.50,
    require_opposite_edge_support=True,
    max_gap=2,
):
    """
    Returns a boolean mask of shape (2, L, W), one channel per rotation:
      rot 0: (L, W, H)
      rot 1: (H, W, L)

    BRIDGING rule (matches env._is_valid_placement):
      - The box rests at base_z = max(placement_area).
      - Support cells = {cells at base_z}.
      - Must satisfy:
          * support_ratio >= min_support_ratio
          * (optional) along X OR along Y:
              supports touch both opposite edges AND max gap of unsupported
              consecutive columns/rows along that axis <= max_gap
          * COM-in-span: the footprint center lies between the min/max supported
              indices along both axes (basic stability check).
      - Height limit: base_z + h <= max_height
      - Bounds satisfied
    """
    L, W = pallet_size
    mask = np.zeros((2, L, W), dtype=bool)
    rotations = [(0, 1, 2), (2, 1, 0)]  # must match env.get_rotated_box_dims

    def axis_ok(support_mask, w, d, axis, max_gap_local):
        # axis=0 → spanning along X (columns), need left & right edges touched
        # axis=1 → spanning along Y (rows), need front & back edges touched
        if axis == 0:
            col_any = support_mask.any(axis=1)  # length w
            edges = (col_any[0], col_any[-1])
            if not (edges[0] and edges[1]):
                return False
            return _max_zero_run(col_any) <= max_gap_local
        else:
            row_any = support_mask.any(axis=0)  # length d
            edges = (row_any[0], row_any[-1])
            if not (edges[0] and edges[1]):
                return False
            return _max_zero_run(row_any) <= max_gap_local

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
                base = np.max(region)
                if base + h > max_height:
                    continue

                support_mask = (region == base)
                support_count = int(support_mask.sum())
                total = l * w
                support_ratio = support_count / total if total > 0 else 0.0
                if support_ratio < min_support_ratio:
                    continue

                # Center of the footprint (index space 0..l-1, 0..w-1)
                com_x = (l - 1) / 2.0
                com_y = (w - 1) / 2.0
                xs, ys = np.where(support_mask)
                if xs.size == 0:
                    continue
                if not (xs.min() <= com_x <= xs.max() and ys.min() <= com_y <= ys.max()):
                    # COM must lie within the span of supports (basic stability)
                    continue

                # Opposite-edge spanning condition (either X or Y)
                if require_opposite_edge_support:
                    ok_x = axis_ok(support_mask, l, w, axis=0, max_gap_local=max_gap)
                    ok_y = axis_ok(support_mask, l, w, axis=1, max_gap_local=max_gap)
                    if not (ok_x or ok_y):
                        continue

                mask[r_idx, x, y] = True
    return mask


# ==================
# Environment
# ==================
class BoxPilingEnv:
    def __init__(
        self,
        pallet_size=(10, 10),
        max_height=10,
        # --- NEW BRIDGING PARAMS ---
        min_support_ratio=0.50,
        require_opposite_edge_support=True,
        max_gap=2,
    ):
        self.pallet_size = pallet_size
        self.max_height = max_height
        self.current_height_map = np.zeros(pallet_size)
        self.current_box = None
        self.placed_boxes = []

        # bridging params
        self.min_support_ratio = float(min_support_ratio)
        self.require_opposite_edge_support = bool(require_opposite_edge_support)
        self.max_gap = int(max_gap)

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
        # 1) Bounds
        if (x + w > self.pallet_size[0]) or (y + d > self.pallet_size[1]):
            return False

        region = self.current_height_map[x:x + w, y:y + d]
        base_z = int(np.max(region))

        # 2) Height limit
        if base_z + h > self.max_height:
            return False

        # 3) BRIDGING support rule
        support_mask = (region == base_z)
        support_ratio = support_mask.sum() / (w * d)
        if support_ratio < self.min_support_ratio:
            return False

        # basic COM-in-span stability
        com_x = (w - 1) / 2.0
        com_y = (d - 1) / 2.0
        xs, ys = np.where(support_mask)
        if xs.size == 0:
            return False
        if not (xs.min() <= com_x <= xs.max() and ys.min() <= com_y <= ys.max()):
            return False

        # opposite-edge spanning check
        def axis_ok(mask, span_axis, max_gap_local):
            if span_axis == 0:  # X axis spanning
                col_any = mask.any(axis=1)  # length w
                if not (col_any[0] and col_any[-1]):
                    return False
                return _max_zero_run(col_any) <= max_gap_local
            else:               # Y axis spanning
                row_any = mask.any(axis=0)  # length d
                if not (row_any[0] and row_any[-1]):
                    return False
                return _max_zero_run(row_any) <= max_gap_local

        if self.require_opposite_edge_support:
            ok_x = axis_ok(support_mask, 0, self.max_gap)
            ok_y = axis_ok(support_mask, 1, self.max_gap)
            if not (ok_x or ok_y):
                return False

        return True

    def _update_height_map(self, x, y, w, d, h):
        region = self.current_height_map[x:x + w, y:y + d]
        base_z = int(np.max(region))
        self.current_height_map[x:x + w, y:y + d] = base_z + h
        return base_z

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
                if self._is_valid_placement(xx, yy, w, d, h):
                    action = xx * W * 2 + yy * 2 + rotation
                    valid_actions.append(action)
                    mapping[str(action)] = str((xx, yy, w, d, h))
        return valid_actions, mapping

    def can_place_box(self, box_dims):
        acts, _ = self.get_valid_actions(box_dims)
        return len(acts) > 0

    # ----- Heuristics (unchanged) -----
    def heuristic_stacking(self, valid_actions):
        best_action, best_support = None, -np.inf
        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            region = self.current_height_map[xx:xx + w, yy:yy + d]
            support_level = np.max(region)
            if support_level > best_support:
                best_support, best_action = support_level, action
        return best_action

    def heuristic_best_fit(self, valid_actions):
        best_action, best_gap = None, np.inf
        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            region = self.current_height_map[xx:xx + w, yy:yy + d]
            support_level = np.max(region)
            gap = self.max_height - (support_level + h)
            if gap < best_gap:
                best_gap, best_action = gap, action
        return best_action

    def heuristic_semi_perfect_fit(self, valid_actions):
        best_action, best_score = None, np.inf
        for action in valid_actions:
            rotation = action % 2
            remaining = action // 2
            yy = remaining % self.pallet_size[1]
            xx = remaining // self.pallet_size[1]
            w, d, h = self.get_rotated_box_dims(self.current_box, rotation)
            region = self.current_height_map[xx:xx + w, yy:yy + d]
            support_level = np.max(region)
            support_count = np.sum(region == support_level)
            waste = (w * d - support_count)
            gap = self.max_height - (support_level + h)
            score = waste + gap
            if score < best_score:
                best_score, best_action = score, action
        return best_action

    def heuristic_random_fit(self, valid_actions):
        placed_volume = sum(b[2] * b[3] * b[4] for b in self.placed_boxes)
        pallet_volume = self.pallet_size[0] * self.pallet_size[1] * self.max_height
        utilization = placed_volume / pallet_volume if pallet_volume > 0 else 0
        threshold = 0.10
        p_stack = 0.66 if utilization < threshold else 0.33
        return (self.heuristic_stacking(valid_actions)
                if np.random.rand() < p_stack
                else self.heuristic_semi_perfect_fit(valid_actions))

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
        if not self._is_valid_placement(xx, yy, w, d, h):
            self.invalid_actions_attempted += 1
            reward = -2000
            return self._get_state(), reward, False, {"invalid": True}

        base_z = self._update_height_map(xx, yy, w, d, h)
        self.placed_boxes.append((xx, yy, w, d, h, base_z))

        # (reward unchanged; still encourages creating large flat areas)
        original_max_space = self._calculate_maximal_flat_area(self.current_height_map)
        temp_height_map = self.current_height_map.copy()
        temp_height_map[xx:xx + w, yy:yy + d] += h
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
        action_size=4,
        max_height=10,
        # same bridging params for GT mask
        min_support_ratio=0.50,
        require_opposite_edge_support=True,
        max_gap=2,
        learning_rate=0.001,
    ):
        self.state_dims = state_dims
        self.action_size = action_size
        self.max_height = max_height
        self.learning_rate = learning_rate

        # store GT policy for mask supervision
        self.min_support_ratio = float(min_support_ratio)
        self.require_opposite_edge_support = bool(require_opposite_edge_support)
        self.max_gap = int(max_gap)

        self.model = self._build_model()
        self.target_model = self._build_model()

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        self.memory = []
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.mask_loss_weight = 0.2
        self.optimizer_step_count = 0

    def _build_model(self):
        class Net(nn.Module):
            def __init__(self, pallet_size, action_size):
                super().__init__()
                L, W = pallet_size
                self.L, self.W = L, W
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                    nn.Flatten()
                )
                with torch.no_grad():
                    conv_out = self.conv(torch.zeros(1, 1, L, W)).shape[1]
                self.val = nn.Sequential(nn.Linear(conv_out + 3, 128), nn.ReLU(), nn.Linear(128, 1))
                self.adv = nn.Sequential(nn.Linear(conv_out + 3, 128), nn.ReLU(), nn.Linear(128, action_size))
                self.mask_head = nn.Sequential(nn.Linear(conv_out + 3, 256), nn.ReLU(),
                                               nn.Linear(256, 2 * L * W))

            def forward(self, hm, bd):
                f = self.conv(hm)
                z = torch.cat([f, bd], dim=1)
                v = self.val(z)
                a = self.adv(z)
                q = v + (a - a.mean(dim=1, keepdim=True))
                mask_logits = self.mask_head(z).view(-1, 2, self.L, self.W)
                return q, mask_logits

        return Net(self.state_dims['height_map'], self.action_size)

    # memory
    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    # normalization
    def _norm(self, state):
        L, W = self.state_dims['height_map']
        hm = state['height_map'] / float(self.max_height)
        bd = state['box_dims'] / np.array([L, W, self.max_height], dtype=np.float32)
        return (torch.FloatTensor(hm).unsqueeze(0).unsqueeze(0),
                torch.FloatTensor(bd).unsqueeze(0))

    # action selection (soft bias is computed outside and passed in)
    @torch.no_grad()
    def act_with_mask_bias(self, state, mask_bias=None, beta=0.5, eps=1e-6):
        if np.random.rand() <= self.epsilon:
            return int(np.random.randint(0, self.action_size))
        hm, bd = self._norm(state)
        q, _ = self.model(hm, bd)
        q = q.squeeze(0)
        if mask_bias is not None:
            b = torch.tensor(mask_bias, dtype=torch.float32)
            b = (b - b.mean()) / (b.std() + eps)
            q = q + beta * b
        return int(torch.argmax(q).item())

    @torch.no_grad()
    def predict_mask(self, state, threshold=0.5):
        hm, bd = self._norm(state)
        _, ml = self.model(hm, bd)
        return (torch.sigmoid(ml)[0].cpu().numpy() > threshold)

    # training
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        L, W = self.state_dims['height_map']
        hm_s_raw = np.array([s['height_map'] for s in states], np.float32)
        bd_s_raw = np.array([s['box_dims'] for s in states], np.float32)
        hm_n_raw = np.array([s['height_map'] for s in next_states], np.float32)
        bd_n_raw = np.array([s['box_dims'] for s in next_states], np.float32)

        hm_s = torch.FloatTensor(hm_s_raw / float(self.max_height)).unsqueeze(1)
        bd_s = torch.FloatTensor(bd_s_raw / np.array([L, W, self.max_height], np.float32))
        hm_n = torch.FloatTensor(hm_n_raw / float(self.max_height)).unsqueeze(1)
        bd_n = torch.FloatTensor(bd_n_raw / np.array([L, W, self.max_height], np.float32))

        q_cur, mask_logits = self.model(hm_s, bd_s)
        q_cur = q_cur.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next, _ = self.target_model(hm_n, bd_n)
            q_next, _ = q_next.max(1)
        dones_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones])
        tgt = torch.FloatTensor(rewards) + self.gamma * q_next * (1 - dones_t)
        q_loss = nn.MSELoss()(q_cur, tgt)

        # GT masks using the SAME BRIDGING RULE
        gt_list = []
        for hm_raw, bd_raw in zip(hm_s_raw, bd_s_raw):
            gt = compute_gt_mask(
                hm_raw, bd_raw, pallet_size=(L, W), max_height=self.max_height,
                min_support_ratio=self.min_support_ratio,
                require_opposite_edge_support=self.require_opposite_edge_support,
                max_gap=self.max_gap,
            )
            gt_list.append(gt.astype(np.float32))
        gt_masks = torch.from_numpy(np.stack(gt_list, 0))  # (B,2,L,W)

        # skip samples without a box
        has_box = torch.from_numpy((bd_s_raw.sum(axis=1) > 0).astype(np.float32))

        pos = gt_masks.sum()
        neg = gt_masks.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp_(1, 50)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        per_elem = bce(mask_logits, gt_masks)
        per_sample = per_elem.mean(dim=(1, 2, 3))
        mask_loss = (per_sample * has_box).sum() / (has_box.sum() + 1e-6)

        loss = q_loss + self.mask_loss_weight * mask_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer_step_count += 1

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# ===========================================
# Training loop (Option-A soft-bias unchanged)
# ===========================================
def train(
    episodes_boxes,
    output_dir,
    verbose=False,
    episode_to_show=100,
    defer_penalty=-5.0,
    mask_bias_beta=0.5,
    # expose bridging params here (env+agent share them)
    min_support_ratio=0.50,
    require_opposite_edge_support=True,
    max_gap=2,
    learning_rate=0.001,
    max_buffer_size=float('inf'),
    return_agent=False,
):
    env = BoxPilingEnv(
        min_support_ratio=min_support_ratio,
        require_opposite_edge_support=require_opposite_edge_support,
        max_gap=max_gap,
    )
    agent = DQNAgent(
        state_dims={'height_map': env.pallet_size, 'box_dims': 3},
        action_size=4,
        max_height=env.max_height,
        min_support_ratio=min_support_ratio,
        require_opposite_edge_support=require_opposite_edge_support,
        max_gap=max_gap,
        learning_rate=learning_rate,
    )

    total_episodes = len(episodes_boxes)
    total_utilization = 0.0
    all_metrics = []
    heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'random_fit'}

    def proxy_scores_for_heuristics(env_obj, pred_mask):
        scores = {'stacking': -1e9, 'best_fit': -1e9, 'semi_perfect_fit': -1e9, 'random_fit': 0.0}
        box = env_obj.current_box
        if box is None:
            return [0.0, 0.0, 0.0, 0.0]
        Lp, Wp = env_obj.pallet_size
        for rot in [0, 1]:
            w, d, h = env_obj.get_rotated_box_dims(box, rot)
            cand = np.argwhere(pred_mask[rot])
            for xx, yy in cand:
                xx = int(xx); yy = int(yy)
                if xx > Lp - w or yy > Wp - d:
                    continue
                region = env_obj.current_height_map[xx:xx + w, yy:yy + d]
                support = float(np.max(region))
                support_count = int(np.sum(region == support))
                gap = float(env_obj.max_height - (support + h))
                waste = int(w * d - support_count)
                scores['stacking'] = max(scores['stacking'], support)
                scores['best_fit'] = max(scores['best_fit'], -gap)
                scores['semi_perfect_fit'] = max(scores['semi_perfect_fit'], -(waste + gap))
        return [scores['stacking'], scores['best_fit'], scores['semi_perfect_fit'], scores['random_fit']]

    for episode in range(total_episodes):
        state = env.reset()
        done = False
        boxes = episodes_boxes[episode]
        box_idx = 0

        episode_reward = 0
        episode_heuristic_counts = {k: 0 for k in heuristic_map.values()}
        total_decisions = 0
        buffer = []
        placements_this_episode = 0
        attempts_this_episode = 0

        def try_place_one_box(box_dims):
            nonlocal state, episode_reward, total_decisions, placements_this_episode, attempts_this_episode
            state = env.new_box_arrival(box_dims)
            attempts_this_episode += 1

            pred_mask = agent.predict_mask(state)
            mask_bias = proxy_scores_for_heuristics(env, pred_mask)
            h_idx = agent.act_with_mask_bias(state, mask_bias=mask_bias, beta=mask_bias_beta)
            heuristic = heuristic_map[h_idx]
            episode_heuristic_counts[heuristic] += 1
            total_decisions += 1

            action, mapping = env.choose_action_by_heuristic(heuristic, pred_mask=pred_mask)
            if action is None:
                action, mapping = env.choose_action_by_heuristic(heuristic, pred_mask=None)
                if action is None:
                    agent.remember(state, h_idx, defer_penalty, state, False)
                    agent.replay()
                    agent.update_target_model()
                    return False, defer_penalty

            if verbose and episode == episode_to_show:
                print(env.current_height_map)
                print(f"Heuristic: {heuristic} | Action: {mapping}")

            next_state, reward, local_done, info = env.step(action)
            agent.remember(state, h_idx, reward, next_state, local_done)
            agent.replay()
            agent.update_target_model()
            state = next_state
            episode_reward += reward
            placements_this_episode += 1
            return True, reward

        # initial stream
        while not done and box_idx < len(boxes):
            box_dims = boxes[box_idx]
            box_idx += 1
            placed, _ = try_place_one_box(box_dims)
            if not placed:
                if len(buffer) < max_buffer_size:
                    buffer.append(box_dims)
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
                placed, _ = try_place_one_box(box_dims)
                if placed:
                    made_progress = True
                else:
                    if len(new_buffer) < max_buffer_size:
                        new_buffer.append(box_dims)
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
        if agent.optimizer_step_count > 0:
            agent.scheduler.step()
            agent.optimizer_step_count = 0

        print(f"Episode: {episode + 1:04d} | Util: {utilization:.1%} | "
              f"Invalid Lrn/Att: {env.invalid_actions_learned:02d}/{env.invalid_actions_attempted:02d} | "
              f"Placed: {len(env.placed_boxes):02d} | ε: {agent.epsilon:.3f}")

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
        'perc_random_fit': avg_perc_rand
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
    
    if return_agent:
        return final_metrics_df, agent
    return final_metrics_df

""" # -------------
# Usage Example
# -------------
if __name__ == "__main__":
    # Demo data
    episodes_boxes = [
        [np.random.randint(1, 5, size=3).tolist() for _ in range(random.randint(5, 15))]
        for _ in range(2100)
    ]
    out_dir = os.path.join(os.getcwd(), "training_output")
    os.makedirs(out_dir, exist_ok=True)

    # You can tweak these three to match your hardware scenario
    final_metrics = train(
        episodes_boxes,
        output_dir=out_dir,
        # BRIDGING parameters:
        min_support_ratio=0.50,                # fraction of footprint on top level
        require_opposite_edge_support=True,    # must touch both opposite edges along X or Y
        max_gap=2,                             # max unsupported run across spanning axis
    )
    final_metrics.to_csv(os.path.join(out_dir, "final_metrics.csv"), index=False) """
