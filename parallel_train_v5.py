import numpy as np
import torch
from tqdm import tqdm
import os

def make_env():
    from oskp_rl_up_buffer_experiments_v5 import BoxPilingEnv
    return BoxPilingEnv()

def parallel_train_one_epoch(agent, episodes_boxes, output_dir, env_params, n_envs=8):
    from vec_env_v5 import SubprocVecEnv

    # --- Configuration ---
    max_buffer_size = env_params.get('max_buffer_size', 0)
    heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'corner', 4: 'complex_fit', 5: 'BUFFER'}

    # Scale batch size
    original_batch_size = agent.batch_size
    agent.batch_size = 64 * n_envs

    envs = SubprocVecEnv([make_env for _ in range(n_envs)])

    n_episodes = len(episodes_boxes)
    pbar = tqdm(total=n_episodes, desc="Parallel Training v5", unit="ep")

    # State tracking
    active_ep_indices = list(range(n_envs))
    next_ep_idx = n_envs

    envs.reset()

    slot_boxes_ptr = [0] * n_envs
    slot_buffers = [[] for _ in range(n_envs)]
    slot_phases = ['STREAM'] * n_envs
    slot_done = [False] * n_envs
    slot_states = [None] * n_envs
    slot_buffer_tried_count = [0] * n_envs
    slot_pending_buffer_memory = [None] * n_envs

    # Metrics
    q_losses = []
    mask_losses = []
    total_utilization = 0.0
    heuristic_counts = {k: 0 for k in heuristic_map.values()}
    total_decisions = 0
    total_valid_learned = 0
    total_valid_attempted = 0
    buffer_defer_count = 0
    buffer_place_after_count = 0

    while any(not d for d in slot_done) or next_ep_idx < n_episodes:
        # 1. Determine which environments need content this step
        step_envs_indices = []
        step_box_dims_list = []

        for i in range(n_envs):
            if slot_done[i]: continue

            # STREAM Phase
            if slot_phases[i] == 'STREAM':
                boxes = episodes_boxes[active_ep_indices[i]]
                if slot_boxes_ptr[i] < len(boxes):
                    box_dims = boxes[slot_boxes_ptr[i]]
                    slot_boxes_ptr[i] += 1
                    step_envs_indices.append(i)
                    step_box_dims_list.append(box_dims)
                    continue
                else:
                    slot_phases[i] = 'BUFFER_PASS'
                    slot_buffer_tried_count[i] = 0

            # BUFFER_PASS Phase
            if slot_phases[i] == 'BUFFER_PASS':
                if len(slot_buffers[i]) > 0:
                    if slot_buffer_tried_count[i] >= len(slot_buffers[i]):
                        slot_done[i] = True
                        continue
                    box_dims = slot_buffers[i].pop(0)
                    step_envs_indices.append(i)
                    step_box_dims_list.append(box_dims)
                else:
                    slot_done[i] = True

        if not step_envs_indices and next_ep_idx >= n_episodes:
            if all(slot_done): break
            continue

        if not step_envs_indices:
            break

        # 2. Worker: New Box Arrival
        for idx_in_batch, slot_idx in enumerate(step_envs_indices):
            envs.remotes[slot_idx].send(('new_box_arrival', step_box_dims_list[idx_in_batch]))

        step_states = []
        for slot_idx in step_envs_indices:
            s = envs.remotes[slot_idx].recv()
            slot_states[slot_idx] = s
            step_states.append(s)

        # 2b. Resolve pending buffer-defer memories with actual next state
        for slot_idx in step_envs_indices:
            if slot_pending_buffer_memory[slot_idx] is not None:
                ps, pa, pr, pbuf, pps = slot_pending_buffer_memory[slot_idx]
                agent.remember(ps, pa, pr, slot_states[slot_idx], False, pbuf, pps)
                slot_pending_buffer_memory[slot_idx] = None

        # 3. Agent: Mask Confidence & Proxy Scores (with buffer info)
        step_buffer_boxes = [list(slot_buffers[i]) for i in step_envs_indices]
        mask_probs_batch = agent.get_mask_confidence_batch(step_states, step_buffer_boxes)

        for idx, slot_idx in enumerate(step_envs_indices):
            envs.remotes[slot_idx].send(('get_proxy_scores', (mask_probs_batch[idx], step_buffer_boxes[idx], max_buffer_size)))

        step_proxy_scores = []
        for slot_idx in step_envs_indices:
            step_proxy_scores.append(envs.remotes[slot_idx].recv())

        # 4. Agent Selection (6 actions: 0-4 heuristics, 5 buffer defer)
        h_indices = agent.get_action_with_prior_batch(step_states, step_proxy_scores, step_buffer_boxes)
        for h_idx in h_indices:
            heuristic_counts[heuristic_map[h_idx]] += 1
            total_decisions += 1

        # 5. Execute Actions
        # For heuristic actions (0-4): choose placement via heuristic
        # For buffer action (5): defer to buffer
        step_actions = []
        for idx, slot_idx in enumerate(step_envs_indices):
            if h_indices[idx] == 5:
                # Buffer defer action - no heuristic needed
                step_actions.append(None)  # Sentinel for buffer defer
            else:
                heuristic = heuristic_map[h_indices[idx]]
                pred_mask_bool = mask_probs_batch[idx] > 0.5
                envs.remotes[slot_idx].send(('choose_action_by_heuristic', (heuristic, pred_mask_bool)))
                step_actions.append('PENDING')

        # Receive heuristic actions (only for non-buffer actions)
        for idx, slot_idx in enumerate(step_envs_indices):
            if step_actions[idx] == 'PENDING':
                action, _ = envs.remotes[slot_idx].recv()
                if action is None:
                    # Fallback: try without mask
                    heuristic = heuristic_map[h_indices[idx]]
                    envs.remotes[slot_idx].send(('choose_action_by_heuristic', (heuristic, None)))
                    action, _ = envs.remotes[slot_idx].recv()
                step_actions[idx] = action

        # 6. Step Processing
        for idx, slot_idx in enumerate(step_envs_indices):
            action = step_actions[idx]
            state = slot_states[slot_idx]
            h_idx = h_indices[idx]
            proxy_scores = step_proxy_scores[idx]
            box_dims = step_box_dims_list[idx]
            buffer_boxes = list(slot_buffers[slot_idx])

            if h_idx == 5:
                # --- Buffer Defer Action ---
                buffer_defer_count += 1

                if len(slot_buffers[slot_idx]) < max_buffer_size:
                    # Context-dependent reward based on placement quality
                    best_heuristic_score = max(proxy_scores[0:5])
                    buffer_fill_ratio = len(slot_buffers[slot_idx]) / max_buffer_size if max_buffer_size > 0 else 1.0
                    DEFER_THRESHOLD = 0.4

                    if best_heuristic_score < DEFER_THRESHOLD:
                        # Poor placements available -> reward deferring
                        quality_bonus = (DEFER_THRESHOLD - best_heuristic_score) / DEFER_THRESHOLD
                        reward = 0.02 * quality_bonus - 0.03 * buffer_fill_ratio
                    else:
                        # Good placements available -> penalize deferring
                        reward = -0.05 - 0.05 * best_heuristic_score - 0.05 * buffer_fill_ratio

                    slot_buffers[slot_idx].append(box_dims)
                    # Deferred memory: store pending, resolve with true next_state later
                    slot_pending_buffer_memory[slot_idx] = (state, h_idx, reward, buffer_boxes, proxy_scores)
                else:
                    # Buffer full when action 5 chosen
                    reward = -0.5
                    agent.remember(state, h_idx, reward, state, True, buffer_boxes, proxy_scores)
                    slot_done[slot_idx] = True

                if slot_phases[slot_idx] == 'BUFFER_PASS':
                    # Re-buffering during buffer pass: stop checking this slot
                    slot_buffer_tried_count[slot_idx] = len(slot_buffers[slot_idx])

            elif action is None:
                # Heuristic couldn't place: try to buffer, else episode over
                agent.remember(state, h_idx, -0.5, state, True, buffer_boxes, proxy_scores)
                if len(slot_buffers[slot_idx]) < max_buffer_size:
                    slot_buffers[slot_idx].append(box_dims)
                else:
                    slot_done[slot_idx] = True
                if slot_phases[slot_idx] == 'BUFFER_PASS':
                    slot_buffer_tried_count[slot_idx] += 1
            else:
                # --- Successful Heuristic Placement ---
                free_buf = max_buffer_size - len(slot_buffers[slot_idx])
                envs.remotes[slot_idx].send(('step', (action, free_buf, max_buffer_size)))
                next_state, reward, local_done, info = envs.remotes[slot_idx].recv()

                agent.remember(state, h_idx, reward, next_state, local_done, buffer_boxes, proxy_scores)

                if slot_phases[slot_idx] == 'BUFFER_PASS':
                    slot_buffer_tried_count[slot_idx] = 0
                if local_done:
                    slot_done[slot_idx] = True

                # --- Post-Placement Buffer Check (STREAM phase only) ---
                if slot_phases[slot_idx] == 'STREAM' and len(slot_buffers[slot_idx]) > 0 and not slot_done[slot_idx]:
                    buf_check_attempts = 0
                    max_buf_checks = max_buffer_size  # Bounded loop

                    while buf_check_attempts < max_buf_checks and len(slot_buffers[slot_idx]) > 0:
                        buf_check_attempts += 1
                        buf_box = slot_buffers[slot_idx][0]

                        # Quick placability check
                        envs.remotes[slot_idx].send(('can_place_box', buf_box))
                        can_place = envs.remotes[slot_idx].recv()

                        if not can_place:
                            break

                        # Place using agent (greedy, no exploration)
                        envs.remotes[slot_idx].send(('new_box_arrival', buf_box))
                        buf_state = envs.remotes[slot_idx].recv()

                        buf_buffer_boxes = list(slot_buffers[slot_idx][1:])  # Remaining buffer without current box

                        # Get mask & scores
                        buf_mask_probs = agent.get_mask_confidence_batch([buf_state], [buf_buffer_boxes])

                        envs.remotes[slot_idx].send(('get_proxy_scores', (buf_mask_probs[0], buf_buffer_boxes, max_buffer_size)))
                        buf_proxy_scores = envs.remotes[slot_idx].recv()

                        # Greedy action (no exploration)
                        orig_eps = agent.epsilon
                        agent.epsilon = 0.0
                        buf_h_idx = agent.get_action_with_prior(buf_state, buf_proxy_scores, buffer_boxes=buf_buffer_boxes)
                        agent.epsilon = orig_eps

                        if buf_h_idx == 5:
                            # Agent chose to re-buffer: stop checking
                            break

                        heuristic = heuristic_map[buf_h_idx]
                        buf_pred_mask_bool = buf_mask_probs[0] > 0.5
                        envs.remotes[slot_idx].send(('choose_action_by_heuristic', (heuristic, buf_pred_mask_bool)))
                        buf_action, _ = envs.remotes[slot_idx].recv()

                        if buf_action is None:
                            envs.remotes[slot_idx].send(('choose_action_by_heuristic', (heuristic, None)))
                            buf_action, _ = envs.remotes[slot_idx].recv()

                        if buf_action is not None:
                            free_buf_now = max_buffer_size - len(slot_buffers[slot_idx]) + 1  # +1 because we're placing from buffer
                            envs.remotes[slot_idx].send(('step', (buf_action, free_buf_now, max_buffer_size)))
                            buf_next_state, buf_reward, buf_done, _ = envs.remotes[slot_idx].recv()

                            # Bonus for successful buffer-to-placement (vindication of defer)
                            buf_reward += 0.03

                            # Record experience for the buffer placement
                            agent.remember(buf_state, buf_h_idx, buf_reward, buf_next_state, buf_done, buf_buffer_boxes, buf_proxy_scores)

                            slot_buffers[slot_idx].pop(0)
                            buffer_place_after_count += 1
                            heuristic_counts[heuristic_map[buf_h_idx]] += 1
                            total_decisions += 1

                            if buf_done:
                                slot_done[slot_idx] = True
                                break
                        else:
                            break

            # Target Update
            if total_decisions % 50 == 0:
                agent.update_target_model()

        # 7. Training Step
        agent.replay()
        if hasattr(agent, 'last_q_loss'): q_losses.append(agent.last_q_loss)
        if hasattr(agent, 'last_mask_loss'): mask_losses.append(agent.last_mask_loss)

        # 8. Episode Completions
        for i in range(n_envs):
            if slot_done[i] is True:
                # Flush any pending buffer-defer memory
                if slot_pending_buffer_memory[i] is not None:
                    ps, pa, pr, pbuf, pps = slot_pending_buffer_memory[i]
                    agent.remember(ps, pa, pr, slot_states[i] if slot_states[i] is not None else ps, True, pbuf, pps)
                    slot_pending_buffer_memory[i] = None

                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

                pbar.update(1)
                envs.remotes[i].send(('get_env_info', {'full': True}))
                info = envs.remotes[i].recv()
                total_valid_learned += info['invalid_learned']
                total_valid_attempted += info['invalid_attempted']

                pallet_vol = env_params.get('pallet_size', (10, 10))[0] * env_params.get('pallet_size', (10, 10))[1] * env_params.get('max_height', 10)
                if info['placed_boxes']:
                    placed_vol = sum(b[2] * b[3] * b[4] for b in info['placed_boxes'])
                    utilization = placed_vol / pallet_vol
                else:
                    utilization = 0.0
                total_utilization += utilization

                if next_ep_idx < n_episodes:
                    active_ep_indices[i] = next_ep_idx
                    next_ep_idx += 1
                    envs.remotes[i].send(('reset', None))
                    envs.remotes[i].recv()
                    slot_boxes_ptr[i] = 0
                    slot_buffers[i] = []
                    slot_phases[i] = 'STREAM'
                    slot_done[i] = False
                    slot_pending_buffer_memory[i] = None
                else:
                    slot_done[i] = None

    envs.close()
    pbar.close()

    # Restore batch size
    agent.batch_size = original_batch_size

    h_percs = {k: v / total_decisions if total_decisions > 0 else 0 for k, v in heuristic_counts.items()}

    return {
        'avg_util': total_utilization / n_episodes if n_episodes > 0 else 0,
        'avg_q_loss': np.mean(q_losses) if q_losses else 0.0,
        'avg_mask_loss': np.mean(mask_losses) if mask_losses else 0.0,
        'heuristics': h_percs,
        'epsilon': agent.epsilon,
        'invalid_learned': total_valid_learned,
        'invalid_attempted': total_valid_attempted,
        'buffer_defer_count': buffer_defer_count,
        'buffer_place_after_count': buffer_place_after_count
    }
