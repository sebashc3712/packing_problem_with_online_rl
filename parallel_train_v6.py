import numpy as np
import torch
from tqdm import tqdm
import os

def make_env():
    from oskp_rl_up_buffer_experiments_v6 import BoxPilingEnv
    return BoxPilingEnv()

def parallel_train_one_epoch(agent, episodes_boxes, output_dir, env_params, n_envs=8):
    from vec_env_v6 import SubprocVecEnv
    
    # --- Configuration ---
    max_buffer_size = env_params.get('max_buffer_size', 0)
    defer_penalty = -0.5 
    
    # OPTIMIZATION: Scale batch size to match sequential 'learning power' 
    # without running multiple slow updates.
    # Sequential: 1 step -> learn on 32. 
    # Parallel (8 envs): 8 steps -> learn on (32 * 8) = 256.
    original_batch_size = agent.batch_size
    agent.batch_size = 32 * n_envs 
    
    envs = SubprocVecEnv([make_env for _ in range(n_envs)])
    
    heuristic_map = {0: 'corner'}
    n_episodes = len(episodes_boxes)
    pbar = tqdm(total=n_episodes, desc="Parallel Training", unit="ep")
    
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
    
    # Metrics
    q_losses = []
    mask_losses = []
    total_utilization = 0.0
    heuristic_counts = {k: 0 for k in heuristic_map.values()}
    total_decisions = 0
    total_valid_learned = 0
    total_valid_attempted = 0
    
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
            
        # 3. Agent: Mask Confidence & Proxy Scores
        step_buffer_counts = [len(slot_buffers[i]) for i in step_envs_indices]
        mask_probs_batch = agent.get_mask_confidence_batch(step_states, step_buffer_counts)
        
        for idx, slot_idx in enumerate(step_envs_indices):
            envs.remotes[slot_idx].send(('get_proxy_scores', mask_probs_batch[idx]))
            
        step_proxy_scores = []
        for slot_idx in step_envs_indices:
            step_proxy_scores.append(envs.remotes[slot_idx].recv())
            
        # 4. Agent Selection & Execution
        h_indices = agent.get_action_with_prior_batch(step_states, step_proxy_scores, step_buffer_counts)
        for h_idx in h_indices:
            heuristic_counts[heuristic_map[h_idx]] += 1
            total_decisions += 1
        
        for idx, slot_idx in enumerate(step_envs_indices):
            heuristic = heuristic_map[h_indices[idx]]
            pred_mask_bool = mask_probs_batch[idx] > 0.5
            envs.remotes[slot_idx].send(('choose_action_by_heuristic', (heuristic, pred_mask_bool)))
            
        step_actions = []
        for idx, slot_idx in enumerate(step_envs_indices):
            action, _ = envs.remotes[slot_idx].recv()
            if action is None:
                # Fallback
                envs.remotes[slot_idx].send(('choose_action_by_heuristic', (heuristic_map[h_indices[idx]], None)))
                action, _ = envs.remotes[slot_idx].recv()
            step_actions.append(action)
            
        # 5. Step Processing
        for idx, slot_idx in enumerate(step_envs_indices):
            action = step_actions[idx]
            state = slot_states[slot_idx]
            h_idx = h_indices[idx]
            proxy_scores = step_proxy_scores[idx]
            box_dims = step_box_dims_list[idx]
            
            if action is None:
                agent.remember(state, h_idx, defer_penalty, state, True, step_buffer_counts[idx], proxy_scores)
                if len(slot_buffers[slot_idx]) < max_buffer_size:
                    slot_buffers[slot_idx].append(box_dims)
                else:
                    slot_done[slot_idx] = True
                if slot_phases[slot_idx] == 'BUFFER_PASS':
                    slot_buffer_tried_count[slot_idx] += 1
            else:
                envs.remotes[slot_idx].send(('step', action))
                next_state, reward, local_done, info = envs.remotes[slot_idx].recv()
                
                agent.remember(state, h_idx, reward, next_state, local_done, step_buffer_counts[idx], proxy_scores)
                
                if slot_phases[slot_idx] == 'BUFFER_PASS':
                    slot_buffer_tried_count[slot_idx] = 0
                if local_done:
                    slot_done[slot_idx] = True
            
            # Fix Target Update Frequency
            # Sequential updates every episode (~50 steps). 2000 was too slow.
            if total_decisions % 50 == 0:
                agent.update_target_model()
        
        # 6. Training Step (Vectorized)
        # Train once per parallel batch, but with the scaled batch size.
        agent.replay()
        if hasattr(agent, 'last_q_loss'): q_losses.append(agent.last_q_loss)
        if hasattr(agent, 'last_mask_loss'): mask_losses.append(agent.last_mask_loss)

        # 7. Episode Completions
        for i in range(n_envs):
            if slot_done[i] is True:
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
                else:
                    slot_done[i] = None
    
    envs.close()
    pbar.close()
    
    # Restore batch size at end to avoid side effects
    agent.batch_size = original_batch_size 
    
    h_percs = {k: v / total_decisions if total_decisions > 0 else 0 for k, v in heuristic_counts.items()}
    
    return {
        'avg_util': total_utilization / n_episodes if n_episodes > 0 else 0,
        'avg_q_loss': np.mean(q_losses) if q_losses else 0.0,
        'avg_mask_loss': np.mean(mask_losses) if mask_losses else 0.0,
        'heuristics': h_percs,
        'epsilon': agent.epsilon,
        'invalid_learned': total_valid_learned,
        'invalid_attempted': total_valid_attempted
    }