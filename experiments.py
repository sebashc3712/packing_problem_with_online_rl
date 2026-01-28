import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from oskp_rl_up_with_buffer_with_mask import train, DQNAgent, BoxPilingEnv

# ==========================================
# Evaluation Utility
# ==========================================
def evaluate(agent, episodes_boxes, env_params):
    # Separate params
    env_args = env_params.copy()
    max_buffer_size = env_args.pop('max_buffer_size', float('inf'))
    
    env = BoxPilingEnv(**env_args)
    
    total_utilization = 0.0
    total_placed = 0
    total_boxes = 0
    
    agent.epsilon = 0.0  # Greedy policy for evaluation
    
    for boxes in episodes_boxes:
        state = env.reset()
        done = False
        box_idx = 0
        buffer = [] # Buffer is part of the environment dynamics, but here we assume standard dynamics or agent-controlled?
        # Ideally the agent class or env should handle this, but the current code has the loop in 'train'.
        # We need to replicate the 'inference' loop.
        # Actually, the 'buffer' logic is in the 'train' loop in the original code.
        # So we need to replicate that logic here.
        # During inference, we usually use the same buffer logic.
        
        # NOTE: The agent was trained with a specific buffer behavior!
        # If we are evaluating, we should probably follow the same buffer rules (size).
        # We will assume infinite buffer for evaluation unless specified? 
        # Actually, for fairness, we should probably use the SAME buffer size as training or a standard one.
        # Let's assume standard infinite buffer (or large) for 'evaluation' unless we are testing buffer constraints.
        # BUT, if we trained with buffer=1, should we evaluate with buffer=1?
        # Usually yes. 
        # For Experiment 1 & 3: standard buffer (unlimited/large).
        # For Experiment 2: usage of buffer size `k` is the variable.
        # So `evaluate` should accept `max_buffer_size`.
        
        if env.current_box is None:
            # Maybe terminal or waiting for box?
            # actually new_box_arrival sets current_box. 
            pass

        def try_place_one_box(box_dims, current_state):
            pred_mask = agent.predict_mask(current_state)
            # For inference, we can just use the best heuristic (argmax)
            # But the agent returns an action index which corresponds to a heuristic.
            h_idx = agent.act_with_mask_bias(current_state, mask_bias=None) # No soft bias needed for eval usually, or use same?
            # The training code used:
            # mask_bias = proxy_scores_for_heuristics(...) 
            # h_idx = agent.act_with_mask_bias(..., mask_bias=mask_bias, ...)
            # We should probably replicate that for best performance if the agent relies on it.
            
            # Simple version: just ask agent.
            return h_idx

        # We need the full heuristic map and logic. 
        # It's better if we can reuse the loop logic. 
        # Since I can't easily import the loop body, I'll rewrite a simplified inference loop.
        
        heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'random_fit', 4: 'corner'}
        
        while not done and box_idx < len(boxes):
            box_dims = boxes[box_idx]
            box_idx += 1
            
            # Try place
            state = env.new_box_arrival(box_dims)
            pred_mask = agent.predict_mask(state)
            # Calculate mask bias (expensive but consistent with train)
            # To save time in eval, maybe skip if not strictly needed? 
            # But let's be consistent.
            # I'll skip mask_bias for speed in this script unless critical. The agent learned Q-values.
            h_idx = agent.act_with_mask_bias(state, mask_bias=None, beta=0.0) 
            heuristic = heuristic_map[h_idx]
            
            action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=pred_mask)
            if action is None:
                 action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=None)
            
            placed = False
            if action is not None:
                next_state, reward, done, _ = env.step(action)
                state = next_state
                placed = True
                
            if not placed:
                if len(buffer) < max_buffer_size:
                    buffer.append(box_dims)
            
            if env._is_terminal():
                done = True
                break
                
        # Buffer passes
        while not done and len(buffer) > 0:
            made_progress = False
            new_buffer = []
            for box_dims in buffer:
                if not env.can_place_box(box_dims):
                    if len(new_buffer) < max_buffer_size:
                        new_buffer.append(box_dims)
                    continue
                    
                state = env.new_box_arrival(box_dims)
                pred_mask = agent.predict_mask(state)
                h_idx = agent.act_with_mask_bias(state, mask_bias=None, beta=0.0)
                heuristic = heuristic_map[h_idx]
                
                action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=pred_mask)
                if action is None:
                    action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=None)
                
                if action is not None:
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    made_progress = True
                else:
                    if len(new_buffer) < max_buffer_size:
                        new_buffer.append(box_dims)
                
                if env._is_terminal():
                    done = True; break
            
            buffer = new_buffer
            if not made_progress:
                break
        
        # Metrics
        pallet_volume = env.pallet_size[0] * env.pallet_size[1] * env.max_height
        placed_volume = sum(b[2] * b[3] * b[4] for b in env.placed_boxes)
        utilization = placed_volume / pallet_volume if pallet_volume > 0 else 0
        total_utilization += utilization
        
    return total_utilization / len(episodes_boxes)

# ==========================================
# Experiment Runners
# ==========================================

def load_instances(path):
    print(f"Loading instances from {path}...")
    import pickle
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:
        # print(f"torch.load failed ({e}), trying pickle.load...")
        with open(path, 'rb') as f:
            return pickle.load(f)

def run_experiment_1(output_dir, train_episodes=2100, val_episodes=None):
    print("\n=== Experiment 1: Learning Strategy Confirmation ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    train_data = load_instances("approachesO3DKP/ga_mixed.pt")
    val_cut1 = load_instances("approachesO3DKP/cut_1.pt")
    val_cut2 = load_instances("approachesO3DKP/cut_2.pt")
    val_rs = load_instances("approachesO3DKP/rs.pt")
    
    if val_episodes:
        val_cut1 = val_cut1[:val_episodes]
        val_cut2 = val_cut2[:val_episodes]
        val_rs = val_rs[:val_episodes]
    
    # Train
    print(f"Training on artificial instances ({len(train_data)} available)...")
    # Using specific episodes count or full
    train_subset = train_data[:train_episodes] if train_episodes < len(train_data) else train_data
    # Shuffle to remove GA's fitness-based ordering bias (critical for fair training)
    import random
    train_subset = list(train_subset)
    random.shuffle(train_subset)
    
    train_out_dir = os.path.join(output_dir, "train_ga_mixed")
    os.makedirs(train_out_dir, exist_ok=True)
    metrics, agent = train(
        train_subset, 
        output_dir=train_out_dir,
        return_agent=True,
        # Default params
        learning_rate=0.001,
        max_buffer_size=float('inf')
    )
    
    # Validate
    print("Validating...")
    env_params = {'max_buffer_size': float('inf')}
    
    score_cut1 = evaluate(agent, val_cut1, env_params)
    score_cut2 = evaluate(agent, val_cut2, env_params)
    score_rs = evaluate(agent, val_rs, env_params)
    
    print(f"Validation Results (Utilization):")
    print(f"  CUT-1: {score_cut1:.2%}")
    print(f"  CUT-2: {score_cut2:.2%}")
    print(f"  RS:    {score_rs:.2%}")
    
    # Save results
    results = pd.DataFrame({
        'Dataset': ['CUT-1', 'CUT-2', 'RS'],
        'Utilization': [score_cut1, score_cut2, score_rs]
    })
    results.to_csv(os.path.join(output_dir, "validation_results.csv"), index=False)

def run_experiment_2(output_dir, train_episodes=2100, val_episodes=None):
    print("\n=== Experiment 2: Buffer Size Comparison ===")
    os.makedirs(output_dir, exist_ok=True)
    
    train_data = load_instances("approachesO3DKP/ga_mixed.pt")
    val_cut1 = load_instances("approachesO3DKP/cut_1.pt")
    val_cut2 = load_instances("approachesO3DKP/cut_2.pt")
    val_rs = load_instances("approachesO3DKP/rs.pt")

    if val_episodes:
        val_cut1 = val_cut1[:val_episodes]
        val_cut2 = val_cut2[:val_episodes]
        val_rs = val_rs[:val_episodes]
    
    buffer_sizes = [1, 2, 3, 4]
    results = []
    
    for buf in buffer_sizes:
        print(f"\n--- Testing Buffer Size: {buf} ---")
        run_name = f"buffer_{buf}"
        
        # Train
        train_subset = train_data[:train_episodes]
        # Shuffle to remove GA's fitness-based ordering bias
        import random
        train_subset = list(train_subset)
        random.shuffle(train_subset)
        
        train_out_dir = os.path.join(output_dir, f"train_{run_name}")
        os.makedirs(train_out_dir, exist_ok=True)
        metrics, agent = train(
            train_subset,
            output_dir=train_out_dir,
            return_agent=True,
            max_buffer_size=buf,
            learning_rate=0.001
        )
        
        # Validate (using SAME buffer size constraint)
        env_params = {'max_buffer_size': buf}
        s_cut1 = evaluate(agent, val_cut1, env_params)
        s_cut2 = evaluate(agent, val_cut2, env_params)
        s_rs = evaluate(agent, val_rs, env_params)
        
        print(f"  Buffer {buf} -> CUT-1: {s_cut1:.1%}, CUT-2: {s_cut2:.1%}, RS: {s_rs:.1%}")
        results.append({
            'buffer_size': buf,
            'cut1': s_cut1,
            'cut2': s_cut2,
            'rs': s_rs,
            'avg': (s_cut1 + s_cut2 + s_rs) / 3
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "buffer_comparison.csv"), index=False)
    print("\nBuffer Experiment Summary:")
    print(df)

def run_experiment_3(output_dir, train_episodes=2100, val_episodes=None):
    print("\n=== Experiment 3: Learning Rate Comparison ===")
    os.makedirs(output_dir, exist_ok=True)
    
    train_data = load_instances("approachesO3DKP/ga_mixed.pt")
    val_cut1 = load_instances("approachesO3DKP/cut_1.pt")
    val_cut2 = load_instances("approachesO3DKP/cut_2.pt")
    val_rs = load_instances("approachesO3DKP/rs.pt")

    if val_episodes:
        val_cut1 = val_cut1[:val_episodes]
        val_cut2 = val_cut2[:val_episodes]
        val_rs = val_rs[:val_episodes]

    lrs = [0.0001, 0.001, 0.005] 
    results = []
    
    for lr in lrs:
        print(f"\n--- Testing Learning Rate: {lr} ---")
        run_name = f"lr_{lr}"
        
        train_subset = train_data[:train_episodes]
        # Shuffle to remove GA's fitness-based ordering bias
        import random
        train_subset = list(train_subset)
        random.shuffle(train_subset)
        
        train_out_dir = os.path.join(output_dir, f"train_{run_name}")
        os.makedirs(train_out_dir, exist_ok=True)
        metrics, agent = train(
            train_subset,
            output_dir=train_out_dir,
            return_agent=True,
            learning_rate=lr,
            max_buffer_size=float('inf')
        )
        
        env_params = {'max_buffer_size': float('inf')}
        s_cut1 = evaluate(agent, val_cut1, env_params)
        s_cut2 = evaluate(agent, val_cut2, env_params)
        s_rs = evaluate(agent, val_rs, env_params)
        
        print(f"  LR {lr} -> CUT-1: {s_cut1:.1%}, CUT-2: {s_cut2:.1%}, RS: {s_rs:.1%}")
        results.append({
            'lr': lr,
            'cut1': s_cut1,
            'cut2': s_cut2,
            'rs': s_rs,
            'avg': (s_cut1 + s_cut2 + s_rs) / 3
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "lr_comparison.csv"), index=False)
    print("\nLearning Rate Experiment Summary:")
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], help="Experiment number (1, 2, or 3)")
    parser.add_argument("--episodes", type=int, default=2100, help="Number of training episodes")
    parser.add_argument("--val-episodes", type=int, default=None, help="Number of validation episodes")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()
    
    base_output = "experiments_results_v2"
    
    if args.all:
        run_experiment_1(os.path.join(base_output, "exp1"), args.episodes, args.val_episodes)
        run_experiment_2(os.path.join(base_output, "exp2"), args.episodes, args.val_episodes)
        run_experiment_3(os.path.join(base_output, "exp3"), args.episodes, args.val_episodes)
    elif args.exp == 1:
        run_experiment_1(os.path.join(base_output, "exp1"), args.episodes, args.val_episodes)
    elif args.exp == 2:
        run_experiment_2(os.path.join(base_output, "exp2"), args.episodes, args.val_episodes)
    elif args.exp == 3:
        run_experiment_3(os.path.join(base_output, "exp3"), args.episodes, args.val_episodes)
    else:
        print("Please specify --exp [1,2,3] or --all")
