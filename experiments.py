import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
# Import from the OPTIMIZED file now
from oskp_rl_up_buffer_experiments import train, DQNAgent, BoxPilingEnv, proxy_scores_for_heuristics

# ==========================================
# Evaluation Utility
# ==========================================
def evaluate(agent, episodes_boxes, env_params):
    # Separate params
    env_args = env_params.copy()
    max_buffer_size = env_args.pop('max_buffer_size', float('inf'))
    
    # Remove keys not accepted by BoxPilingEnv
    env_args.pop('min_support_ratio', None)
    env_args.pop('require_opposite_edge_support', None)
    
    env = BoxPilingEnv(**env_args)
    
    total_utilization = 0.0
    total_placed = 0
    total_boxes = 0
    
    # Store original epsilon and set to greedy
    original_eps = agent.epsilon
    agent.epsilon = 0.0  
    
    heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'random_fit', 4: 'corner'}

    for boxes in episodes_boxes:
        state = env.reset()
        done = False
        box_idx = 0
        buffer = [] 
        
        while not done and box_idx < len(boxes):
            box_dims = boxes[box_idx]
            box_idx += 1
            
            # --- Try Place One Box ---
            state = env.new_box_arrival(box_dims)
            # Use the agent to predict mask
            pred_mask = agent.predict_mask(state, buffer_count=len(buffer))
            
            # Use the proxy scores function from the module!
            mask_bias = proxy_scores_for_heuristics(env, pred_mask)
            
            h_idx = agent.act_with_mask_bias(state, buffer_count=len(buffer), mask_bias=mask_bias, beta=0.5) 
            heuristic = heuristic_map[h_idx]
            
            action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=pred_mask)
            if action is None:
                 action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=None)
            
            placed = False
            if action is not None:
                next_state, reward, local_done, _ = env.step(action)
                state = next_state
                placed = True
                
            if not placed:
                if len(buffer) < max_buffer_size:
                    buffer.append(box_dims)
            
            if env._is_terminal():
                done = True
                break
                
        # --- Buffer Passes ---
        while not done and len(buffer) > 0:
            made_progress = False
            new_buffer = []
            for box_dims in buffer:
                # Basic check if it fits at all before complex logic
                if not env.can_place_box(box_dims):
                    if len(new_buffer) < max_buffer_size:
                        new_buffer.append(box_dims)
                    continue
                    
                state = env.new_box_arrival(box_dims)
                pred_mask = agent.predict_mask(state, buffer_count=len(buffer))
                mask_bias = proxy_scores_for_heuristics(env, pred_mask)
                h_idx = agent.act_with_mask_bias(state, buffer_count=len(buffer), mask_bias=mask_bias, beta=0.5)
                heuristic = heuristic_map[h_idx]
                
                action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=pred_mask)
                if action is None:
                    action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=None)
                
                if action is not None:
                    next_state, reward, local_done, _ = env.step(action)
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
        
    # Restore epsilon
    agent.epsilon = original_eps
    return total_utilization / len(episodes_boxes)

# ==========================================
# Experiment Runners
# ==========================================

def load_instances(path):
    print(f"Loading instances from {path}...")
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:
        with open(path, 'rb') as f:
            return pickle.load(f)

def run_experiment_1(output_dir, train_episodes=10000, val_episodes=None):
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
    train_subset = train_data[:train_episodes] if train_episodes and train_episodes < len(train_data) else train_data
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

def run_experiment_2(output_dir, train_episodes=10000, val_episodes=None):
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
        train_subset = train_data[:train_episodes] if train_episodes and train_episodes < len(train_data) else train_data
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

def run_experiment_3(output_dir, train_episodes=10000, val_episodes=None):
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
        
        train_subset = train_data[:train_episodes] if train_episodes and train_episodes < len(train_data) else train_data
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

def run_paper_experiment(output_dir, train_episodes=10000, val_episodes=None):
    print("\n=== Paper Experiment: Buffer=3, LR=0.0001 ===")
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
    train_subset = train_data[:train_episodes] if train_episodes and train_episodes < len(train_data) else train_data
    import random
    train_subset = list(train_subset)
    random.shuffle(train_subset)
    
    train_out_dir = os.path.join(output_dir, "train_log")
    os.makedirs(train_out_dir, exist_ok=True)
    
    metrics, agent = train(
        train_subset, 
        output_dir=train_out_dir,
        return_agent=True,
        learning_rate=0.0001,
        max_buffer_size=3
    )
    
    # Validate
    print("Validating...")
    env_params = {'max_buffer_size': 3}
    
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
        'Utilization': [score_cut1, score_cut2, score_rs],
        'Config': ['Buffer=3, LR=0.0001', 'Buffer=3, LR=0.0001', 'Buffer=3, LR=0.0001']
    })
    results.to_csv(os.path.join(output_dir, "validation_results.csv"), index=False)

def run_buffer_comparison_v2(output_dir, train_episodes=10000, val_episodes=None):
    print("\n=== Experiment 5: Buffer vs No Buffer (Optimized) ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Real Data
    train_data = load_instances("approachesO3DKP/ga_mixed.pt")
    val_cut1 = load_instances("approachesO3DKP/cut_1.pt")
    val_cut2 = load_instances("approachesO3DKP/cut_2.pt")
    val_rs = load_instances("approachesO3DKP/rs.pt")
    
    if val_episodes:
        val_cut1 = val_cut1[:val_episodes]
        val_cut2 = val_cut2[:val_episodes]
        val_rs = val_rs[:val_episodes]
    
    # Prepare Train Subset
    train_subset = train_data[:train_episodes] if train_episodes and train_episodes < len(train_data) else train_data
    import random
    train_subset = list(train_subset)
    random.shuffle(train_subset)
    
    results = []
    
    # Define Configurations
    configs = [
        {"name": "No Buffer", "buffer_size": 0},
        {"name": "With Buffer", "buffer_size": float('inf')} # or 100
    ]
    
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    for cfg in configs:
        name = cfg['name']
        buf_size = cfg['buffer_size']
        print(f"\n--- Running: {name} ---")
        
        train_out_dir = os.path.join(output_dir, f"train_{name.lower().replace(' ', '_')}")
        os.makedirs(train_out_dir, exist_ok=True)
        
        # Train
        metrics, agent = train(
            train_subset,
            output_dir=train_out_dir,
            return_agent=True,
            model_save_path=os.path.join(models_dir, f"dqn_{name.lower().replace(' ', '_')}.pt"),
            max_buffer_size=buf_size,
            learning_rate=0.001,
            min_support_ratio=0.50,
            require_opposite_edge_support=True
        )
        # Save metrics
        metrics.to_csv(os.path.join(train_out_dir, "train_metrics.csv"), index=False)
        
        # Evaluate
        print(f"Evaluating {name} on Test Sets...")
        env_params = {'max_buffer_size': buf_size, 'min_support_ratio':0.50, 'require_opposite_edge_support':True}
        
        s_cut1 = evaluate(agent, val_cut1, env_params)
        s_cut2 = evaluate(agent, val_cut2, env_params)
        s_rs = evaluate(agent, val_rs, env_params)
        
        print(f"  {name} -> CUT-1: {s_cut1:.2%}, CUT-2: {s_cut2:.2%}, RS: {s_rs:.2%}")
        results.append({
            'Configuration': name,
            'Buffer Size': buf_size,
            'CUT-1': s_cut1,
            'CUT-2': s_cut2,
            'RS': s_rs
        })
        
    # Save Final Comparison
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "buffer_vs_nobuffer_results.csv"), index=False)
    print("\nFinal Results:")
    print(df)



def run_experiment_6(output_dir, train_episodes=10000, val_episodes=None):
    print("\n=== Experiment 6: Buffer Size Comparison (LR=0.005) ===")
    os.makedirs(output_dir, exist_ok=True)
    
    train_data = load_instances("approachesO3DKP/ga_mixed.pt")
    val_cut1 = load_instances("approachesO3DKP/cut_1.pt")
    val_cut2 = load_instances("approachesO3DKP/cut_2.pt")
    val_rs = load_instances("approachesO3DKP/rs.pt")

    if val_episodes:
        val_cut1 = val_cut1[:val_episodes]
        val_cut2 = val_cut2[:val_episodes]
        val_rs = val_rs[:val_episodes]
    
    # Prepare Train Subset
    train_subset = train_data[:train_episodes] if train_episodes and train_episodes < len(train_data) else train_data
    import random
    train_subset = list(train_subset)
    random.shuffle(train_subset)

    buffer_sizes = [0, 1, 2, 3, 4]
    results = []
    
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, "exp6_buffer_results.csv")

    for buf in buffer_sizes:
        print(f"\n--- Testing Buffer Size: {buf} (LR=0.005) ---")
        run_name = f"buffer_{buf}_lr_0.005"
        
        train_out_dir = os.path.join(output_dir, f"train_{run_name}")
        os.makedirs(train_out_dir, exist_ok=True)
        
        metrics, agent = train(
            train_subset,
            output_dir=train_out_dir,
            return_agent=True,
            model_save_path=os.path.join(models_dir, f"dqn_{run_name}.pt"),
            max_buffer_size=buf,
            learning_rate=0.005,
            min_support_ratio=0.50, # Ensure consistent physics
            require_opposite_edge_support=True
        )
        
        # Validate
        print(f"Evaluating Buffer Size {buf} on Test Sets...")
        env_params = {'max_buffer_size': buf, 'min_support_ratio': 0.50, 'require_opposite_edge_support': True}
        s_cut1 = evaluate(agent, val_cut1, env_params)
        s_cut2 = evaluate(agent, val_cut2, env_params)
        s_rs = evaluate(agent, val_rs, env_params)
        
        print(f"  Buf={buf} -> CUT1: {s_cut1:.1%}, CUT2: {s_cut2:.1%}, RS: {s_rs:.1%}")
        results.append({
            'buffer_size': buf,
            'lr': 0.005,
            'cut1': s_cut1,
            'cut2': s_cut2,
            'rs': s_rs
        })
        # Incremental Save
        pd.DataFrame(results).to_csv(results_csv, index=False)
    
    print("\nExperiment 6 Final Summary:")
    print(pd.DataFrame(results))

def run_experiment_7(output_dir, train_episodes=10000, val_episodes=None):
    print("\n=== Experiment 7: Learning Rate Comparison (No Buffer) ===")
    os.makedirs(output_dir, exist_ok=True)
    
    train_data = load_instances("approachesO3DKP/ga_mixed.pt")
    val_cut1 = load_instances("approachesO3DKP/cut_1.pt")
    val_cut2 = load_instances("approachesO3DKP/cut_2.pt")
    val_rs = load_instances("approachesO3DKP/rs.pt")

    if val_episodes:
        val_cut1 = val_cut1[:val_episodes]
        val_cut2 = val_cut2[:val_episodes]
        val_rs = val_rs[:val_episodes]

    # Prepare Train Subset
    train_subset = train_data[:train_episodes] if train_episodes and train_episodes < len(train_data) else train_data
    import random
    train_subset = list(train_subset)
    random.shuffle(train_subset)

    lrs = [0.0001, 0.001, 0.005] 
    results = []
    
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, "exp7_lr_results.csv")

    for lr in lrs:
        print(f"\n--- Testing LR: {lr} (No Buffer) ---")
        run_name = f"lr_{lr}_no_buffer"
        
        train_out_dir = os.path.join(output_dir, f"train_{run_name}")
        os.makedirs(train_out_dir, exist_ok=True)
        
        metrics, agent = train(
            train_subset,
            output_dir=train_out_dir,
            return_agent=True,
            model_save_path=os.path.join(models_dir, f"dqn_{run_name}.pt"),
            learning_rate=lr,
            max_buffer_size=0,
            min_support_ratio=0.50,
            require_opposite_edge_support=True
        )
        
        # Validate
        print(f"Evaluating LR={lr} on Test Sets...")
        env_params = {'max_buffer_size': 0, 'min_support_ratio': 0.50, 'require_opposite_edge_support': True}
        s_cut1 = evaluate(agent, val_cut1, env_params)
        s_cut2 = evaluate(agent, val_cut2, env_params)
        s_rs = evaluate(agent, val_rs, env_params)
        
        print(f"  LR={lr} -> CUT1: {s_cut1:.1%}, CUT2: {s_cut2:.1%}, RS: {s_rs:.1%}")
        results.append({
            'lr': lr,
            'buffer_size': 0,
            'cut1': s_cut1,
            'cut2': s_cut2,
            'rs': s_rs
        })
        # Incremental Save
        pd.DataFrame(results).to_csv(results_csv, index=False)
        
    print("\nExperiment 7 Final Summary:")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[5, 6, 7], default=5, help="Experiment number (5=Buf vs NoBuf, 6=Buf Sizes LR 0.005, 7=LRs NoBuf)")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--val-episodes", type=int, default=None, help="Number of validation episodes")
    args = parser.parse_args()
    
    base_output = "experiments_results_refactored"
    
    if args.exp == 5:
        run_buffer_comparison_v2(os.path.join(base_output, "exp5_buffer_comparison"), args.episodes, args.val_episodes)
    elif args.exp == 6:
        run_experiment_6(os.path.join(base_output, "exp6_buffer_lr_005"), args.episodes, args.val_episodes)
    elif args.exp == 7:
        run_experiment_7(os.path.join(base_output, "exp7_lr_nobuffer"), args.episodes, args.val_episodes)
    else:
        print("Use --exp 5, 6, or 7.")
