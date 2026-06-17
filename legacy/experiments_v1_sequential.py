import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
# Import from the OPTIMIZED file now
from oskp_rl_up_buffer_experiments_v1_sequential import train, DQNAgent, BoxPilingEnv, proxy_scores_for_heuristics

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
    heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'corner'}
    heuristic_counts = {k: 0 for k in heuristic_map.values()}
    total_decisions = 0
    total_invalid_learned = 0
    total_invalid_attempted = 0
    
    # Store original epsilon and set to greedy
    original_eps = agent.epsilon
    agent.epsilon = 0.0  
    
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
            # Refactored to pass state explicitly
            mask_bias = proxy_scores_for_heuristics(
                env.current_height_map, env.current_box, 
                env.pallet_size, env.max_height, pred_mask
            )
            
            h_idx = agent.get_action_with_prior(state, mask_bias, buffer_count=len(buffer)) 
            heuristic = heuristic_map[h_idx]
            heuristic_counts[heuristic] += 1
            total_decisions += 1
            
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
                else:
                    # Cannot place AND cannot buffer → end episode
                    done = True
                    break
            
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
                mask_bias = proxy_scores_for_heuristics(
                    env.current_height_map, env.current_box, 
                    env.pallet_size, env.max_height, pred_mask
                )
                h_idx = agent.get_action_with_prior(state, mask_bias, buffer_count=len(buffer))
                heuristic = heuristic_map[h_idx]
                heuristic_counts[heuristic] += 1
                total_decisions += 1
                
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
                    else:
                        # Buffer full and cannot place → end episode
                        done = True
                        break
                
                if env._is_terminal():
                    done = True; break
            
            buffer = new_buffer
            if not made_progress:
                break
        
        # Metrics
        total_invalid_learned += env.invalid_actions_learned
        total_invalid_attempted += env.invalid_actions_attempted
        
        pallet_volume = env.pallet_size[0] * env.pallet_size[1] * env.max_height
        placed_volume = sum(b[2] * b[3] * b[4] for b in env.placed_boxes)
        utilization = placed_volume / pallet_volume if pallet_volume > 0 else 0
        total_utilization += utilization
        
    # Restore epsilon
    agent.epsilon = original_eps
    
    avg_util = total_utilization / len(episodes_boxes)
    h_percs = {k: v / total_decisions if total_decisions > 0 else 0 for k, v in heuristic_counts.items()}
    
    return {
        'avg_util': avg_util, 
        'heuristics': h_percs,
        'invalid_learned': total_invalid_learned,
        'invalid_attempted': total_invalid_attempted
    }

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
        learning_rate=0.0001,
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
            min_support_ratio=0.60, # Ensure consistent physics
            require_opposite_edge_support=True
        )
        
        # Validate
        print(f"Evaluating Buffer Size {buf} on Test Sets...")
        env_params = {'max_buffer_size': buf, 'min_support_ratio': 0.60, 'require_opposite_edge_support': True}
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
            min_support_ratio=0.60,
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


def run_final_best_config_experiment(output_dir, train_episodes=10000, val_episodes=None):
    print("\n=== Experiment 8: Final Best Configuration (Buffer=2, LR=0.0001) ===")
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
    
    # Prepare Train Subset
    train_subset = train_data[:train_episodes] if train_episodes and train_episodes < len(train_data) else train_data
    import random
    train_subset = list(train_subset)
    random.shuffle(train_subset)

    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, "final_best_config_results.csv")

    print(f"\n--- Training with Buffer Size: 2, LR: 0.0001 ---")
    
    train_out_dir = os.path.join(output_dir, "train_best_config")
    os.makedirs(train_out_dir, exist_ok=True)
    
    metrics, agent = train(
        train_subset,
        output_dir=train_out_dir,
        return_agent=True,
        model_save_path=os.path.join(models_dir, "dqn_best_config_b2_lr0001.pt"),
        max_buffer_size=2,
        learning_rate=0.0001,
        min_support_ratio=0.60,
        require_opposite_edge_support=True
    )
    
    # Validate
    print(f"Evaluating Final Model on Test Sets...")
    env_params = {'max_buffer_size': 2, 'min_support_ratio': 0.50, 'require_opposite_edge_support': True}
    s_cut1 = evaluate(agent, val_cut1, env_params)
    s_cut2 = evaluate(agent, val_cut2, env_params)
    s_rs = evaluate(agent, val_rs, env_params)
    
    print(f"  Final Results -> CUT1: {s_cut1:.1%}, CUT2: {s_cut2:.1%}, RS: {s_rs:.1%}")
    
    result = {
        'buffer_size': 2,
        'lr': 0.0001,
        'cut1': s_cut1,
        'cut2': s_cut2,
        'rs': s_rs
    }
    pd.DataFrame([result]).to_csv(results_csv, index=False)
    
    print("\nExperiment 8 Final Summary:")
    print(result)

def grid_search(output_dir, train_episodes=10000, val_episodes=None):
    print("\n=== Experiment 9: Grid Search ===")
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
    buffer_sizes = [0, 1, 2, 3, 4]
    results = []
    
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, "exp9_grid_search_results.csv")

    for lr in lrs:
        for buffer_size in buffer_sizes:
            print(f"\n--- Testing LR: {lr} (Buffer Size: {buffer_size}) ---")
            run_name = f"lr_{lr}_buffer_{buffer_size}"
        
            train_out_dir = os.path.join(output_dir, f"train_{run_name}")
            os.makedirs(train_out_dir, exist_ok=True)
        
            metrics, agent = train(
                train_subset,
                output_dir=train_out_dir,
                return_agent=True,
                model_save_path=os.path.join(models_dir, f"dqn_{run_name}.pt"),
                learning_rate=lr,
                max_buffer_size=buffer_size,
                min_support_ratio=0.60,
                require_opposite_edge_support=True
            )
        
            # Validate
            print(f"Evaluating LR={lr} on Test Sets...")
            env_params = {'max_buffer_size': buffer_size, 'min_support_ratio': 0.60, 'require_opposite_edge_support': True}
            s_cut1 = evaluate(agent, val_cut1, env_params)
            s_cut2 = evaluate(agent, val_cut2, env_params)
            s_rs = evaluate(agent, val_rs, env_params)
            
            print(f"  LR={lr} (Buffer Size: {buffer_size}) -> CUT1: {s_cut1:.1%}, CUT2: {s_cut2:.1%}, RS: {s_rs:.1%}")
            results.append({
                'lr': lr,
                'buffer_size': buffer_size,
                'cut1': s_cut1,
                'cut2': s_cut2,
                'rs': s_rs
            })
            # Incremental Save
            pd.DataFrame(results).to_csv(results_csv, index=False)
        
    print("\nExperiment 9 Final Summary:")
    print(pd.DataFrame(results))
    

def run_epoch_training(output_dir, train_episodes=10000, val_episodes=None, patience=30, max_epochs=20, buffer_size=0):
    print(f"\n=== Experiment 10: Epoch-Based Training (Buffer={buffer_size}, LR=0.0001) ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data (Keep your existing loading code)
    train_data = load_instances("approachesO3DKP/ga_mixed_large.pt")
    val_cut1 = load_instances("approachesO3DKP/cut_1.pt")
    val_cut2 = load_instances("approachesO3DKP/cut_2.pt")
    val_rs = load_instances("approachesO3DKP/rs.pt")
    
    if val_episodes:
        val_cut1 = val_cut1[:val_episodes]
        val_cut2 = val_cut2[:val_episodes]
        val_rs = val_rs[:val_episodes]
    
    train_subset = list(train_data[:train_episodes]) if train_episodes else list(train_data)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Init Agent
    env = BoxPilingEnv()
    agent = DQNAgent(
        state_dims={'height_map': env.pallet_size, 'box_dims': 3},
        action_size=4,
        max_height=env.max_height,
        learning_rate=0.0001, # Start with 0.0001
        min_support_ratio=0.60,
        require_opposite_edge_support=True,
    )
    
    # FIX 3: Add Learning Rate Scheduler
    # Reduce LR by factor of 0.5 every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=5, gamma=0.5)

    # FIX 4: Reset Epsilon Logic
    # Start high, decay per EPISODE but very slowly to last across epochs
    agent.epsilon = 1.0 
    # Calculate decay to reach 0.1 after 50% of Total Epochs * Episodes
    # e.g., 20 epochs * 10k episodes = 200k steps. 
    # We want to explore for at least the first 5-8 epochs.
    # 0.99995 ^ 50000 approx 0.08
    agent.epsilon_decay = 0.999985 
    
    epoch_results = []
    best_avg_util = 0.0
    epochs_without_improvement = 0
    best_model_path = os.path.join(models_dir, "dqn_best_epoch.pt")
    results_csv = os.path.join(output_dir, "epoch_training_results.csv")
    
    env_params = {'max_buffer_size': buffer_size, 'min_support_ratio': 0.60, 'require_opposite_edge_support': True}
    
    for epoch in range(1, max_epochs + 1):
        print(f"\n{'='*50}")
        current_lr = scheduler.get_last_lr()[0]
        print(f"EPOCH {epoch}/{max_epochs} | LR: {current_lr:.6f} | Start Eps: {agent.epsilon:.4f}")
        print(f"{'='*50}")
        
        import random
        # Subsample N episodes from the large dataset (50k) for this epoch
        # This prevents overheating on the same instances
        train_subset = random.sample(train_data, min(len(train_data), train_episodes))
        random.shuffle(train_subset)
        
        train_out_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(train_out_dir, exist_ok=True)
        
        # Train one epoch
        metrics = train_one_epoch(agent, train_subset, train_out_dir, env_params)
        
        # Step the scheduler
        scheduler.step()
        
        # Validate
        print(f"\nValidating Epoch {epoch}...")
        v_cut1 = evaluate(agent, val_cut1, env_params)
        v_cut2 = evaluate(agent, val_cut2, env_params)
        v_rs = evaluate(agent, val_rs, env_params)
        
        s_cut1 = v_cut1['avg_util']
        s_cut2 = v_cut2['avg_util']
        s_rs = v_rs['avg_util']
        avg_util = (s_cut1 + s_cut2 + s_rs) / 3
        
        print(f"  Epoch {epoch} Results:")
        print(f"    CUT-1: {s_cut1:.2%}, CUT-2: {s_cut2:.2%}, RS: {s_rs:.2%}")
        print(f"    Avg Util: {avg_util:.2%} | End Eps: {agent.epsilon:.4f}")
        
        # Tracking data
        result_row = {
            'epoch': epoch,
            'train_util': metrics['avg_util'],
            'train_q_loss': metrics['avg_q_loss'],
            'train_mask_loss': metrics['avg_mask_loss'],
            'val_cut1_util': s_cut1,
            'val_cut2_util': s_cut2,
            'val_rs_util': s_rs,
            'avg_val_util': avg_util,
            'invalid_learned': metrics['invalid_learned'],
            'invalid_attempted': metrics['invalid_attempted'],
            'epsilon': agent.epsilon,
            'lr': current_lr
        }
        # Add heuristic data
        for k, v in metrics['heuristics'].items(): result_row[f'train_h_{k}'] = v
        for k, v in v_cut1['heuristics'].items(): result_row[f'val_cut1_h_{k}'] = v
        for k, v in v_cut2['heuristics'].items(): result_row[f'val_cut2_h_{k}'] = v
        for k, v in v_rs['heuristics'].items(): result_row[f'val_rs_h_{k}'] = v
        
        epoch_results.append(result_row)
        pd.DataFrame(epoch_results).to_csv(results_csv, index=False)
        
        # Save Visualizations (10 samples per set)
        print(f"Saving visualizations for Epoch {epoch}...")
        vis_epoch_dir = os.path.join(output_dir, "visualizations", f"epoch_{epoch}")
        save_visualizations(agent, val_cut1, os.path.join(vis_epoch_dir, "cut1"), env_params, n_samples=10)
        save_visualizations(agent, val_cut2, os.path.join(vis_epoch_dir, "cut2"), env_params, n_samples=10)
        save_visualizations(agent, val_rs, os.path.join(vis_epoch_dir, "rs"), env_params, n_samples=10)
        
        # Plot Progress
        plot_experiment_results(epoch_results, output_dir)
        
        # Save Best & Early Stopping
        if avg_util >= best_avg_util: 
            best_avg_util = avg_util
            agent.save_model(best_model_path)
            epochs_without_improvement = 0
            print(f"  ★ New best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"  No significant improvement ({epochs_without_improvement}/{patience})")
        
        if epochs_without_improvement >= patience:
            print("Early Stopping Reached.")
            break
            
    return epoch_results

def save_visualizations(agent, episodes_boxes, output_dir, env_params, n_samples=10):
    """Run agent on N samples and save 3D visualizations."""
    import random
    from oskp_rl_up_buffer_experiments_v1_sequential import BoxPilingEnv, proxy_scores_for_heuristics
    
    os.makedirs(output_dir, exist_ok=True)
    samples = random.sample(episodes_boxes, min(n_samples, len(episodes_boxes)))
    
    # Temporarily set epsilon to 0
    orig_eps = agent.epsilon
    agent.epsilon = 0.0
    
    env = BoxPilingEnv()
    max_buffer_size = env_params.get('max_buffer_size', 0)
    heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'corner'}

    for i, boxes in enumerate(samples):
        state = env.reset()
        done = False
        box_idx = 0
        buffer = []
        
        while not done and box_idx < len(boxes):
            box_dims = boxes[box_idx]
            box_idx += 1
            state = env.new_box_arrival(box_dims)
            pred_mask = agent.predict_mask(state, buffer_count=len(buffer))
            mask_bias = proxy_scores_for_heuristics(
                env.current_height_map, env.current_box, 
                env.pallet_size, env.max_height, pred_mask
            )
            h_idx = agent.get_action_with_prior(state, mask_bias, buffer_count=len(buffer))
            heuristic = heuristic_map[h_idx]
            action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=pred_mask)
            if action is None:
                action, _ = env.choose_action_by_heuristic(heuristic, pred_mask=None)
            
            if action is not None:
                next_state, _, _, _ = env.step(action)
                state = next_state
            else:
                if len(buffer) < max_buffer_size:
                    buffer.append(box_dims)
                else:
                    done = True; break
            if env._is_terminal():
                done = True; break
                
        # Buffer passes (simplified)
        while not done and len(buffer) > 0:
            made_progress = False
            for box in buffer[:]:
                state = env.new_box_arrival(box)
                # For buffer items, we still need the two-pass logic
                pred_mask = agent.predict_mask(state, buffer_count=len(buffer))
                mask_bias = proxy_scores_for_heuristics(
                    env.current_height_map, env.current_box, 
                    env.pallet_size, env.max_height, pred_mask
                )
                h_idx = agent.get_action_with_prior(state, mask_bias, buffer_count=len(buffer))
                action, _ = env.choose_action_by_heuristic(heuristic_map[h_idx], pred_mask=None)
                if action is not None:
                    env.step(action); buffer.remove(box); made_progress = True
            if not made_progress: break

        # Visualize
        env.visualize_pallet(
            episode_num=i+1,
            boxes_attempted=box_idx,
            utilization=env.current_height_map.sum() / (env.pallet_size[0] * env.pallet_size[1] * env.max_height),
            invalid_learned=env.invalid_actions_learned, 
            invalid_attempted=env.invalid_actions_attempted,
            output_dir=output_dir
        )
    
    agent.epsilon = orig_eps

def plot_experiment_results(results, output_dir):
    """Plot training metrics over epochs."""
    import matplotlib.pyplot as plt
    df = pd.DataFrame(results)
    if len(df) < 1: return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Utilization
    axes[0, 0].plot(df['epoch'], df['train_util'], label='Train Util', marker='o')
    axes[0, 0].plot(df['epoch'], df['avg_val_util'], label='Avg Val Util', marker='s')
    axes[0, 0].set_title('Utilization Trend')
    axes[0, 0].legend(); axes[0, 0].grid(True)
    
    # 2. Losses
    ax_loss = axes[0, 1]
    ax_mask = ax_loss.twinx()
    p1, = ax_loss.plot(df['epoch'], df['train_q_loss'], color='blue', label='Q-Loss', marker='.')
    p2, = ax_mask.plot(df['epoch'], df['train_mask_loss'], color='green', label='Mask-Loss', marker='x')
    ax_loss.set_title('Training Losses')
    ax_loss.set_ylabel('Q-Loss (MSE)', color='blue')
    ax_mask.set_ylabel('Mask-Loss (MSE)', color='green')
    ax_loss.legend(handles=[p1, p2]); axes[0, 1].grid(True)
    
    # 3. Epsilon & LR
    ax_eps = axes[1, 0]
    ax_lr = ax_eps.twinx()
    ax_eps.plot(df['epoch'], df['epsilon'], color='orange', label='Epsilon', marker='o')
    ax_lr.plot(df['epoch'], df['lr'], color='red', label='Learning Rate', linestyle='--')
    ax_eps.set_title('Scheduler & Entropy')
    ax_eps.legend(loc='upper left'); ax_lr.legend(loc='upper right')
    
    # 4. Heuristics (Train)
    h_cols = [c for c in df.columns if c.startswith('train_h_')]
    for col in h_cols:
        axes[1, 1].plot(df['epoch'], df[col], label=col.replace('train_h_', ''))
    axes[1, 1].set_title('Heuristic Usage (Train)')
    axes[1, 1].legend(); axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary_plots.png'))
    plt.close()

def train_one_epoch(agent, episodes_boxes, output_dir, env_params):
    """
    Train the agent for one pass through the dataset using ProNet + Maximal Space Bias.
    """
    from oskp_rl_up_buffer_experiments_v1_sequential import BoxPilingEnv, proxy_scores_for_heuristics
    from tqdm import tqdm
    
    env = BoxPilingEnv()
    # Beta is constant for this investigation
    BETA_VAL = 0.5 
    
    total_utilization = 0.0
    q_losses = []
    mask_losses = []
    
    heuristic_map = {0: 'stacking', 1: 'best_fit', 2: 'semi_perfect_fit', 3: 'corner'}
    heuristic_counts = {k: 0 for k in heuristic_map.values()}
    total_decisions = 0
    all_metrics = []
    
    pbar = tqdm(episodes_boxes, desc="Training", unit="ep")
    
    for episode_idx, boxes in enumerate(pbar):
        state = env.reset()
        done = False
        box_idx = 0
        buffer = []
        
        # --- STEP LOOP (Interacting with Environment) ---
        while not done and box_idx < len(boxes):
            box_dims = boxes[box_idx]
            box_idx += 1
            state = env.new_box_arrival(box_dims)
            
            # 1. Get Mask Probabilities
            mask_probs = agent.get_mask_confidence(state, len(buffer))
            
            # 2. Get Boolean Mask (Constraint)
            valid_mask_bool = mask_probs > 0.5
            
            # 3. Calculate Scores (Maximal Space Passthrough)
            mask_bias = proxy_scores_for_heuristics(
                env.current_height_map, env.current_box, 
                env.pallet_size, env.max_height, mask_probs
            )
            
            # 4. Act
            h_idx = agent.get_action_with_prior(state, mask_bias, len(buffer))
            heuristic = heuristic_map[h_idx]
            
            heuristic_counts[heuristic] += 1
            total_decisions += 1
            
            # 5. Choose Action
            action, mapping = env.choose_action_by_heuristic(heuristic, pred_mask=valid_mask_bool)
            
            # Fallback
            if action is None:
                action, mapping = env.choose_action_by_heuristic(heuristic, pred_mask=None)
                if action is None:
                    agent.remember(state, h_idx, -0.5, state, True, len(buffer), mask_bias)
                    agent.replay()
                    if hasattr(agent, 'last_q_loss'): q_losses.append(agent.last_q_loss)
                    if hasattr(agent, 'last_mask_loss'): mask_losses.append(agent.last_mask_loss)
                    
                    if len(buffer) < env_params.get('max_buffer_size', 0):
                        buffer.append(box_dims)
                    else:
                        done = True
                        break
                    continue
            
            # 6. Step & Learn
            next_state, reward, local_done, _ = env.step(action)
            agent.remember(state, h_idx, reward, next_state, local_done, len(buffer), mask_bias)
            agent.replay()
            
            if hasattr(agent, 'last_q_loss'): q_losses.append(agent.last_q_loss)
            if hasattr(agent, 'last_mask_loss'): mask_losses.append(agent.last_mask_loss)
            
            state = next_state
            if env._is_terminal(): done = True
            
            # Target Update (Frequency: per step count)
            if agent.optimizer_step_count % 2000 == 0:
                agent.update_target_model()
        
        # --- END OF EPISODE UPDATES ---
        
        # 1. Epsilon Decay (PER EPISODE)
        # 0.99995 ^ 50,000 approx 0.08 (Lasts ~5 epochs)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        # Metrics
        utilization = env.current_height_map.sum() / (env.pallet_size[0] * env.pallet_size[1] * env.max_height)
        total_utilization += utilization
        
        all_metrics.append({
            'invalid_learned': env.invalid_actions_learned,
            'invalid_attempted': env.invalid_actions_attempted
        })
        
        if (episode_idx + 1) % 100 == 0:
             learned = sum(m['invalid_learned'] for m in all_metrics[-100:])
             att = sum(m['invalid_attempted'] for m in all_metrics[-100:])
             pbar.set_postfix({
                 'Util': f'{utilization:.1%}', 
                 'Eps': f'{agent.epsilon:.4f}', 
                 'Q': f'{q_losses[-1]:.2f}' if q_losses else '0.0',
                 'Inv': f'{learned}/{att}'
             })

    h_percs = {k: v / total_decisions if total_decisions > 0 else 0 for k, v in heuristic_counts.items()}
    
    return {
        'avg_util': total_utilization / len(episodes_boxes),
        'avg_q_loss': np.mean(q_losses) if q_losses else 0.0,
        'avg_mask_loss': np.mean(mask_losses) if mask_losses else 0.0,
        'heuristics': h_percs,
        'epsilon': agent.epsilon,
        'invalid_learned': sum(m['invalid_learned'] for m in all_metrics),
        'invalid_attempted': sum(m['invalid_attempted'] for m in all_metrics)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[5, 6, 7, 8, 9, 10], default=10, help="Experiment number")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes per epoch")
    parser.add_argument("--val-episodes", type=int, default=None, help="Number of validation episodes")
    parser.add_argument("--max-epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--buffer-size", type=int, default=0, help="Buffer size for Experiment 10")
    args = parser.parse_args()
    
    base_output = "experiments_results_refactored"
    
    if args.exp == 5:
        run_buffer_comparison_v2(os.path.join(base_output, "exp5_buffer_comparison"), args.episodes, args.val_episodes)
    elif args.exp == 6:
        run_experiment_6(os.path.join(base_output, "exp6_buffer_lr_005"), args.episodes, args.val_episodes)
    elif args.exp == 7:
        run_experiment_7(os.path.join(base_output, "exp7_lr_nobuffer"), args.episodes, args.val_episodes)
    elif args.exp == 8:
        run_final_best_config_experiment(os.path.join(base_output, "exp8_final_best"), args.episodes, args.val_episodes)
    elif args.exp == 9:
        grid_search(os.path.join(base_output, "exp9_grid_search"), args.episodes, args.val_episodes)
    elif args.exp == 10:
        run_epoch_training(os.path.join(base_output, "exp10_epoch_training"), args.episodes, args.val_episodes, args.patience, args.max_epochs, args.buffer_size)
    else:
        print("Use --exp 5, 6, 7, 8, 9, or 10.")
