#!/usr/bin/env python3
"""
Generate a grid of hyperparameter combinations for grid search.
Limits to max 50 combinations by intelligently sampling from the grid.
"""
import itertools
import json
import numpy as np
from pathlib import Path
import src.config as cfg

def generate_grid_combinations(max_combinations=50):
    """
    Generate hyperparameter combinations for grid search.
    Ensures the best known combination is included, then samples diversely.
    """
    grid = cfg.DEMAND_MODEL_HYPERPARAMETER_GRID
    
    # Best known hyperparameters from previous tuning (achieved pinball loss 0.229367)
    best_known = {
        "lr": 8.891397050194612e-05,
        "hidden_dim": 384,
        "lstm_n_layers": 3,
        "dropout_p": 0.75,
        "weight_decay": 0.0001,
        "batch_size": 16,
        "context_dim": 12,
        "sku_emb": 8,
        "brand_emb": 16,
        "early_stopping_patience": 8
    }
    
    # Extract discrete and continuous parameters
    discrete_params = {}
    
    for key, value in grid.items():
        if isinstance(value, list):
            discrete_params[key] = value
        elif isinstance(value, tuple) and len(value) == 2:
            # For continuous params, sample more values to get better coverage
            if key == "lr" or key == "weight_decay":
                # Log-uniform sampling - use more samples
                low, high = value
                num_samples = 5  # Increased from 3
                samples = np.logspace(np.log10(low), np.log10(high), num_samples)
                # Ensure best known value is included if in range
                if key in best_known:
                    best_val = best_known[key]
                    if low <= best_val <= high:
                        # Replace closest sample with best value
                        closest_idx = np.argmin(np.abs(samples - best_val))
                        samples[closest_idx] = best_val
                discrete_params[key] = samples.tolist()
            else:
                # Uniform sampling for dropout_p
                low, high = value
                num_samples = 5  # Increased from 3
                samples = np.linspace(low, high, num_samples)
                # Ensure best known value is included if in range
                if key in best_known:
                    best_val = best_known[key]
                    if low <= best_val <= high:
                        # Replace closest sample with best value
                        closest_idx = np.argmin(np.abs(samples - best_val))
                        samples[closest_idx] = best_val
                discrete_params[key] = samples.tolist()
    
    # Generate all combinations
    param_names = list(discrete_params.keys())
    param_values = [discrete_params[name] for name in param_names]
    all_combinations = list(itertools.product(*param_values))
    
    print(f"Total possible combinations: {len(all_combinations)}")
    
    # Always include the best known combination first
    best_combo = tuple(best_known.get(name, discrete_params[name][0]) 
                       if name in best_known else discrete_params[name][0]
                       for name in param_names)
    
    # If we have too many, sample diversely
    if len(all_combinations) > max_combinations:
        selected = []
        remaining_combos = list(all_combinations)
        
        # 1. Always include best known combination if it exists
        if best_combo in remaining_combos:
            selected.append(best_combo)
            remaining_combos.remove(best_combo)
            print(f"Included best known combination")
        
        # 2. Sample remaining combinations randomly for diversity
        remaining_needed = max_combinations - len(selected)
        if remaining_needed > 0 and len(remaining_combos) > 0:
            # Use random sampling for diversity
            np.random.seed(cfg.RANDOM_SEED)
            num_to_sample = min(remaining_needed, len(remaining_combos))
            indices = np.random.choice(len(remaining_combos), 
                                      size=num_to_sample,
                                      replace=False)
            selected.extend([remaining_combos[i] for i in indices])
            print(f"Randomly sampled {num_to_sample} additional combinations for diversity")
    else:
        selected = list(all_combinations)
        # Ensure best is included even if we have fewer than max
        if best_combo not in selected:
            selected.insert(0, best_combo)
            print(f"Included best known combination")
    
    # Convert to list of dicts
    combinations = []
    for combo in selected:
        combo_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
        # Add fixed parameters
        combo_dict["epochs"] = cfg.DEMAND_MODEL_EPOCHS
        combinations.append(combo_dict)
    
    return combinations, param_names

if __name__ == "__main__":
    import sys
    try:
        combinations, param_names = generate_grid_combinations(max_combinations=50)
        
        # Save to JSON file
        output_file = Path("data/demand_grid_search_combinations.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                "combinations": combinations,
                "param_names": param_names,
                "num_combinations": len(combinations)
            }, f, indent=2)
        
        print(f"\nGenerated {len(combinations)} hyperparameter combinations")
        print(f"Saved to {output_file}")
    except Exception as e:
        print(f"Error generating grid: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check if best known combination is included
    best_known_dict = {
        "lr": 8.891397050194612e-05,
        "hidden_dim": 384,
        "lstm_n_layers": 3,
        "dropout_p": 0.75,
        "weight_decay": 0.0001,
        "batch_size": 16,
        "context_dim": 12,
        "sku_emb": 8,
        "brand_emb": 16,
        "early_stopping_patience": 8
    }
    
    best_included = False
    for i, combo in enumerate(combinations):
        matches = all(combo.get(k) == v for k, v in best_known_dict.items() if k in combo)
        if matches:
            print(f"\n✓ Best known combination found at index {i}")
            best_included = True
            break
    
    if not best_included:
        print(f"\n⚠ Warning: Best known combination not exactly found (may be close)")
    
    print(f"\nFirst combination example:")
    for key, value in combinations[0].items():
        print(f"  {key}: {value}")

