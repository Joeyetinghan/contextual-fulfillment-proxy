#!/usr/bin/env python3
"""
Generate a grid of hyperparameter combinations for delivery time DL grid search.
Limits to a max number of combinations by sampling from the grid.
"""
import itertools
import json
import numpy as np
from pathlib import Path
import src.config as cfg

def generate_grid_combinations(max_combinations=200, grid_name="delivery"):
    """
    Generate hyperparameter combinations for grid search.
    Ensures diverse sampling across the hyperparameter space.
    """
    if grid_name == "simulator":
        grid = cfg.DELIVERY_DL_SIMULATOR_HYPERPARAMETER_GRID
    else:
        grid = cfg.DELIVERY_DL_HYPERPARAMETER_GRID
    
    # Extract discrete and continuous parameters
    discrete_params = {}
    
    for key, value in grid.items():
        if isinstance(value, list):
            discrete_params[key] = value
        elif isinstance(value, tuple) and len(value) == 2:
            # For continuous params, sample values to get better coverage
            if key == "lr" or key == "weight_decay":
                # Log-uniform sampling
                low, high = value
                num_samples = 5
                samples = np.logspace(np.log10(low), np.log10(high), num_samples)
                discrete_params[key] = samples.tolist()
            else:
                # Uniform sampling for dropout_p
                low, high = value
                num_samples = 5
                samples = np.linspace(low, high, num_samples)
                discrete_params[key] = samples.tolist()
    
    # Generate all combinations
    param_names = list(discrete_params.keys())
    param_values = [discrete_params[name] for name in param_names]
    all_combinations = list(itertools.product(*param_values))
    
    print(f"Total possible combinations: {len(all_combinations)}")
    
    # If we have too many, sample (optionally biased)
    if len(all_combinations) > max_combinations:
        np.random.seed(cfg.RANDOM_SEED)
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        num_to_sample = min(max_combinations, len(all_combinations))

        weights = None
        if grid_name == "simulator":
            # Bias sampling toward higher-capacity / higher-lr regions observed at boundary.
            bias_power = 2.0
            bias_high = {"lr", "hidden_dim", "n_layers", "weight_decay"}
            bias_low = {"batch_size"}
            val_rank = {}
            for name, vals in discrete_params.items():
                val_rank[name] = {val: idx for idx, val in enumerate(vals)}
            w = []
            for combo in all_combinations:
                score = 1.0
                for name, val in zip(param_names, combo):
                    vals = discrete_params[name]
                    n = len(vals)
                    idx = val_rank[name].get(val, 0)
                    if name in bias_high:
                        base = (idx + 1) / n
                    elif name in bias_low:
                        base = (n - idx) / n
                    else:
                        base = 1.0
                    score *= base ** bias_power
                w.append(score)
            weights = np.asarray(w, dtype=np.float64)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = None

        indices = rng.choice(
            len(all_combinations),
            size=num_to_sample,
            replace=False,
            p=weights,
        )
        selected = [all_combinations[i] for i in indices]
        mode = "biased" if weights is not None else "random"
        print(f"{mode.title()} sampled {num_to_sample} combinations for diversity")
    else:
        selected = list(all_combinations)
    
    # Convert to list of dicts
    combinations = []
    for combo in selected:
        combo_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
        # Add fixed parameters
        combo_dict["epochs"] = cfg.DELIVERY_DL_EPOCHS
        combinations.append(combo_dict)
    
    return combinations, param_names

if __name__ == "__main__":
    import sys
    import argparse
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--max_combinations",
            type=int,
            default=200,
            help="Maximum number of hyperparameter combinations to generate.",
        )
        parser.add_argument(
            "--grid_name",
            choices=["delivery", "simulator"],
            default="delivery",
            help="Which grid to use (delivery or simulator).",
        )
        parser.add_argument(
            "--output_file",
            type=Path,
            default=Path("data/delivery_dl_grid_search_combinations.json"),
            help="Where to write the generated grid JSON.",
        )
        args = parser.parse_args()

        combinations, param_names = generate_grid_combinations(
            max_combinations=args.max_combinations, grid_name=args.grid_name
        )
        
        # Save to JSON file
        output_file = args.output_file
        if args.grid_name == "simulator":
            default_delivery = Path("data/delivery_dl_grid_search_combinations.json")
            if output_file == default_delivery:
                output_file = Path("data/delivery_dl_simulator_grid_search_combinations.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "combinations": combinations,
            "param_names": param_names,
            "num_combinations": len(combinations),
        }

        try:
            with open(output_file, "w") as f:
                json.dump(payload, f, indent=2)
        except PermissionError:
            fallback = Path("data/delivery_dl_simulator_grid_search_combinations.json")
            if output_file != fallback:
                fallback.parent.mkdir(parents=True, exist_ok=True)
                print(f"Permission denied for {output_file}. Falling back to {fallback}")
                with open(fallback, "w") as f:
                    json.dump(payload, f, indent=2)
                output_file = fallback
            else:
                raise
        
        print(f"\nGenerated {len(combinations)} hyperparameter combinations")
        print(f"Saved to {output_file}")
        
        print(f"\nFirst combination example:")
        for key, value in combinations[0].items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error generating grid: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

