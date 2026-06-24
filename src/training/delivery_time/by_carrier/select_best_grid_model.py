#!/usr/bin/env python3
"""
After grid search completes, find the best model for each carrier and copy it to the standard location
for evaluation.
"""
import json
import shutil
import torch
from pathlib import Path
import src.config as cfg
import pandas as pd

def find_best_models_per_carrier(tune_dir, train_on_proxy=False, grid_file=None):
    """Find the best model for each carrier from grid search results."""
    tune_dir = Path(tune_dir)
    mode_suffix = "_with_proxy" if train_on_proxy else ""
    
    # Load grid file to get hyperparameter values
    if grid_file is None:
        grid_file = Path("data/delivery_dl_grid_search_combinations.json")
    else:
        grid_file = Path(grid_file)
    
    grid_data = None
    if grid_file.exists():
        with open(grid_file, 'r') as f:
            grid_data = json.load(f)
        print(f"Loaded grid file: {grid_file}")
    else:
        print(f"Warning: Grid file not found at {grid_file}, will only print combo indices")
    
    # Load cost models to get expected carriers
    cost_models_df = pd.read_csv('data/params/real_cost_models_cs.csv')
    expected_carriers = sorted(cost_models_df['carrier_service_id'].dropna().astype(int).unique())
    
    best_models = {}
    
    for carrier_id in expected_carriers:
        # Find all model files for this carrier
        # Use a pattern that matches the mode suffix
        if mode_suffix:
            pattern = f"dl_model_*_combo_*{mode_suffix}.pt"
        else:
            # When no suffix, match files without _with_proxy
            pattern = "dl_model_*_combo_*.pt"
        model_files = list(tune_dir.glob(pattern))
        
        # Filter to this carrier and exclude wrong mode files
        carrier_models = []
        for model_file in model_files:
            # Exclude files with _with_proxy when we're looking for non-proxy models
            if not mode_suffix and "_with_proxy" in model_file.stem:
                continue
            # Exclude files without _with_proxy when we're looking for proxy models
            if mode_suffix and "_with_proxy" not in model_file.stem:
                continue
            
            # Extract carrier ID from filename (format: dl_model_{carrier_id}_combo_{idx}_with_proxy.pt)
            parts = model_file.stem.split('_')
            if len(parts) >= 3:
                try:
                    file_carrier_id = int(parts[2])
                    if file_carrier_id == carrier_id:
                        carrier_models.append(model_file)
                except ValueError:
                    continue
        
        if not carrier_models:
            print(f"  Carrier {carrier_id}: No models found")
            continue
        
        # Find best model for this carrier
        best_model_file = None
        best_val_loss = float('inf')
        best_combo_idx = None
        
        for model_file in carrier_models:
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                val_loss = checkpoint.get('best_val_loss')
                
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_file = model_file
                    best_combo_idx = checkpoint.get('combination_idx')
            except Exception as e:
                print(f"  Error loading {model_file}: {e}")
                continue
        
        if best_model_file:
            # Get hyperparameter values for this combo
            hp_values = None
            if grid_data and best_combo_idx is not None:
                combinations = grid_data.get("combinations", [])
                if 0 <= best_combo_idx < len(combinations):
                    hp_values = combinations[best_combo_idx]
            
            best_models[carrier_id] = {
                'model_file': best_model_file,
                'val_loss': best_val_loss,
                'combo_idx': best_combo_idx,
                'hp_values': hp_values
            }
            
            # Print combo index and hyperparameters
            print(f"  Carrier {carrier_id}: Best combo {best_combo_idx}, val_loss={best_val_loss:.6f}")
            if hp_values:
                print(f"    Hyperparameters:")
                for key, value in sorted(hp_values.items()):
                    if key != 'epochs':  # Skip epochs as it's less interesting
                        print(f"      {key}: {value}")
        else:
            print(f"  Carrier {carrier_id}: No valid models found")
    
    return best_models

def copy_best_models(best_models, tune_dir, target_dir, train_on_proxy=False):
    """Copy the best models to the standard location for evaluation."""
    tune_dir = Path(tune_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    mode_suffix = "_with_proxy" if train_on_proxy else ""
    
    from src.training.delivery_time.common import format_carrier_id_for_path
    
    copied_count = 0
    for carrier_id, best_info in best_models.items():
        source_path = best_info['model_file']
        carrier_id_str = format_carrier_id_for_path(carrier_id)
        target_path = target_dir / f"dl_model_{carrier_id_str}{mode_suffix}.pt"
        
        try:
            shutil.copy2(source_path, target_path)
            print(f"  Copied carrier {carrier_id} model to {target_path}")
            copied_count += 1
        except Exception as e:
            print(f"  Error copying carrier {carrier_id} model: {e}")
    
    print(f"\nSuccessfully copied {copied_count} models to {target_dir}")
    return copied_count

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_dir", type=Path, default=cfg.DELIVERY_MODELS_CS_DIR / "tune",
                       help="Directory containing grid search models")
    parser.add_argument("--target_dir", type=Path, default=cfg.DELIVERY_MODELS_CS_DIR / "tune",
                       help="Directory to copy best models to")
    parser.add_argument("--grid_file", type=Path, default=Path("data/delivery_dl_grid_search_combinations.json"),
                       help="Path to grid search combinations JSON file")
    parser.add_argument("--train_on_proxy", action="store_true",
                       help="Whether to select models trained with proxy data")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Finding best models from grid search results")
    print("=" * 80)
    print(f"Searching in: {args.tune_dir}")
    print(f"Target directory: {args.target_dir}")
    print()
    
    best_models = find_best_models_per_carrier(args.tune_dir, args.train_on_proxy, args.grid_file)
    
    if not best_models:
        print("\nNo best models found!")
        return
    
    print(f"\nFound best models for {len(best_models)} carriers")
    print("\nCopying best models to standard location...")
    copy_best_models(best_models, args.tune_dir, args.target_dir, args.train_on_proxy)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

if __name__ == "__main__":
    main()

