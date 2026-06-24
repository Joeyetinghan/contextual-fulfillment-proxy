#!/usr/bin/env python3
"""
After grid search completes, find the best model and copy it to the standard location
for evaluation.
"""
import json
import shutil
from pathlib import Path
import src.config as cfg

def find_best_model(models_dir, train_on_proxy=False):
    """Find the best model from grid search results."""
    models_dir = Path(models_dir)
    
    # Find all result files - filter by training mode
    if train_on_proxy:
        # Look for files with _with_proxy suffix
        result_files = sorted(models_dir.glob("grid_results_with_proxy_*.json"))
    else:
        # Look for files without _with_proxy suffix (but not the old ones we renamed)
        all_files = sorted(models_dir.glob("grid_results_*.json"))
        result_files = [f for f in all_files if "_with_proxy" not in f.name]
    
    if not result_files:
        print("No grid search results found!")
        return None
    
    best_result = None
    best_loss = float('inf')
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            # Use validation loss if available, otherwise test loss
            loss = result.get("validation_loss")
            if loss is None:
                loss = result.get("test_loss")
            
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_result = result
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
            continue
    
    if best_result is None:
        print("No valid results found!")
        return None
    
    print(f"Best model found:")
    print(f"  Combination index: {best_result['combination_idx']}")
    print(f"  Validation loss: {best_result.get('validation_loss', 'N/A')}")
    print(f"  Test loss: {best_result.get('test_loss', 'N/A')}")
    print(f"  Model path: {best_result['model_path']}")
    print(f"\nHyperparameters:")
    for key, value in best_result['hyperparameters'].items():
        print(f"  {key}: {value}")
    
    return best_result

def copy_best_model(best_result, models_dir, train_on_proxy=False):
    """Copy the best model and scalers to the standard location for evaluation."""
    models_dir = Path(models_dir)
    source_path = Path(best_result['model_path'])
    source_scalers_path = Path(best_result.get('scalers_path', ''))
    
    if not source_path.exists():
        print(f"Error: Source model file not found: {source_path}")
        return False
    
    # Determine target filenames (save in tune/ subdirectory)
    train_mode = "_with_proxy" if train_on_proxy else ""
    target_model_filename = f"mqrnn_model{train_mode}_tuned.pt"
    target_scalers_filename = f"mqrnn_scalers{train_mode}_tuned.pt"
    # Save in tune/ subdirectory (same directory as grid search results)
    target_model_path = models_dir / target_model_filename
    target_scalers_path = models_dir / target_scalers_filename
    
    # Copy the model
    print(f"\nCopying best model to {target_model_path}")
    shutil.copy2(source_path, target_model_path)
    
    # Copy scalers if they exist
    if source_scalers_path and source_scalers_path.exists():
        print(f"Copying scalers to {target_scalers_path}")
        shutil.copy2(source_scalers_path, target_scalers_path)
    else:
        print(f"Warning: Scalers file not found at {source_scalers_path}")
    
    print(f"Successfully copied best model and scalers!")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=Path, default=cfg.DEMAND_MODELS_DIR / "tune",
                       help="Directory containing grid search models and results")
    parser.add_argument("--train_on_proxy", action="store_true",
                       help="Whether to select model trained with proxy data")
    
    args = parser.parse_args()
    
    best_result = find_best_model(args.models_dir, args.train_on_proxy)
    
    if best_result:
        copy_best_model(best_result, args.models_dir, args.train_on_proxy)
    else:
        print("Failed to find best model")

if __name__ == "__main__":
    main()

