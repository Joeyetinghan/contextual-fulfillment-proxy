"""
Train Delivery Time Simulators by Carrier Service

This script trains separate simulator models for each carrier service to simulate 
delivery times. Supports both CatBoost classifier and Deep Learning quantile models.

It uses the preprocessed_data_cs.csv file which includes carrier_service_id_anon column.

The models are saved to data/models/delivery_time_cs/ directory with naming:
    simulator_{simulator_type}_{carrier_service_id}.joblib
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import joblib
import argparse
import numpy as np
import torch
from pathlib import Path

import src.config as cfg

# Import common utilities
from src.training.delivery_time.common import (
    plot_distribution,
    prepare_categorical_encoders,
    load_split_data,
    create_delivery_time_features,
    format_carrier_id_for_path,
)

# Import simulator training functions
from src.training.delivery_time._train_simulator import train_catboost_simulator, train_dl_simulator


def main(args):
    """Main function to train carrier-specific delivery time simulators."""
    print("=" * 80)
    print("Training Carrier-Specific Delivery Time Simulators")
    print("=" * 80)
    
    # Create output directories
    models_dir = cfg.DELIVERY_TIME_SIMULATOR_PATH
    plots_dir = Path(cfg.DATA_DIR) / 'plots' / 'delivery_time_cs' / 'simulator'
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw preprocessed data with carrier services and engineer features
    # For simulators, we want to use TEST data (to approximate true environment)
    print("\nLoading preprocessed_data_cs.csv and engineering features...")
    df = pd.read_csv('data/processed/preprocessed_data_cs.csv', 
                    parse_dates=['order_time', 'order_date'])
    df = df[df['order_date'] > cfg.PROXY_TRAIN_END_DATE].copy()
    
    print(f"  Loaded {len(df):,} test records")
    
    # Sort by order_time for rolling features
    df.sort_values(by='order_time', inplace=True)
    
    # Engineer all features properly
    print("\nEngineering delivery time features...")
    df = create_delivery_time_features(df)
    
    # Get list of carrier services
    carrier_services = sorted(df['carrier_service_id_anon'].dropna().unique())
    print(f"\nFound {len(carrier_services)} carrier services in test data: {carrier_services}")
    
    # Load cost models to ensure we train for all expected carriers
    cost_models_df = pd.read_csv('data/params/real_cost_models_cs.csv')
    expected_carriers = sorted(cost_models_df['carrier_service_id'].dropna().astype(int).unique())
    print(f"Expected carrier services from cost models: {expected_carriers}")
    
    # Filter to only train on carriers that exist in both
    carriers_to_train = sorted(set(carrier_services) & set(expected_carriers))
    if len(carriers_to_train) < len(expected_carriers):
        missing = set(expected_carriers) - set(carriers_to_train)
        print(f"  Warning: Missing carrier services in data: {missing}")
    
    # If a specific carrier_id is provided, restrict to that one
    if args.carrier_id is not None:
        if args.carrier_id in carriers_to_train:
            carriers_to_train = [args.carrier_id]
        else:
            print(f"Requested carrier_id {args.carrier_id} not in available set; skipping.")
            carriers_to_train = []

    print(f"\nWill train simulators for {len(carriers_to_train)} carrier services")
    
    # Train simulators for each carrier service
    for carrier_id in carriers_to_train:
        print(f"\n{'='*80}")
        print(f"Training simulator for Carrier Service {carrier_id}")
        print(f"{'='*80}")
        
        # Filter data for this carrier service
        df_carrier = df[df['carrier_service_id_anon'] == carrier_id].copy()
        
        print(f"  Samples: {len(df_carrier):,}")
        
        if len(df_carrier) < 100:
            print(f"  Skipping - insufficient data (< 100 samples)")
            continue
        
        try:
            carrier_id_str = format_carrier_id_for_path(carrier_id)
            # Train simulator
            if args.simulator_type == 'catboost':
                calibrated_clf, label_encoder = train_catboost_simulator(
                    df_carrier, n_trials=args.n_trials, carrier_id=carrier_id
                )
            elif args.simulator_type == 'dl':
                # Note: DL sim is global, but we can train it for each carrier subset for comparison
                model, x_scaler, encoders, vocab_sizes, final_params, _, _, _ = train_dl_simulator(
                    df_carrier, n_trials=args.n_trials
                )
                calibrated_clf = model # for saving
                label_encoder = encoders # for saving
            else:
                raise ValueError(f"Unknown simulator type: {args.simulator_type}")

            
            if calibrated_clf is None:
                continue
            
            # Save model
            model_path = models_dir / f"simulator_{args.simulator_type}_{carrier_id_str}.joblib"
            if args.simulator_type == 'dl':
                 bundle_path = models_dir / f"simulator_dl_{carrier_id_str}.pt"
                 torch.save({
                    'state_dict': calibrated_clf.state_dict(),
                    'x_scaler': x_scaler,
                    'categorical_encoders': label_encoder,
                    'vocab_sizes': vocab_sizes,
                    'numerical_dim': x_scaler.mean_.shape[0],
                    'hidden_dim': final_params['hidden_dim'],
                    'n_layers': final_params['n_layers'],
                    'dropout': True,
                    'dropout_p': final_params['dropout_p'],
                    'dc_ori_embedding_dim': cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM,
                    'dc_des_embedding_dim': cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM,
                 }, bundle_path)
                 joblib.dump({'type': 'dl', 'dl_model_path': str(bundle_path), 'carrier_service_id': carrier_id}, model_path)
            else:
                joblib.dump({
                    'model': calibrated_clf,
                    'label_encoder': label_encoder,
                    'carrier_service_id': carrier_id,
                }, model_path)
            print(f"  Saved simulator to {model_path}")
            
            # Generate distribution plot
            try:
                print(f"  Generating distribution plot...")
                features = cfg.DELIVERY_TIME_FEATURES
                X = df_carrier[features].fillna(0)
                y = df_carrier['delivery_time_days']
                
                if args.simulator_type == 'catboost':
                    # CatBoost classifier: use predicted probabilities
                    valid_classes = label_encoder.classes_
                    valid_classes_decoded = label_encoder.inverse_transform(valid_classes)
                    
                    # True distribution
                    classes_sorted = np.array(sorted(valid_classes_decoded))
                    true_dist = np.array([(y == c).mean() for c in classes_sorted])
                    
                    # Predicted distribution
                    y_prob = calibrated_clf.predict_proba(X)
                    clf_classes = label_encoder.inverse_transform(calibrated_clf.classes_)
                    pred_dist = np.zeros_like(classes_sorted, dtype=float)
                    for i, c in enumerate(classes_sorted):
                        if c in clf_classes:
                            idx = np.where(clf_classes == c)[0][0]
                            pred_dist[i] = y_prob[:, idx].mean()
                    
                else:  # DL simulator
                    # DL quantile model: sample from quantiles to get distribution
                    from src.utils import sample_paths
                    
                    # Prepare features for DL model
                    X_num = X[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
                    X_num_scaled = x_scaler.transform(X_num) if x_scaler is not None else X_num.values
                    X_num_tensor = np.asarray(X_num_scaled, dtype=np.float32)
                    dc_ori_enc = encoders['dc_ori'].transform(X['dc_ori'].astype(str))
                    dc_des_enc = encoders['dc_des'].transform(X['dc_des'].astype(str))
                    
                    # Predict quantiles
                    calibrated_clf.eval()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model_device = calibrated_clf.to(device)
                    preds = np.zeros((len(X), len(cfg.DELIVERY_TIME_QUANTILES)), dtype=np.float32)
                    batch_size = 2048
                    with torch.no_grad():
                        for s in range(0, len(X), batch_size):
                            e = min(len(X), s + batch_size)
                            Xb = torch.tensor(X_num_tensor[s:e], dtype=torch.float32, device=device)
                            dc_o = torch.tensor(dc_ori_enc[s:e], dtype=torch.long, device=device)
                            dc_d = torch.tensor(dc_des_enc[s:e], dtype=torch.long, device=device)
                            preds[s:e] = model_device(Xb, dc_o, dc_d).cpu().numpy()
                    
                    # Sample from quantiles and aggregate across observations
                    rng = np.random.default_rng(cfg.RANDOM_SEED)
                    n_samples_per_obs = 100  # Reduced for computational efficiency
                    all_samples_flat = []
                    for i in range(len(preds)):
                        samples = sample_paths(preds[i], n_samples_per_obs, rng=rng, lower=0, upper=5)
                        all_samples_flat.extend(np.rint(samples).astype(int))
                    all_samples_flat = np.array(all_samples_flat)
                    
                    # True and predicted distributions
                    # For fair comparison, compute mean predicted value per class using all samples
                    y_valid = y.dropna().values
                    if len(y_valid) == 0 or len(all_samples_flat) == 0:
                        raise ValueError("No valid samples available to plot DL distribution.")
                    y_classes = np.rint(y_valid).astype(int)
                    pred_classes = all_samples_flat.astype(int)
                    min_class = min(y_classes.min(), pred_classes.min())
                    max_class = max(y_classes.max(), pred_classes.max())
                    classes_sorted = np.arange(min_class, max_class + 1)
                    true_dist = np.array([(y_classes == c).mean() for c in classes_sorted])
                    pred_dist = np.array([(pred_classes == c).mean() for c in classes_sorted])
                
                # Plot
                save_path = plots_dir / f"dist_comparison_carrier_{carrier_id_str}.png"
                plot_distribution(
                    classes_sorted, true_dist, pred_dist,
                    title=f'Carrier {carrier_id}: True vs Predicted Delivery Time Distribution',
                    save_path=save_path
                )
                print(f"  Plot saved to {save_path}")
            except Exception as plot_error:
                print(f"  Warning: Could not generate plot: {plot_error}")
                import traceback
                traceback.print_exc()
        
        except Exception as e:
            print(f"  Error training simulator: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Training complete!")
    print(f"{'='*80}")
    print(f"Models saved to: {models_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train carrier-specific delivery time simulators")
    parser.add_argument('--n_trials', type=int, default=0,
                        help="Number of Optuna hyperparameter tuning trials (0 = no tuning)")
    parser.add_argument('--simulator_type', type=str, default='dl', choices=['catboost', 'dl'],
                        help="Type of simulator to train.")
    parser.add_argument('--carrier_id', type=int, default=None,
                        help="Restrict training/tuning to a single carrier_id (default: train all).")
    args = parser.parse_args()
    
    main(args)
