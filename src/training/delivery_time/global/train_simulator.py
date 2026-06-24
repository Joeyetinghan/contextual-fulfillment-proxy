import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import joblib
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
import src.config as cfg
from src.training.delivery_time.common import (
    plot_distribution,
    create_delivery_time_features
)
from src.training.delivery_time._train_simulator import train_catboost_simulator, train_dl_simulator


def main(args):
    """
    Trains a delivery time simulator on the test set to approximate the true environment.
    Supports both CatBoost classifier and Deep Learning quantile models.
    """
    print("--- Training Delivery Time Simulator ---")
    
    # Ensure model directory exists
    cfg.DELIVERY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and engineer features for test dataset to train the simulator
    print("Loading test dataset and engineering features...")
    df = pd.read_csv(cfg.PREPROCESSED_PATH, parse_dates=['order_time', 'order_date'])
    # Filter to test period
    df = df[df['order_date'] > cfg.PROXY_TRAIN_END_DATE].copy()
    # Sort by order_time for rolling features
    df.sort_values(by='order_time', inplace=True)
    # Engineer all features
    df_train = create_delivery_time_features(df)
    df_train.columns = [c.strip() for c in df_train.columns]
    print(f"Test set for training: {df_train.shape}")

    # Create directory for plots
    plots_dir = cfg.DELIVERY_PLOTS_DIR / 'simulator'
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.simulator_type == 'catboost':
        # --- Global CatBoost Model Training ---
        calibrated_clf, le = train_catboost_simulator(df_train, n_trials=args.n_trials)

        if calibrated_clf is None:
            print("Skipping training: Not enough classes with sufficient samples in the dataset.")
            return

        # Save the model and the label encoder (CatBoost)
        model_path = cfg.DELIVERY_MODELS_DIR / 'delivery_simulator_catboost.joblib'
        joblib.dump({'model': calibrated_clf, 'label_encoder': le}, model_path)
        print(f"Global simulator saved to {model_path}")

        # --- Plotting global performance ---
        print("Generating global performance plot...")
        y = df_train['delivery_time_days']
        valid_classes = le.classes_
        classes_sorted = np.array(sorted(valid_classes))
        true_dist = np.array([(y == c).mean() for c in classes_sorted])
        
        features = cfg.DELIVERY_TIME_FEATURES
        X = df_train[features].fillna(0)
        y_prob = calibrated_clf.predict_proba(X)
        clf_classes_enc = calibrated_clf.classes_
        clf_classes = le.inverse_transform(clf_classes_enc)
        pred_dist = np.zeros_like(classes_sorted, dtype=float)
        for i, c in enumerate(classes_sorted):
            if c in clf_classes:
                idx = np.where(clf_classes == c)[0][0]
                pred_dist[i] = y_prob[:, idx].mean()

        save_path = plots_dir / 'dist_comparison_global.png'
        plot_distribution(classes_sorted, true_dist, pred_dist,
                            title='Global: True vs Predicted Delivery Time Distribution',
                            save_path=save_path)
        print(f"Global distribution plot saved to {save_path}")
        
        # --- DC-specific performance plots ---
        print("Generating DC-specific performance plots...")
        for dc in df_train['dc_ori'].unique():
            dc_mask = (df_train['dc_ori'] == dc)
            dc_data = df_train[dc_mask]
            
            if len(dc_data) < 50: continue
                
            X_dc = dc_data[features]
            y_dc = dc_data[cfg.DELIVERY_TIME_TARGET]
            dc_true_dist = np.array([(y_dc == c).mean() for c in classes_sorted])
            y_dc_prob = calibrated_clf.predict_proba(X_dc)
            dc_pred_dist = np.zeros_like(classes_sorted, dtype=float)
            for i, c in enumerate(classes_sorted):
                if c in clf_classes:
                    idx = np.where(clf_classes == c)[0][0]
                    dc_pred_dist[i] = y_dc_prob[:, idx].mean()
            
            dc_save_path = plots_dir / f'dist_comparison_dc_{dc}.png'
            plot_distribution(classes_sorted, dc_true_dist, dc_pred_dist,
                                title=f'DC {dc}: True vs Predicted Delivery Time Distribution (n={len(dc_data)})',
                                save_path=dc_save_path)
            print(f"DC {dc} distribution plot saved to {dc_save_path}")

        print("\n--- Global CatBoost simulator has been trained. ---")

    elif args.simulator_type == 'dl':
        # --- Global DL Model Training ---
        model, x_scaler, encoders, vocab_sizes, final_params, Xn, Xc, y = train_dl_simulator(df_train, n_trials=args.n_trials)

        bundle_path = cfg.DELIVERY_MODELS_DIR / args.dl_model_name
        torch.save({
            'state_dict': model.state_dict(),
            'x_scaler': x_scaler,
            'categorical_encoders': encoders,
            'vocab_sizes': vocab_sizes,
            'numerical_dim': Xn.shape[1],
            'hidden_dim': final_params['hidden_dim'],
            'n_layers': final_params['n_layers'],
            'dropout': True,
            'dropout_p': final_params['dropout_p'],
            'dc_ori_embedding_dim': cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM,
            'dc_des_embedding_dim': cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM,
        }, bundle_path)
        joblib.dump({'type': 'dl', 'dl_model_path': str(bundle_path)}, cfg.DELIVERY_MODELS_DIR / 'delivery_simulator_dl.joblib')
        print(f"Saved DL simulator bundle to {bundle_path}")

        # --- Plotting DL performance ---
        print("Generating DL performance plots...")
        from src.utils import sample_paths
        
        plots_dir = cfg.DELIVERY_PLOTS_DIR / 'simulator'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        batch_size = 2048
        preds = np.zeros((len(Xn), len(cfg.DELIVERY_TIME_QUANTILES)), dtype=np.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for s in range(0, len(Xn), batch_size):
                e = min(len(Xn), s + batch_size)
                Xn_batch = Xn[s:e].to(device)
                dc_ori_batch = Xc['dc_ori'][s:e].to(device)
                dc_des_batch = Xc['dc_des'][s:e].to(device)
                preds[s:e] = model(Xn_batch, dc_ori_batch, dc_des_batch).cpu().numpy()

        rng = np.random.default_rng(cfg.RANDOM_SEED)
        n_samples = 1000
        all_samples = []
        for i in range(len(preds)):
            samples = sample_paths(preds[i], n_samples, rng=rng, lower=0, upper=5)
            all_samples.append(np.rint(samples).astype(int))
        all_samples = np.array(all_samples)
        
        classes_sorted = np.array(sorted(y.unique()))
        true_dist = np.array([(y == c).mean() for c in classes_sorted])
        
        pred_dist = np.zeros_like(classes_sorted, dtype=float)
        for i, c in enumerate(classes_sorted):
            pred_dist[i] = (all_samples == c).mean()
        
        save_path = plots_dir / 'dist_comparison_global_dl.png'
        plot_distribution(classes_sorted, true_dist, pred_dist,
                          title='DL Global: True vs Predicted Delivery Time Distribution',
                          save_path=save_path)
        print(f"DL global distribution plot saved to {save_path}")
        
        for dc in df['dc_ori'].unique():
            dc_mask = (df['dc_ori'] == dc)
            dc_data = df[dc_mask]
            
            if len(dc_data) < 50: continue
                
            dc_indices = dc_data.index
            y_dc = y.iloc[dc_indices]
            dc_samples = all_samples[dc_indices]
            
            dc_true_dist = np.array([(y_dc == c).mean() for c in classes_sorted])
            dc_pred_dist = np.zeros_like(classes_sorted, dtype=float)
            for i, c in enumerate(classes_sorted):
                dc_pred_dist[i] = (dc_samples == c).mean()
            
            dc_save_path = plots_dir / f'dist_comparison_dc_{dc}_dl.png'
            plot_distribution(classes_sorted, dc_true_dist, dc_pred_dist,
                              title=f'DL DC {dc}: True vs Predicted Delivery Time Distribution (n={len(dc_data)})',
                              save_path=dc_save_path)
            print(f"DL DC {dc} distribution plot saved to {dc_save_path}")

        print("\n--- DL simulator training and plotting complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a delivery time simulator: CatBoost (default) or DL on test.")
    parser.add_argument('--n_trials', type=int, default=0, help="Optuna trials (CatBoost or DL, depending on simulator_type).")
    parser.add_argument('--simulator_type', type=str, default='catboost', choices=['catboost', 'dl'])
    parser.add_argument('--dl_model_name', type=str, default='delivery_simulator_dl_test.pt')
    args = parser.parse_args()
    main(args)
