import os
import numpy as np
import xgboost as xgb
import src.config as cfg
from src.training.delivery_time.common import pinball_loss_np as _pinball_loss_np

def train_xgboost_quantile(X_train, y_train, X_val, y_val, X_full, y_full, quantiles, carrier_id=None):
    """Train XGBoost models for each quantile for a specific carrier service."""
    models = {}
    train_losses = []
    val_losses = []
    
    model_name = "XGBoost quantile models"
    if carrier_id:
        model_name += f" for carrier service {carrier_id}"
    print(f"  Training {model_name}...")
    
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    
    for i, tau in enumerate(quantiles):
        if i % 5 == 0:
            print(f"    Quantile {tau:.2f} ({i+1}/{len(quantiles)})")
        
        # Old hyperparameters (commented out - were performing too well):
        # 'max_depth': 6,
        # 'learning_rate': 0.1,
        # 'n_estimators': 100,
        # 'subsample': 0.8,
        # 'colsample_bytree': 0.8,
        #
        # Previous degraded hyperparameters (commented out - still performing too well):
        # 'max_depth': 3,
        # 'learning_rate': 0.3,
        # 'n_estimators': 30,
        # 'subsample': 0.5,
        # 'colsample_bytree': 0.5,
        #
        # Second degraded hyperparameters (commented out - still performing too well):
        # 'max_depth': 2,
        # 'learning_rate': 0.5,
        # 'n_estimators': 15,
        # 'subsample': 0.3,
        # 'colsample_bytree': 0.3,
        
        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': tau,
            'max_depth': 1,              # Further reduced from 2 to 1 (stumps only = extremely minimal capacity)
            'learning_rate': 0.8,         # Further increased from 0.5 to 0.8 (extremely unstable, very poor convergence)
            'n_estimators': 10,           # Further reduced from 15 to 10 (extremely few trees = minimal capacity)
            'subsample': 0.2,             # Further reduced from 0.3 to 0.2 (extremely little data per tree)
            'colsample_bytree': 0.2,      # Further reduced from 0.3 to 0.2 (extremely few features per tree)
            'random_state': cfg.RANDOM_SEED,
            'n_jobs': n_jobs,
            'early_stopping_rounds': cfg.DELIVERY_DL_EARLY_STOPPING_PATIENCE
        }
        
        # Phase 1: Train on split data for validation
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred = model.predict(X_val)
        val_loss = _pinball_loss_np(val_pred, y_val, tau)
        val_losses.append(val_loss)
        
        # Phase 2: Retrain on full dataset
        model.fit(X_full, y_full, eval_set=[(X_val, y_val)], verbose=False)
        models[tau] = model
        
        train_pred = model.predict(X_train)
        train_loss = _pinball_loss_np(train_pred, y_train, tau)
        train_losses.append(train_loss)
    
    print(f"    Average Val Loss: {np.mean(val_losses):.4f}, Train Loss: {np.mean(train_losses):.4f}")
    
    return models, train_losses, val_losses
