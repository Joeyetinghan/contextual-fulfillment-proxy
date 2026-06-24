import numpy as np
import src.config as cfg
from quantile_forest import RandomForestQuantileRegressor

def train_rf_model(X_train, y_train, X_val, y_val, X_full, y_full, carrier_id=None):
    """Train Random Forest Quantile Regressor for a specific carrier service."""
    model_name = "RF model"
    if carrier_id:
        model_name += f" for carrier {carrier_id}"
    print(f"  Training {model_name}...")
    
    # Old hyperparameters (commented out - were performing too well):
    # n_estimators=100,
    # max_depth=20,
    # min_samples_split=10,
    # min_samples_leaf=5,
    #
    # Previous degraded hyperparameters (commented out - still performing too well):
    # n_estimators=50,
    # max_depth=10,
    # min_samples_split=20,
    # min_samples_leaf=10,
    
    # Train on split for validation
    model = RandomForestQuantileRegressor(
        n_estimators=25,              # Further reduced from 50 to 25 (very few trees = minimal capacity)
        max_depth=5,                   # Further reduced from 10 to 5 (very shallow trees = minimal capacity)
        min_samples_split=30,         # Further increased from 20 to 30 (very restrictive splitting = very simple trees)
        min_samples_leaf=15,          # Further increased from 10 to 15 (very large leaves = minimal detail)
        random_state=cfg.RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Validation loss (approximation using median)
    val_pred = model.predict(X_val, quantiles=[0.5])
    val_loss = np.mean(np.abs(val_pred.ravel() - y_val))
    
    # Retrain on full dataset
    print(f"    Retraining on full dataset...")
    model.fit(X_full, y_full)
    
    train_pred = model.predict(X_train, quantiles=[0.5])
    train_loss = np.mean(np.abs(train_pred.ravel() - y_train))
    
    print(f"    Val Loss: {val_loss:.4f}, Train Loss: {train_loss:.4f}")
    
    return model, train_loss, val_loss
