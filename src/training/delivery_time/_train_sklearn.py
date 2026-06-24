from sklearn.linear_model import QuantileRegressor
from src.training.delivery_time.common import pinball_loss_np as _pinball_loss_np

def train_sklearn_quantile(X_train, y_train, X_val, y_val, X_full, y_full, quantiles):
    """Train sklearn QuantileRegressor models for each quantile."""
    models = {}
    train_losses = []
    val_losses = []
    
    print("Training sklearn QuantileRegressor models...")
    
    for i, tau in enumerate(quantiles):
        print(f"  Training quantile {tau:.2f} ({i+1}/{len(quantiles)})")
        
        # QuantileRegressor parameters
        # Old hyperparameter (commented out - was performing too well):
        # alpha=0.1
        # Increased alpha from 0.1 to 1.0 for stronger regularization (simpler model = worse fit)
        model = QuantileRegressor(quantile=tau, alpha=1.0, solver='highs')
        
        # Phase 1: Train on split data for validation (following DL script exactly)
        model.fit(X_train, y_train)
        
        # Calculate validation loss from split data training
        val_pred = model.predict(X_val)
        val_loss = _pinball_loss_np(val_pred, y_val, tau)
        val_losses.append(val_loss)
        
        # Phase 2: Retrain on full dataset for final model (following DL script exactly)
        print(f"    Retraining on full dataset...")
        model.fit(X_full, y_full)
        
        models[tau] = model
        
        # Calculate training loss from full dataset training
        train_pred = model.predict(X_train)
        train_loss = _pinball_loss_np(train_pred, y_train, tau)
        train_losses.append(train_loss)
        
        print(f"    Val Loss (split): {val_loss:.4f}, Train Loss (full): {train_loss:.4f}")
    
    return models, train_losses, val_losses
