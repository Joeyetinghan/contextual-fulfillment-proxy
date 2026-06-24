import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
import src.config as cfg
from src.utils import pinball_loss
from src.model.time_quantile_model import TimeQuantileModel


# ═══════════════════════════════════════════════════════════════════════════
# Model Training Core Functions
# ═══════════════════════════════════════════════════════════════════════════

def train_dl_model(params, X_numerical_train, X_categorical_train, y_train, 
                           X_numerical_val, X_categorical_val, y_val, vocab_sizes, device, 
                           return_model=False, trial=None):
    """
    Core helper to train and evaluate the DL model.
    - If `X_val` and `y_val` are provided, it performs validation and early stopping.
    - If `trial` is provided (from Optuna), it's a tuning run. It returns the best validation loss.
    - If `return_model` is True, it returns the trained model and loss curves.
    """
    is_validation_run = (X_numerical_val is not None) and (y_val is not None)

    # Create dataset from numerical and categorical features
    if 'dc_ori' in X_categorical_train and 'dc_des' in X_categorical_train:
        train_ds = TensorDataset(X_numerical_train, X_categorical_train['dc_ori'], X_categorical_train['dc_des'], y_train)
    else:
        # Fallback if categorical features are missing
        train_ds = TensorDataset(X_numerical_train, y_train)

    # Use drop_last=False to prevent discarding data for small datasets
    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, drop_last=False, num_workers=4)
    
    if is_validation_run:
        X_numerical_val = X_numerical_val.to(device)
        if X_categorical_val:
            for cat_feature, tensor in X_categorical_val.items():
                X_categorical_val[cat_feature] = tensor.to(device)
        y_val = y_val.to(device)

    # Get embedding dimensions from params if provided, otherwise use config defaults
    dc_ori_emb_dim = params.get('dc_ori_embedding_dim', cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM)
    dc_des_emb_dim = params.get('dc_des_embedding_dim', cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM)
    
    model = TimeQuantileModel(
        numerical_dim=X_numerical_train.shape[1], 
        hidden_dim=params['hidden_dim'], 
        n_layers=params['n_layers'],
        dropout=True, 
        dropout_p=params['dropout_p'],
        dc_ori_vocab_size=vocab_sizes.get('dc_ori'),
        dc_des_vocab_size=vocab_sizes.get('dc_des'),
        dc_ori_embedding_dim=dc_ori_emb_dim,
        dc_des_embedding_dim=dc_des_emb_dim
    ).to(device)
    # model = torch.compile(model)
    opt = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    if is_validation_run:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=cfg.DELIVERY_DL_LR_SCHEDULER_PATIENCE, factor=cfg.DELIVERY_DL_LR_SCHEDULER_FACTOR)
    
    taus = torch.tensor(cfg.DELIVERY_TIME_QUANTILES, device=device)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0
    
    epochs = params.get('epochs', cfg.DELIVERY_DL_EPOCHS)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_batches_processed = 0
        
        for batch in train_dl:
            if len(batch) == 4:  # numerical, dc_ori, dc_des, target
                xb_numerical, xb_dc_ori, xb_dc_des, yb = batch
                xb_numerical = xb_numerical.to(device)
                xb_dc_ori = xb_dc_ori.to(device)
                xb_dc_des = xb_dc_des.to(device)
                yb = yb.to(device)
                
                # When drop_last=False, the last batch can have size 1.
                # BatchNorm1d layers fail on batches of size 1 during training, so we skip them.
                if xb_numerical.shape[0] <= 1:
                    continue
                
                pred = model(xb_numerical, xb_dc_ori, xb_dc_des)
            else:  # fallback for numerical only
                xb_numerical, yb = batch
                xb_numerical = xb_numerical.to(device)
                yb = yb.to(device)
                
                if xb_numerical.shape[0] <= 1:
                    continue
                
                pred = model(xb_numerical)
            
            loss = pinball_loss(pred, yb, taus)
            opt.zero_grad(); loss.backward(); opt.step()
            total_train_loss += loss.item()
            num_batches_processed += 1

        if num_batches_processed > 0:
            train_losses.append(total_train_loss / num_batches_processed)
        else:
            train_losses.append(0.0) # All batches were skipped

        if is_validation_run:
            model.eval()
            with torch.no_grad():
                if model.use_embeddings and X_categorical_val:
                    val_pred = model(
                        X_numerical_val,
                        X_categorical_val.get('dc_ori'),
                        X_categorical_val.get('dc_des')
                    )
                else:
                    val_pred = model(X_numerical_val)
                val_loss = pinball_loss(val_pred, y_val, taus).item()
                val_losses.append(val_loss)
            
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1
            
            # Get early stopping patience from params if provided, otherwise use config default
            early_stopping_patience = params.get('early_stopping_patience', cfg.DELIVERY_DL_EARLY_STOPPING_PATIENCE)
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}")
        
        
    if not train_dl and trial:
        raise optuna.exceptions.TrialPruned("Skipping trial, no training data.")

    # After loop, decide what to do
    if is_validation_run and best_model_state:
        # If there was a validation run, always load the best model state
        model.load_state_dict(best_model_state)

    if trial:
        # This is an Optuna run. Store best epoch and return loss.
        trial.set_user_attr("best_epoch", best_epoch)
        return best_val_loss
    
    if return_model:
        # This is for any final training (retrain or standard with validation)
        return model, train_losses, val_losses
    
    # Fallback for the tuning case where trial is not passed but we expect a loss
    return best_val_loss
