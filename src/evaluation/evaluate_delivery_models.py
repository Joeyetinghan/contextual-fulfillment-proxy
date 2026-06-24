import sys
import os

# Add the current directory to Python path so src module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import src.config as cfg
from src.model.time_quantile_model import TimeQuantileModel
from src.utils import pinball_loss
import joblib, glob

# ------------------------------------------------------------
# CRPS calculation functions
# ------------------------------------------------------------

def crps(y_true, y_pred, sample_weight=None):
    '''
    n log n algorithm for CRPS given samples y_pred (shape [S, N]) and true y_true (shape [N]).
    '''
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)

    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return np.average(per_obs_crps, weights=sample_weight)

def sample_from_quantiles(quantile_predictions, quantiles, num_samples=1000):
    """
    Generate samples from quantile predictions using inverse CDF sampling.
    
    Parameters:
    -----------
    quantile_predictions : np.ndarray, shape (N, num_quantiles)
        Quantile predictions for N observations
    quantiles : list
        List of quantile values corresponding to the predictions
    num_samples : int
        Number of samples to generate per observation
        
    Returns:
    --------
    samples : np.ndarray, shape (num_samples, N)
        Generated samples
    """
    N, num_quantiles = quantile_predictions.shape
    samples = np.zeros((num_samples, N))
    
    # Generate uniform random numbers
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    u = rng.random((num_samples, N))
    
    # For each observation, interpolate between quantiles
    for i in range(N):
        # Sort quantiles and predictions
        sorted_indices = np.argsort(quantile_predictions[i])
        sorted_quantiles = np.array(quantiles)[sorted_indices]
        sorted_predictions = quantile_predictions[i][sorted_indices]
        
        # Interpolate to get samples
        samples[:, i] = np.interp(u[:, i], sorted_quantiles, sorted_predictions)
    
    return samples

def calculate_crps(predictions, y_true, quantiles, model_name, num_samples=1000):
    """Calculate CRPS for quantile predictions by sampling from the distribution."""
    
    print(f"Calculating CRPS for {model_name}...")
    
    # Generate samples from quantile predictions
    samples = sample_from_quantiles(predictions, quantiles, num_samples)
    
    # Calculate CRPS
    crps_score = crps(y_true, samples)
    
    print(f"CRPS for {model_name}: {crps_score:.6f}")
    return crps_score

# ------------------------------------------------------------
# Evaluate delivery time QR (classical) models helper
# ------------------------------------------------------------

def evaluate_delivery_qr_family(model_prefix: str):
    """Load 19 single-quantile QR models of one family (xgboost | sklearn) and evaluate on test set."""
    dir_path = cfg.DELIVERY_QR_MODELS_DIR / 'update_0728'
    pattern = str(dir_path / f"{model_prefix}*with_proxy.joblib")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for pattern {pattern}. Skipping …")
        return
    # Load first bundle to get scaler / encoders
    first_bundle = joblib.load(files[0])
    x_scaler = first_bundle.get('x_scaler')
    cat_encoders = first_bundle.get('categorical_encoders')
    taus = cfg.DELIVERY_TIME_QUANTILES
    tau_to_model = {}
    for fp in files:
        bundle = joblib.load(fp)
        mdl   = list(bundle['models'].values())[0]
        q_single = bundle['quantiles'][0]
        tau_to_model[q_single] = mdl
    missing = [t for t in taus if t not in tau_to_model]
    if missing:
        print(f"Warning: missing quantile models {missing} for family {model_prefix}")
    # Prepare test dataframe
    df_test = pd.read_csv(cfg.DELIVERY_TEST_PATH, parse_dates=['order_time'])
    df_test.columns = [c.strip() for c in df_test.columns]
    X_num  = df_test[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    X_num_scaled = x_scaler.transform(X_num)
    X_cat = df_test[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
    X_cat_encoded = {feat: cat_encoders[feat].transform(X_cat[feat].astype(str)).reshape(-1,1)
                      for feat in cfg.DELIVERY_DL_CATEGORICAL_FEATURES}
    X_mat = np.concatenate([X_num_scaled] + [X_cat_encoded[f] for f in cfg.DELIVERY_DL_CATEGORICAL_FEATURES], axis=1)
    y_true = df_test[cfg.DELIVERY_TIME_TARGET].values
    mask = ~np.isnan(y_true)
    X_mat, y_true = X_mat[mask], y_true[mask]
    preds = np.zeros((len(y_true), len(taus)))
    for j,tau in enumerate(taus):
        if tau in tau_to_model:
            preds[:,j] = tau_to_model[tau].predict(X_mat)
        else:
            preds[:,j] = np.nan
    
    # Round predictions and true values to integers before calculating metrics
    preds_rounded = np.round(preds).astype(int)
    y_true_rounded = np.round(y_true).astype(int)
    
    # Calculate both metrics
    print(f"\n--- {model_prefix.upper()} Model Family Evaluation ---")
    crps_score = calculate_crps(preds_rounded, y_true_rounded, taus, model_prefix)
    
    # Existing pinball loss calculation
    available_cols = [j for j,tau in enumerate(taus) if tau in tau_to_model]
    preds_tensor = torch.tensor(preds_rounded[:,available_cols], dtype=torch.float32)
    y_tensor = torch.tensor(y_true_rounded, dtype=torch.float32)
    taus_tensor = torch.tensor([taus[j] for j in available_cols], dtype=torch.float32)
    total_loss = pinball_loss(preds_tensor, y_tensor, taus_tensor).item()
    per_q = {}
    for j in available_cols:
        q_pred = torch.tensor(preds_rounded[:,j].reshape(-1,1), dtype=torch.float32)
        per_q[taus[j]] = pinball_loss(q_pred, y_tensor, torch.tensor([taus[j]])).item()
    
    print(f"\n{model_prefix.upper()} Summary:")
    print(f"  CRPS: {crps_score:.6f}")
    print(f"  Total Pinball Loss (across available quantiles): {total_loss:.6f}")
    
    print(f"\n=== PINBALL LOSS RESULTS FOR {model_prefix} (single-quantile models) ===")
    print(f"Total Pinball Loss (across available quantiles): {total_loss:.6f}")
    for tau in taus:
        if tau in per_q:
            print(f"  Quantile {tau:.2f}: {per_q[tau]:.6f}")
        else:
            print(f"  Quantile {tau:.2f}: NA")
    print(f"Number of test samples: {len(y_true)}  (after NaN filtering)")

# ────────────────────────── Delivery QRF evaluation helper ───────────────────

def evaluate_delivery_qrf_model(model_path):
    """Evaluate the *single* Random-Forest Quantile Regressor that outputs all
    19 quantiles at once (the model trained when `src/training/delivery_time/global/train_dl_rf.py`
    runs with `--use_dl` **disabled**).

    Parameters
    ----------
    model_path : pathlib.Path  – path to the `.joblib` file that stores the
        fitted `RandomForestQuantileRegressor`.
    """

    if not model_path.exists():
        print(f"QRF model not found at {model_path}. Skipping …")
        return

    print(f"\nEvaluating delivery QRF model at {model_path} …")

    # ---------------- Load model --------------------------------------------
    qrf_model = joblib.load(model_path)

    # ---------------- Prepare test dataframe --------------------------------
    df_test = pd.read_csv(cfg.DELIVERY_TEST_PATH, parse_dates=['order_time'])
    df_test.columns = [c.strip() for c in df_test.columns]

    features = cfg.DELIVERY_TIME_FEATURES
    X_test = df_test[features]
    y_true = df_test[cfg.DELIVERY_TIME_TARGET].values

    # Drop rows with NaN target as was done elsewhere
    mask = ~np.isnan(y_true)
    X_test = X_test.loc[mask]
    y_true = y_true[mask]

    # ---------------- Predictions ------------------------------------------
    taus = cfg.DELIVERY_TIME_QUANTILES
    preds = qrf_model.predict(X_test, quantiles=taus)  # shape [N,19]

    # Round predictions and true values to integers before calculating metrics
    preds_rounded = np.round(preds).astype(int)
    y_true_rounded = np.round(y_true).astype(int)
    
    # ---------------- Calculate both metrics --------------------------------
    print("\n--- QRF Model Evaluation ---")
    crps_score = calculate_crps(preds_rounded, y_true_rounded, taus, "QRF")
    total_loss, per_quantile_losses = calculate_pinball_loss(preds_rounded, y_true_rounded, taus, model_path.name)
    
    print(f"\nQRF Summary:")
    print(f"  CRPS: {crps_score:.6f}")
    print(f"  Total Pinball Loss: {total_loss:.6f}")

def evaluate_delivery_time_model(model_path):
    """Evaluate a delivery time quantile model and calculate pinball loss."""
    
    print(f"Loading delivery model from {model_path}...")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None, None
    
    # Load model and preprocessors
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model components
    state_dict = model_data['state_dict']
    x_scaler = model_data['x_scaler']
    categorical_encoders = model_data['categorical_encoders']
    vocab_sizes = model_data['vocab_sizes']
    numerical_dim = model_data['numerical_dim']
    hidden_dim = model_data['hidden_dim']
    n_layers = model_data['n_layers']
    dropout_p = model_data['dropout_p']
    dc_ori_embedding_dim = model_data['dc_ori_embedding_dim']
    dc_des_embedding_dim = model_data['dc_des_embedding_dim']
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeQuantileModel(
        numerical_dim=numerical_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=True,
        dropout_p=dropout_p,
        dc_ori_vocab_size=vocab_sizes.get('dc_ori'),
        dc_des_vocab_size=vocab_sizes.get('dc_des'),
        dc_ori_embedding_dim=dc_ori_embedding_dim,
        dc_des_embedding_dim=dc_des_embedding_dim
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Loading test data...")
    # Load test data (since this model was trained on forecast+proxy, evaluate on test)
    df_test = pd.read_csv(cfg.DELIVERY_TEST_PATH, parse_dates=['order_time'])
    df_test.columns = [c.strip() for c in df_test.columns]
    
    # Prepare features
    features = cfg.DELIVERY_TIME_FEATURES
    X_test = df_test[features]
    y_test = df_test[cfg.DELIVERY_TIME_TARGET]
    
    # Remove rows with missing target values
    valid_indices = y_test.dropna().index
    X_test = X_test.loc[valid_indices]
    y_test = y_test.loc[valid_indices]
    
    print(f"Evaluating on {len(X_test)} test samples...")
    
    # Prepare numerical and categorical features
    X_numerical = X_test[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    X_categorical = X_test[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
    
    # Scale numerical features
    X_numerical_scaled = x_scaler.transform(X_numerical)
    X_numerical_tensor = torch.tensor(X_numerical_scaled, dtype=torch.float32, device=device)
    
    # Encode categorical features
    X_categorical_encoded = {}
    for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
        if cat_feature in X_categorical.columns:
            encoded = categorical_encoders[cat_feature].transform(X_categorical[cat_feature].astype(str))
            X_categorical_encoded[cat_feature] = torch.tensor(encoded, dtype=torch.long, device=device)
    
    # Convert target to tensor
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=device)
    
    # Make predictions
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        predictions = model(
            X_numerical_tensor,
            X_categorical_encoded.get('dc_ori'),
            X_categorical_encoded.get('dc_des')
        ).cpu().numpy()
    
    # Round predictions and true values to integers before calculating metrics
    predictions_rounded = np.round(predictions).astype(int)
    y_test_rounded = np.round(y_test_tensor.cpu().numpy()).astype(int)
    
    # Calculate both metrics
    print("\n--- MQRNN Model Evaluation ---")
    crps_score = calculate_crps(predictions_rounded, y_test_rounded, cfg.DELIVERY_TIME_QUANTILES, "MQRNN")
    total_loss, per_quantile_losses = calculate_pinball_loss(predictions_rounded, y_test_rounded, cfg.DELIVERY_TIME_QUANTILES, "delivery_model_global_with_proxy.pt")
    
    print(f"\nMQRNN Summary:")
    print(f"  CRPS: {crps_score:.6f}")
    print(f"  Total Pinball Loss: {total_loss:.6f}")
    
    return predictions_rounded, y_test_rounded

def calculate_pinball_loss(predictions, y_true, quantiles, model_name):
    """Calculate pinball loss for predictions across all quantiles using utils.pinball_loss."""
    
    print("Calculating pinball loss...")
    
    # Convert to tensors for the pinball_loss function
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    taus_tensor = torch.tensor(quantiles, dtype=torch.float32)
    
    # Calculate total pinball loss using the utils function
    total_pinball_loss = pinball_loss(predictions_tensor, y_true_tensor, taus_tensor).item()
    
    # Calculate per-quantile losses using the same utils function
    per_quantile_losses = []
    for j, tau in enumerate(quantiles):
        # For each quantile, calculate loss using only that quantile's predictions
        single_quantile_predictions = predictions_tensor[:, j:j+1]  # Shape: [n_samples, 1]
        single_quantile_taus = taus_tensor[j:j+1]  # Shape: [1]
        
        quantile_loss = pinball_loss(single_quantile_predictions, y_true_tensor, single_quantile_taus).item()
        per_quantile_losses.append(quantile_loss)
    
    print(f"\n=== PINBALL LOSS RESULTS FOR {model_name} ===")
    print(f"Total Pinball Loss (across all quantiles): {total_pinball_loss:.6f}")
    print("\nPer-quantile losses:")
    for i, (tau, loss) in enumerate(zip(quantiles, per_quantile_losses)):
        print(f"  Quantile {tau:.2f}: {loss:.6f}")
    
    # Additional statistics
    print(f"\nNumber of test samples: {len(y_true)}")
    print(f"Number of quantiles: {len(quantiles)}")
    print(f"Model: {model_name}")
    
    return total_pinball_loss, per_quantile_losses

def main():
    """Main function to evaluate delivery models."""
    
    print("=== DELIVERY MODEL EVALUATION ===\n")
    
    # Evaluate delivery time model
    delivery_model_path = cfg.DELIVERY_MODELS_DIR / "delivery_model_global_with_proxy.pt"
    if delivery_model_path.exists():
        predictions, y_true = evaluate_delivery_time_model(delivery_model_path)
        if predictions is not None:
            calculate_pinball_loss(
                predictions, y_true, 
                cfg.DELIVERY_TIME_QUANTILES, 
                "delivery_model_global_with_proxy.pt"
            )
    else:
        print(f"Delivery model not found at {delivery_model_path}")
    
    print("\n" + "="*50 + "\n")

    # Evaluate delivery QRF model
    delivery_qrf_path = cfg.DELIVERY_MODELS_DIR / "delivery_model_global_with_proxy.joblib"
    if delivery_qrf_path.exists():
        evaluate_delivery_qrf_model(delivery_qrf_path)
    else:
        print(f"Delivery QRF model not found at {delivery_qrf_path}")
    
    # Evaluate classical delivery-time QR models
    evaluate_delivery_qr_family('delivery_model_global_xgboost')
    evaluate_delivery_qr_family('delivery_model_global_sklearn')

if __name__ == "__main__":
    main() 