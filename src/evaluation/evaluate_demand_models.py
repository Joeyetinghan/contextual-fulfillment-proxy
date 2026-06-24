import sys
import os

# Add the current directory to Python path so src module can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import src.config as cfg
from src.model.mqrnn import MQRNN
from src.utils import pinball_loss
import joblib, glob
from sklearn.preprocessing import StandardScaler

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

def calculate_crps_multi_horizon_avg(predictions_full, y_true_full, quantiles, model_name, num_samples=None):
    """Average CRPS across horizons by sampling per horizon to keep memory bounded."""
    L = min(predictions_full.shape[1], y_true_full.shape[1])
    num_samples = int(os.getenv('EVAL_CRPS_SAMPLES', str(num_samples or 1000)))
    crps_vals = []
    for h in range(L):
        preds_h = predictions_full[:, h, :]
        y_h = y_true_full[:, h]
        samples_h = sample_from_quantiles(preds_h, quantiles, num_samples)
        crps_vals.append(crps(y_h, samples_h))
    mean_crps = float(np.mean(crps_vals))
    print(f"CRPS (avg over {L} horizons) for {model_name}: {mean_crps:.6f}")
    return mean_crps

def evaluate_demand_model(model_path):
    """Evaluate a demand quantile model and calculate pinball loss."""
    
    print(f"Loading demand model from {model_path}...")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return None, None
    
    # Load model and preprocessors
    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model components
    state_dict = model_data['state_dict']
    hyperparameters = model_data['hyperparameters']
    meta = model_data['meta']
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MQRNN(
        num_cal=meta["num_cal"], 
        num_ord=meta["num_ord"],
        num_skus=meta["num_skus"], 
        num_brands=meta["num_brands"],
        sku_emb=hyperparameters.get("sku_emb", cfg.DEMAND_MODEL_SKU_EMBEDDING_DIM),
        brand_emb=hyperparameters.get("brand_emb", cfg.DEMAND_MODEL_BRAND_EMBEDDING_DIM),
        hidden=hyperparameters["hidden_dim"], 
        ctx=hyperparameters.get("context_dim", cfg.DEMAND_MODEL_CONTEXT_DIM),
        num_q=len(cfg.DEMAND_MODEL_QUANTILES),
        Lp=meta["Lp"], 
        Lh=meta["Lh"],
        layers=hyperparameters["lstm_n_layers"],
        dropout=hyperparameters.get("dropout_p", 0) > 0,
        dropout_p=hyperparameters.get("dropout_p", 0.0),
        bidirectional=hyperparameters.get("bidirectional", cfg.DEMAND_MODEL_BIDIRECTIONAL)
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Loading test data...")
    # Load test data from the correct path
    test_data = np.load(cfg.DEMAND_TEST_PATH)
    
    # Extract the required components
    xh = test_data['xh']  # Historical features
    yh = test_data['yh']  # Historical targets
    xp = test_data['xp']  # Prediction features
    yp = test_data['yp']  # Prediction targets (what we want to predict)
    sku_idx = test_data['sku_idx']  # SKU indices
    brand_idx = test_data['brand_idx']  # Brand indices
    
    # ------------------------------------------------------------
    #  Load the scalers that were fitted during training and
    #  standard-scale the historical (xh) and prediction (xp)
    #  feature blocks so that evaluation uses exactly the same
    #  distribution the MQRNN saw during training.
    # ------------------------------------------------------------
    # Determine train mode from model path and try tuned scalers first
    train_mode = "_with_proxy" if "with_proxy" in str(model_path) else ""
    is_tuned = "tuned" in str(model_path)
    
    if is_tuned:
        # Check in tune/ subdirectory first, then fallback to main directory
        scalers_path = cfg.DEMAND_MODELS_DIR / "tune" / f'mqrnn_scalers{train_mode}_tuned.pt'
        if not scalers_path.exists():
            scalers_path = cfg.DEMAND_MODELS_DIR / f'mqrnn_scalers{train_mode}_tuned.pt'
        if not scalers_path.exists():
            scalers_path = cfg.DEMAND_MODELS_DIR / f'mqrnn_scalers{train_mode}.pt'
    else:
        scalers_path = cfg.DEMAND_MODELS_DIR / f'mqrnn_scalers{train_mode}.pt'
    
    if scalers_path.exists():
        scalers = torch.load(scalers_path, map_location='cpu', weights_only=False)
        sc_h = scalers.get('scaler_hist')
        sc_p = scalers.get('scaler_pred')
        if sc_h is not None and sc_p is not None:
            xh_scaled = sc_h.transform(xh.reshape(-1, xh.shape[-1])
                                        ).reshape(xh.shape)
            xp_scaled = sc_p.transform(xp.reshape(-1, xp.shape[-1])
                                        ).reshape(xp.shape)
            xh = xh_scaled; xp = xp_scaled
            print("Applied feature scalers to xh and xp blocks.")
        else:
            print("Warning: Scalers dictionary is missing keys; proceeding without scaling.")
    else:
        print(f"Warning: Scaler file not found at {scalers_path}. Proceeding without scaling.")

    print(f"Evaluating on {len(xh)} test samples...")
    
    # Convert to tensors (keep on CPU; move per-batch to device during inference)
    xh_tensor = torch.tensor(xh, dtype=torch.float32)
    yh_tensor = torch.tensor(yh, dtype=torch.float32)
    xp_tensor = torch.tensor(xp, dtype=torch.float32)
    sku_tensor = torch.tensor(sku_idx, dtype=torch.long)
    brand_tensor = torch.tensor(brand_idx, dtype=torch.long)
    
    # Mini-batch predictions with OOM-safe fallback
    print("Making predictions...")
    B = xh.shape[0]
    L = xp.shape[1]
    Q = len(cfg.DEMAND_MODEL_QUANTILES)
    preds_full = np.zeros((B, L, Q), dtype=np.float32)
    bs = 1024
    print(f"Using batch size: {bs} on device: {device}")

    model.eval()
    try:
        with torch.no_grad():
            for s in range(0, B, bs):
                e = min(B, s + bs)
                xb = xh_tensor[s:e].to(device, non_blocking=True)
                yb = yh_tensor[s:e].to(device, non_blocking=True)
                pb = xp_tensor[s:e].to(device, non_blocking=True)
                sk = sku_tensor[s:e].to(device, non_blocking=True)
                br = brand_tensor[s:e].to(device, non_blocking=True)
                qb = model(xb, yb, pb, sk, br)
                if qb.ndim == 2:
                    qb = qb[:, None, :]
                preds_full[s:e] = qb[:, :L, :].float().cpu().numpy()
                del xb, yb, pb, sk, br, qb
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    except RuntimeError as err:
        if 'out of memory' in str(err).lower() and device.type == 'cuda':
            print("CUDA OOM encountered. Falling back to CPU inference …")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            device = torch.device('cpu')
            model = model.to(device)
            # Try larger CPU batches for efficiency
            bs_cpu = max(bs * 4, 1024)
            preds_full.fill(0.0)
            with torch.no_grad():
                for s in range(0, B, bs_cpu):
                    e = min(B, s + bs_cpu)
                    qb = model(
                        xh_tensor[s:e],
                        yh_tensor[s:e],
                        xp_tensor[s:e],
                        sku_tensor[s:e],
                        brand_tensor[s:e]
                    )
                    if qb.ndim == 2:
                        qb = qb[:, None, :]
                    preds_full[s:e] = qb[:, :L, :].float().cpu().numpy()
        else:
            raise
    
    # Ground-truth for all horizons
    # Inspect prediction shape/sample (concise)
    if preds_full.shape[0] > 0:
        print(f"MQRNN preds shape: {preds_full.shape}; sample[0,:3,:3]={preds_full[0, :3, :3]}")
    else:
        print(f"MQRNN preds shape: {preds_full.shape}")
    y_true_full = yp[:, :L]
    
    # Round predictions and true values to integers before calculating metrics
    predictions_rounded_full = np.round(preds_full).astype(int)
    y_true_rounded_full = np.round(y_true_full).astype(int)
    
    return predictions_rounded_full, y_true_rounded_full

def calculate_pinball_loss(predictions, y_true, quantiles, model_name, verbose=True):
    """Calculate pinball loss for quantile demand predictions.
    
    Computes the pinball loss to evaluate how well predicted quantiles
    match the true demand values.
    
    Parameters
    ----------
    predictions : np.ndarray, shape (N, Q)
        Quantile predictions for demand. Each row represents one sample (e.g., a specific
        SKU at a specific time point), and each column represents one quantile level.
        predictions[i, j] is the predicted demand at quantile level quantiles[j] for sample i.
        For example, if quantiles[j] = 0.05, then predictions[i, j] is the predicted 5th
        percentile of demand (i.e., 5% of the time, demand will be below this value).
        Note: The horizon(s) represented depend on how this function is called:
        - For "last horizon" evaluation: predictions and y_true contain values for horizon 24 only
        - For "avg over horizons" evaluation: predictions and y_true contain values from
          all horizons (1-24) flattened together (N*24 samples total)
    y_true : np.ndarray, shape (N,)
        True observed demand values for each sample. y_true[i] is the actual demand
        that occurred for sample i at the horizon(s) corresponding to predictions[i].
    quantiles : list or np.ndarray, length Q
        Quantile levels being predicted (e.g., [0.05, 0.10, 0.15, ..., 0.95]).
        Each value represents a probability level: quantiles[j] = 0.05 means the 5th
        percentile (5% of the distribution is below this value).
    model_name : str
        Name of the model being evaluated (used for printing/logging purposes).
    verbose : bool, default=True
        If True, prints detailed results including total loss, per-quantile losses, and
        statistics. If False, silently computes and returns the losses.
    
    Returns
    -------
    total_pinball_loss : float
        Total pinball loss averaged across all quantiles and all samples.
    per_quantile_losses : list
        List of pinball losses for each individual quantile, in the same order as quantiles.
    """
    
    if verbose:
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
    
    if verbose:
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

def calculate_pinball_loss_multi_horizon_avg(predictions_full, y_true_full, quantiles, model_name):
    """Flatten horizons and compute averaged pinball loss across all horizons."""
    N, Lp, Q = predictions_full.shape
    Lt = y_true_full.shape[1]
    L = min(Lp, Lt)
    preds_flat = predictions_full[:, :L, :].reshape(N * L, Q)
    y_flat = y_true_full[:, :L].reshape(N * L)
    return calculate_pinball_loss(preds_flat, y_flat, quantiles, f"{model_name} (avg over {L} horizons)")

# ────────────────────────── Demand QRF evaluation helper ─────────────────────

def evaluate_demand_qrf_model(model_path):
    """Evaluate the *single* Random-Forest Quantile Regressor that outputs all
    19 quantiles for all 24 horizon steps (the model trained when `src/training/demand/train_qrf.py` runs).

    Parameters
    ----------
    model_path : pathlib.Path  – path to the `.joblib` file that stores the
        fitted `RandomForestQuantileRegressor`.
    """

    if not model_path.exists():
        print(f"Demand QRF model not found at {model_path}. Skipping …")
        return

    print(f"\nEvaluating demand QRF model at {model_path} …")

    # ---------------- Load model --------------------------------------------
    qrf_bundle = joblib.load(model_path)
    
    # Extract the model from the bundle (we know it's saved with key 'model')
    qrf_model = qrf_bundle['model']
    
    print(f"Successfully loaded QRF model: {type(qrf_model)}")

    # ---------------- Load test data ---------------------------------------
    test_npz = np.load(cfg.DEMAND_TEST_PATH)
    xh, yh, xp, yp = (test_npz[k] for k in ('xh','yh','xp','yp'))
    sku_idx, brand_idx = test_npz['sku_idx'], test_npz['brand_idx']
    y_true_full = yp  # [N, L]

    # ---------------- Flatten features using mean/std ----------------------
    X_test = _flatten_mean_std(xh, yh, xp, sku_idx, brand_idx)

    # ---------------- Predictions ------------------------------------------
    taus = cfg.DEMAND_MODEL_QUANTILES
    preds_full = qrf_model.predict(X_test, quantiles=taus)  # shape [N, Lp, Q]
    # Inspect prediction shape/sample (concise)
    if preds_full.shape[0] > 0:
        print(f"QRF preds shape: {preds_full.shape}; sample[0,:3,:3]={preds_full[0, :3, :3]}")
    else:
        print(f"QRF preds shape: {preds_full.shape}")
    
    # Align horizons to min length
    Lp = preds_full.shape[1]
    Lt = y_true_full.shape[1]
    L = min(Lp, Lt)

    # Round predictions and true values to integers before calculating metrics
    preds_full_rounded = np.round(preds_full[:, :L, :]).astype(int)
    y_true_full_rounded = np.round(y_true_full[:, :L]).astype(int)

    # ---------------- Calculate both CRPS and pinball loss --------------------------------
    print("\n--- Demand QRF Model Evaluation ---")
    # Last-horizon metrics
    last_preds = preds_full_rounded[:, -1, :]
    last_y = y_true_full_rounded[:, -1]
    crps_last = calculate_crps(last_preds, last_y, taus, "QRF (last horizon)")
    total_loss_last, _ = calculate_pinball_loss(last_preds, last_y, taus, model_path.name + " (last horizon)", verbose=False)
    # Averaged across horizons
    crps_avg = calculate_crps_multi_horizon_avg(preds_full_rounded, y_true_full_rounded, taus, "QRF")
    total_loss_avg, _ = calculate_pinball_loss_multi_horizon_avg(preds_full_rounded, y_true_full_rounded, taus, model_path.name)
    
    print(f"\nDemand QRF Summary:")
    print(f"  CRPS (last): {crps_last:.6f}")
    print(f"  CRPS (avg over {L} horizons): {crps_avg:.6f}")
    print(f"  Total Pinball Loss (last): {total_loss_last:.6f}")
    print(f"  Total Pinball Loss (avg over {L} horizons): {total_loss_avg:.6f}")

# ────────────────────────── Demand QR evaluation helper ─────────────────────

def _flatten_mean_std(xh, yh, xp, sku_idx, brand_idx):
    """Mean/std flattening (same as train_demand_xgb.py)."""
    n = xh.shape[0]
    xh_mean = np.mean(xh, axis=1); xh_std = np.std(xh, axis=1)
    yh_mean = np.mean(yh, axis=1, keepdims=True); yh_std = np.std(yh, axis=1, keepdims=True)
    xp_mean = np.mean(xp, axis=1); xp_std = np.std(xp, axis=1)
    return np.concatenate([xh_mean, xh_std, yh_mean, yh_std, xp_mean, xp_std,
                           sku_idx.reshape(-1,1), brand_idx.reshape(-1,1)], axis=1)


def _flatten_full_temporal(xh, yh, xp, sku_idx, brand_idx):
    """Full temporal flattening (same as train_demand_xgb.py)."""
    n = xh.shape[0]
    xh_flat = xh.reshape(n, -1)
    yh_flat = yh.reshape(n, -1)
    xp_flat = xp.reshape(n, -1)
    return np.concatenate([xh_flat, yh_flat, xp_flat,
                           sku_idx.reshape(-1,1), brand_idx.reshape(-1,1)], axis=1)


def evaluate_demand_qr_family(model_prefix: str):
    """Evaluate a family of *single-quantile* demand QR models (XGB / sklearn).

    For safety we reload the *scalers* and *flattening choice* from **each**
    bundle when making its prediction, rather than assuming all bundles share
    identical preprocessing objects.  This guarantees every model is evaluated
    with the exact feature pipeline it was trained with.
    """
    dir_path = cfg.DEMAND_QR_MODELS_DIR / 'demand_qr_meanstd'  # Path to meanstd models
    pattern  = str(dir_path / f"{model_prefix}*with_proxy.joblib")
    files    = sorted(glob.glob(pattern))

    if not files:
        print(f"No demand-QR files for pattern {pattern}. Skipping …")
        return

    # ---------------- Load test arrays once ----------------------------------
    test_npz   = np.load(cfg.DEMAND_TEST_PATH)
    xh, yh, xp, yp = (test_npz[k] for k in ('xh','yh','xp','yp'))
    sku_idx, brand_idx = test_npz['sku_idx'], test_npz['brand_idx']
    y_true_full = yp  # [N, L]

    # ---------------- Load training data to fit scalers ---------------------
    print(f"Loading training data to fit scalers for {model_prefix}...")
    forecast_data = np.load(cfg.DEMAND_FORECAST_TRAIN_PATH)
    proxy_data = np.load(cfg.DEMAND_PROXY_TRAIN_PATH)
    
    # Combine forecast and proxy data for training
    train_raw = {k: np.concatenate([forecast_data[k], proxy_data[k]]) for k in forecast_data}
    
    # Fit scalers on training data (same as during training)
    print("Fitting scalers on training data...")
    hist_flat = train_raw["xh"].reshape(-1, train_raw["xh"].shape[-1])
    pred_flat = train_raw["xp"].reshape(-1, train_raw["xp"].shape[-1])
    sc_hist = StandardScaler().fit(hist_flat)
    sc_pred = StandardScaler().fit(pred_flat)
    
    # Apply scalers to test data (same as during training)
    print("Applying scalers to test data...")
    xh_scaled = sc_hist.transform(xh.reshape(-1, xh.shape[-1])).reshape(xh.shape)
    xp_scaled = sc_pred.transform(xp.reshape(-1, xp.shape[-1])).reshape(xp.shape)

    # ---------------- Loop over bundles --------------------------------------
    tau_to_pred = {}
    for fp in files:
        b   = joblib.load(fp)
        tau = b['quantiles'][0]                  # each file stores one τ
        model = list(b['models'].values())[0]

        # Use the same flattening function that was used during training
        # Since we're evaluating meanstd models, use mean/std flattening
        X_test_tau = _flatten_mean_std(xh_scaled, yh, xp_scaled, sku_idx, brand_idx)

        tau_to_pred[tau] = model.predict(X_test_tau)   # [N, L] multi-output

    # ---------------- Assemble prediction tensor [N, L, Q] -------------------
    taus = sorted(tau_to_pred.keys())
    first_tau = taus[0]
    first = tau_to_pred[first_tau]
    # Inspect first-quantile prediction shape/sample (concise)
    if first.ndim == 1:
        print(f"{model_prefix} pred[τ={first_tau:.2f}] shape {first.shape}; head {first[:5]}")
        print(f"ERROR: {model_prefix} predicted only 1 horizon; expected 24. Re-train with multi-output targets (yp) and MultiOutputRegressor.")
        return
    else:
        print(f"{model_prefix} pred[τ={first_tau:.2f}] shape {first.shape}; first-row head {first[0, :5]}")
    N, L = first.shape
    preds_full = np.zeros((N, L, len(taus)), dtype=np.float32)
    for j, tau in enumerate(taus):
        preds_full[:, :, j] = tau_to_pred[tau]

    # Align horizons with ground truth
    L_eff = min(L, y_true_full.shape[1])
    preds_full = preds_full[:, :L_eff, :]
    y_true_full_eff = y_true_full[:, :L_eff]

    # Round predictions and true values to integers before calculating metrics
    preds_full_rounded = np.round(preds_full).astype(int)
    y_true_full_rounded = np.round(y_true_full_eff).astype(int)

    # ---------------- Calculate metrics (last and averaged) ------------------
    # Last-horizon
    preds_last = preds_full_rounded[:, -1, :]
    y_last = y_true_full_rounded[:, -1]
    crps_last = calculate_crps(preds_last, y_last, taus, model_prefix + " (last horizon)")
    total_loss_last, per_quantile_losses_last = calculate_pinball_loss(preds_last, y_last, taus, model_prefix + " (last horizon)", verbose=False)

    # Averaged across horizons
    crps_avg = calculate_crps_multi_horizon_avg(preds_full_rounded, y_true_full_rounded, taus, model_prefix)
    total_loss_avg, per_quantile_losses_avg = calculate_pinball_loss_multi_horizon_avg(preds_full_rounded, y_true_full_rounded, taus, model_prefix)

    # ---------------- Reporting (match MQRNN structure) ----------------------
    print(f"\n{model_prefix} Summary:")
    print(f"  CRPS (last): {crps_last:.6f}")
    print(f"  CRPS (avg over {L_eff} horizons): {crps_avg:.6f}")
    print(f"  Total Pinball Loss (last): {total_loss_last:.6f}")
    print(f"  Total Pinball Loss (avg over {L_eff} horizons): {total_loss_avg:.6f}")

def evaluate_mqrnn_model_helper(model_path, model_name):
    """Helper function to evaluate an MQRNN model with detailed metrics."""
    if not model_path.exists():
        print(f"MQRNN model not found at {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")
    
    predictions_full, y_true_full = evaluate_demand_model(model_path)
    if predictions_full is not None:
        taus = cfg.DEMAND_MODEL_QUANTILES
        L_eff = min(predictions_full.shape[1], y_true_full.shape[1])
        
        # Last-horizon metrics (no detailed per-quantile output)
        preds_last = predictions_full[:, L_eff-1, :]
        y_last = y_true_full[:, L_eff-1]
        crps_last = calculate_crps(preds_last, y_last, taus, f"{model_name} (last horizon)")
        total_loss_last, per_quantile_losses_last = calculate_pinball_loss(
            preds_last, y_last, 
            taus, 
            f"{model_name} (last horizon)",
            verbose=False  # Suppress detailed per-quantile metrics for last horizon
        )
        
        # Averaged across horizons with detailed per-quantile output
        crps_avg = calculate_crps_multi_horizon_avg(
            predictions_full[:, :L_eff, :], y_true_full[:, :L_eff], taus, model_name
        )
        total_loss_avg, per_quantile_losses_avg = calculate_pinball_loss_multi_horizon_avg(
            predictions_full[:, :L_eff, :], y_true_full[:, :L_eff], taus, model_name
        )
        
        # Summary
        print(f"\n{model_name} Summary:")
        print(f"  CRPS (last): {crps_last:.6f}")
        print(f"  CRPS (avg over {L_eff} horizons): {crps_avg:.6f}")
        print(f"  Total Pinball Loss (last): {total_loss_last:.6f}")
        print(f"  Total Pinball Loss (avg over {L_eff} horizons): {total_loss_avg:.6f}")

def find_model_path(filename):
    """Find model path checking tune/ subdirectory first, then falling back to main directory.
    
    Parameters
    ----------
    filename : str
        Model filename (e.g., "mqrnn_model_with_proxy.pt")
        
    Returns
    -------
    pathlib.Path or None
        Path to the model file if found, None otherwise
    """
    # Check tune/ subdirectory first
    tune_path = cfg.DEMAND_MODELS_DIR / "tune" / filename
    if tune_path.exists():
        return tune_path
    
    # Fall back to main directory
    main_path = cfg.DEMAND_MODELS_DIR / filename
    if main_path.exists():
        return main_path
    
    return None

def main():
    """Main function to evaluate demand models."""
    
    print("=== DEMAND MODEL EVALUATION ===\n")
    
    # Evaluate tuned MQRNN models first (if they exist)
    # Check in tune/ subdirectory first, then fallback to main directory
    tuned_with_proxy_path = find_model_path("mqrnn_model_with_proxy_tuned.pt")
    tuned_no_proxy_path = find_model_path("mqrnn_model_tuned.pt")
    
    if tuned_with_proxy_path:
        evaluate_mqrnn_model_helper(tuned_with_proxy_path, "MQRNN (tuned, with proxy)")
    if tuned_no_proxy_path:
        evaluate_mqrnn_model_helper(tuned_no_proxy_path, "MQRNN (tuned, no proxy)")
    
    # Evaluate regular MQRNN models (if they exist and tuned versions weren't evaluated)
    demand_model_path = find_model_path("mqrnn_model_with_proxy.pt")
    demand_model_path_no_proxy = find_model_path("mqrnn_model.pt")
    
    if demand_model_path:
        evaluate_mqrnn_model_helper(demand_model_path, "MQRNN (with proxy)")
    if demand_model_path_no_proxy:
        evaluate_mqrnn_model_helper(demand_model_path_no_proxy, "MQRNN (no proxy)")
    
    # Evaluate demand QRF model
    demand_qrf_path = cfg.DEMAND_QR_MODELS_DIR / "demand_qrf_model_with_proxy.joblib"
    if demand_qrf_path.exists():
        evaluate_demand_qrf_model(demand_qrf_path)
    else:
        print(f"Demand QRF model not found at {demand_qrf_path}")
    
    # Evaluate demand QR models (mean/std variants)
    evaluate_demand_qr_family('demand_model_global_xgboost_meanstd')
    evaluate_demand_qr_family('demand_model_global_sklearn_meanstd')

if __name__ == "__main__":
    main() 