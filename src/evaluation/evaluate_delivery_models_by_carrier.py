import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

import src.config as cfg
from src.model.time_quantile_model import TimeQuantileModel
from src.utils import pinball_loss
from src.training.delivery_time.common import load_split_data


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def crps(y_true: np.ndarray, y_pred_samples: np.ndarray, sample_weight=None) -> float:
    """CRPS for samples y_pred_samples with true targets y_true."""
    num_samples = y_pred_samples.shape[0]
    absolute_error = np.mean(np.abs(y_pred_samples - y_true), axis=0)

    if num_samples == 1:
        return float(np.average(absolute_error, weights=sample_weight))

    y_sorted = np.sort(y_pred_samples, axis=0)
    diff = y_sorted[1:] - y_sorted[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)

    per_obs = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return float(np.average(per_obs, weights=sample_weight))


def sample_from_quantiles(
    quantile_predictions: np.ndarray, quantiles: List[float], num_samples: int = 1000
) -> np.ndarray:
    """Inverse-CDF sampling from quantile predictions (shape: N x Q)."""
    n, _ = quantile_predictions.shape
    samples = np.zeros((num_samples, n))
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    u = rng.random((num_samples, n))

    for i in range(n):
        sorted_idx = np.argsort(quantile_predictions[i])
        sorted_q = np.array(quantiles)[sorted_idx]
        sorted_pred = quantile_predictions[i][sorted_idx]
        samples[:, i] = np.interp(u[:, i], sorted_q, sorted_pred)

    return samples


def calculate_crps(predictions: np.ndarray, y_true: np.ndarray, quantiles: List[float], tag: str) -> float:
    print(f"Calculating CRPS for {tag} ...")
    samples = sample_from_quantiles(predictions, quantiles, num_samples=1000)
    score = crps(y_true, samples)
    print(f"CRPS for {tag}: {score:.6f}")
    return score


def calculate_pinball_loss(
    predictions: np.ndarray, y_true: np.ndarray, quantiles: List[float], tag: str
) -> Tuple[float, List[float]]:
    print(f"Calculating pinball loss for {tag} ...")
    preds_t = torch.tensor(predictions, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)
    taus_t = torch.tensor(quantiles, dtype=torch.float32)

    total = pinball_loss(preds_t, y_t, taus_t).item()
    per_q: List[float] = []
    for j, tau in enumerate(quantiles):
        q_pred = preds_t[:, j : j + 1]
        q_tau = taus_t[j : j + 1]
        per_q.append(pinball_loss(q_pred, y_t, q_tau).item())

    print(f"Total pinball loss ({tag}): {total:.6f}")
    for tau, loss in zip(quantiles, per_q):
        print(f"  Quantile {tau:.2f}: {loss:.6f}")
    return total, per_q


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_df() -> pd.DataFrame:
    """
    Load the *carrier-specific* evaluation dataframe with engineered features.
    
    We reuse the same pipeline as training by calling `load_split_data` with
    `use_cs_data=True` and `train_on_proxy=True`, and treat its evaluation
    split as our test set for per-carrier evaluation.
    """
    _, df_eval = load_split_data(train_on_proxy=True, use_cs_data=True)
    return df_eval


def filter_carrier(df: pd.DataFrame, carrier_id: int) -> pd.DataFrame:
    return df[df["carrier_service_id_anon"] == carrier_id].copy()


def extract_carriers(models_dir: Path) -> List[int]:
    """Extract carrier IDs from model filenames (both with and without _with_proxy suffix).
    
    DL models are checked in tune/ subdirectory (if it exists), then main directory.
    QRF, XGBoost, and sklearn models are only checked in main directory.
    """
    carriers = set()
    
    # DL model patterns (checked in both tune/ and main directory)
    dl_patterns = [
        r"dl_model_(\d+)_with_proxy\.pt",
        r"dl_model_(\d+)\.pt",  # Exclude combo files: dl_model_*_combo_*.pt
    ]
    
    # QRF, XGBoost, sklearn patterns (only checked in main directory)
    other_patterns = [
        r"rf_model_(\d+)_with_proxy\.joblib",
        r"rf_model_(\d+)\.joblib",
        r"delivery_model_xgboost_(\d+)_with_proxy\.joblib",
        r"delivery_model_xgboost_(\d+)\.joblib",
        r"delivery_model_sklearn_(\d+)_with_proxy\.joblib",
        r"delivery_model_sklearn_(\d+)\.joblib",
    ]
    
    # Check tune/ subdirectory for DL models only
    tune_dir = models_dir / "tune"
    if tune_dir.exists():
        for path in tune_dir.glob("*"):
            if path.is_file():
                name = path.name
                # Skip combo files (grid search intermediate files)
                if "_combo_" in name:
                    continue
                for pat in dl_patterns:
                    m = re.match(pat, name)
                    if m:
                        carriers.add(int(m.group(1)))
    
    # Check main directory for all model types
    for path in models_dir.glob("*"):
        if path.is_file():
            name = path.name
            # Skip combo files
            if "_combo_" in name:
                continue
            # Check DL patterns
            for pat in dl_patterns:
                m = re.match(pat, name)
                if m:
                    carriers.add(int(m.group(1)))
            # Check other model patterns
            for pat in other_patterns:
                m = re.match(pat, name)
                if m:
                    carriers.add(int(m.group(1)))
    
    return sorted(carriers)


# ---------------------------------------------------------------------------
# Model evaluators
# ---------------------------------------------------------------------------

def evaluate_dl_carrier(model_path: Path, df_test: pd.DataFrame, carrier_id: int, quantiles: List[float]):
    print(f"\nEvaluating DL model for carrier {carrier_id}: {model_path}")
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)

    # Models trained by by-carrier `train_dl_rf.py` store a compact bundle
    # with `model_state_dict` and a nested `model_params` dict.
    state_dict = bundle["model_state_dict"]
    x_scaler = bundle["x_scaler"]
    categorical_encoders = bundle["categorical_encoders"]
    vocab_sizes = bundle["vocab_sizes"]
    mp = bundle.get("model_params", {})

    # Some older/tuned bundles may miss certain entries in `model_params`.
    # Fall back to sensible defaults where we can infer them safely.
    if "numerical_dim" not in mp:
        # Must match the number of numerical features used during training.
        mp["numerical_dim"] = len(cfg.DELIVERY_DL_NUMERICAL_FEATURES)
        print(f"  Warning: 'numerical_dim' missing in model_params, using fallback: {mp['numerical_dim']}")
    if "hidden_dim" not in mp:
        mp["hidden_dim"] = cfg.DELIVERY_DL_HIDDEN_DIM
        print(f"  Warning: 'hidden_dim' missing in model_params, using fallback: {mp['hidden_dim']}")
    if "n_layers" not in mp:
        mp["n_layers"] = cfg.DELIVERY_DL_N_LAYERS
        print(f"  Warning: 'n_layers' missing in model_params, using fallback: {mp['n_layers']}")
    if "dropout_p" not in mp:
        mp["dropout_p"] = cfg.DELIVERY_DL_DROPOUT_P
        print(f"  Warning: 'dropout_p' missing in model_params, using fallback: {mp['dropout_p']}")
    if "dc_ori_embedding_dim" not in mp:
        mp["dc_ori_embedding_dim"] = cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM
        print(f"  Warning: 'dc_ori_embedding_dim' missing in model_params, using fallback: {mp['dc_ori_embedding_dim']}")
    if "dc_des_embedding_dim" not in mp:
        mp["dc_des_embedding_dim"] = cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM
        print(f"  Warning: 'dc_des_embedding_dim' missing in model_params, using fallback: {mp['dc_des_embedding_dim']}")
    
    # Print hyperparameters if available (from grid search)
    combo_idx = bundle.get("combination_idx")
    if combo_idx is not None:
        print(f"  Grid search combination index: {combo_idx}")
        # Try to load grid file to show HP values
        grid_file = Path("data/delivery_dl_grid_search_combinations.json")
        if grid_file.exists():
            try:
                with open(grid_file, 'r') as f:
                    grid_data = json.load(f)
                combinations = grid_data.get("combinations", [])
                if 0 <= combo_idx < len(combinations):
                    hp_values = combinations[combo_idx]
                    print(f"  Hyperparameters:")
                    for key, value in sorted(hp_values.items()):
                        if key != 'epochs':
                            print(f"    {key}: {value}")
            except Exception as e:
                print(f"  (Could not load grid file to show HP values: {e})")
    numerical_dim = mp["numerical_dim"]
    hidden_dim = mp["hidden_dim"]
    n_layers = mp["n_layers"]
    dropout_p = mp["dropout_p"]
    dc_ori_embedding_dim = mp["dc_ori_embedding_dim"]
    dc_des_embedding_dim = mp["dc_des_embedding_dim"]

    df_cs = filter_carrier(df_test, carrier_id)
    if df_cs.empty:
        print("  No test rows for this carrier; skipping.")
        return None

    features = cfg.DELIVERY_TIME_FEATURES
    X = df_cs[features]
    y = df_cs[cfg.DELIVERY_TIME_TARGET]
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    if X.empty:
        print("  No valid targets after filtering; skipping.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeQuantileModel(
        numerical_dim=numerical_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=True,
        dropout_p=dropout_p,
        dc_ori_vocab_size=vocab_sizes.get("dc_ori"),
        dc_des_vocab_size=vocab_sizes.get("dc_des"),
        dc_ori_embedding_dim=dc_ori_embedding_dim,
        dc_des_embedding_dim=dc_des_embedding_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    X_num = X[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    X_cat = X[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
    X_num_scaled = x_scaler.transform(X_num)
    X_num_t = torch.tensor(X_num_scaled, dtype=torch.float32, device=device)
    X_cat_encoded = {}
    for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
        if cat_feature in X_cat.columns:
            enc = categorical_encoders[cat_feature].transform(X_cat[cat_feature].astype(str))
            X_cat_encoded[cat_feature] = torch.tensor(enc, dtype=torch.long, device=device)

    y_t = torch.tensor(y.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(
            X_num_t,
            X_cat_encoded.get("dc_ori"),
            X_cat_encoded.get("dc_des"),
        ).cpu().numpy()

    preds_round = np.round(preds).astype(int)
    y_round = np.round(y_t.cpu().numpy()).astype(int)

    crps_score = calculate_crps(preds_round, y_round, quantiles, f"DL carrier {carrier_id}")
    pinball_total, _ = calculate_pinball_loss(preds_round, y_round, quantiles, f"DL carrier {carrier_id}")
    return {"carrier": carrier_id, "crps": crps_score, "pinball": pinball_total}


def evaluate_qrf_carrier(model_path: Path, df_test: pd.DataFrame, carrier_id: int, quantiles: List[float]):
    print(f"\nEvaluating QRF model for carrier {carrier_id}: {model_path}")
    bundle = joblib.load(model_path)
    model = bundle.get("model")
    x_scaler = bundle.get("x_scaler")
    if model is None or x_scaler is None:
        print("  Missing 'model' or 'x_scaler' in QRF bundle; skipping.")
        return None

    df_cs = filter_carrier(df_test, carrier_id)
    if df_cs.empty:
        print("  No test rows for this carrier; skipping.")
        return None

    # RF was trained on scaled numerical features only
    X_num = df_cs[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    y = df_cs[cfg.DELIVERY_TIME_TARGET].values
    mask = ~np.isnan(y)
    X_num = X_num.loc[mask]
    y = y[mask]
    if X_num.empty:
        print("  No valid targets after filtering; skipping.")
        return None

    X_num_scaled = x_scaler.transform(X_num)

    preds = model.predict(X_num_scaled, quantiles=quantiles)
    preds_round = np.round(preds).astype(int)
    y_round = np.round(y).astype(int)

    crps_score = calculate_crps(preds_round, y_round, quantiles, f"QRF carrier {carrier_id}")
    pinball_total, _ = calculate_pinball_loss(preds_round, y_round, quantiles, f"QRF carrier {carrier_id}")
    return {"carrier": carrier_id, "crps": crps_score, "pinball": pinball_total}


def evaluate_qr_family_carrier(
    model_path: Path, df_test: pd.DataFrame, carrier_id: int, quantiles: List[float], family_tag: str
):
    print(f"\nEvaluating {family_tag} models for carrier {carrier_id}: {model_path}")
    bundle = joblib.load(model_path)

    models = bundle.get("models")
    stored_quantiles = bundle.get("quantiles")
    cat_encoders = bundle.get("categorical_encoders")
    x_scaler = bundle.get("x_scaler")

    if models is None or stored_quantiles is None:
        print("  Missing models/quantiles; skipping.")
        return None
    if x_scaler is None:
        print("  Missing scaler; skipping to avoid incorrect evaluation.")
        return None

    tau_to_model: Dict[float, object] = {}
    for q, mdl in models.items():
        tau_to_model[float(q)] = mdl

    df_cs = filter_carrier(df_test, carrier_id)
    if df_cs.empty:
        print("  No test rows for this carrier; skipping.")
        return None

    X = df_cs[cfg.DELIVERY_TIME_FEATURES]
    y = df_cs[cfg.DELIVERY_TIME_TARGET]
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    if X.empty:
        print("  No valid targets after filtering; skipping.")
        return None

    X_num = X[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    X_cat = X[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
    X_num_scaled = x_scaler.transform(X_num)

    X_cat_encoded = {}
    for cat_feature in cfg.DELIVERY_DL_CATEGORICAL_FEATURES:
        if cat_feature in X_cat.columns:
            enc = cat_encoders[cat_feature].transform(X_cat[cat_feature].astype(str))
            X_cat_encoded[cat_feature] = enc.reshape(-1, 1)

    X_mat = np.concatenate([X_num_scaled] + list(X_cat_encoded.values()), axis=1) if X_cat_encoded else X_num_scaled
    preds = np.zeros((len(X_mat), len(quantiles)))

    for j, tau in enumerate(quantiles):
        mdl = tau_to_model.get(tau)
        if mdl is None:
            preds[:, j] = np.nan
        else:
            preds[:, j] = mdl.predict(X_mat)

    mask = ~np.isnan(preds).any(axis=1)
    preds = preds[mask]
    y_arr = y.values[mask]
    if preds.size == 0:
        print("  No predictions after masking; skipping.")
        return None

    preds_round = np.round(preds).astype(int)
    y_round = np.round(y_arr).astype(int)

    crps_score = calculate_crps(preds_round, y_round, quantiles, f"{family_tag} carrier {carrier_id}")
    pinball_total, _ = calculate_pinball_loss(preds_round, y_round, quantiles, f"{family_tag} carrier {carrier_id}")
    return {"carrier": carrier_id, "crps": crps_score, "pinball": pinball_total}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== DELIVERY MODEL EVALUATION (CARRIER-SPECIFIC) ===\n")

    models_dir = Path(cfg.DATA_DIR) / "models" / "delivery_time_cs"
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return

    carriers = extract_carriers(models_dir)
    if not carriers:
        print(f"No carrier-specific models found in {models_dir}")
        return

    print(f"Found carriers: {carriers}")

    quantiles = cfg.DELIVERY_TIME_QUANTILES
    df_test = load_test_df()

    # Check if tune/ directory has any DL models - if so, prefer tune/
    tune_dir = models_dir / "tune"
    use_tune_dir = tune_dir.exists()

    # Metric collectors - separate results for with_proxy and without_proxy
    results: Dict[str, List[Dict[str, float]]] = {
        "dl_with_proxy": [],
        "dl_no_proxy": [],
        "qrf_with_proxy": [],
        "qrf_no_proxy": [],
        "xgboost_with_proxy": [],
        "xgboost_no_proxy": [],
        "sklearn_with_proxy": [],
        "sklearn_no_proxy": [],
    }

    for carrier_id in carriers:
        carrier_tag = str(carrier_id)

        # DL - check both with_proxy and no_proxy versions
        # Try with_proxy first
        if use_tune_dir:
            dl_path_with = models_dir / "tune" / f"dl_model_{carrier_tag}_with_proxy.pt"
            dl_path_no = models_dir / "tune" / f"dl_model_{carrier_tag}.pt"
        else:
            dl_path_with = models_dir / f"dl_model_{carrier_tag}_with_proxy.pt"
            dl_path_no = models_dir / f"dl_model_{carrier_tag}.pt"
        
        if dl_path_with.exists():
            res = evaluate_dl_carrier(dl_path_with, df_test, carrier_id, quantiles)
            if res:
                res["train_mode"] = "with_proxy"
                results["dl_with_proxy"].append(res)
        
        if dl_path_no.exists():
            res = evaluate_dl_carrier(dl_path_no, df_test, carrier_id, quantiles)
            if res:
                res["train_mode"] = "no_proxy"
                results["dl_no_proxy"].append(res)

        # QRF - check both versions (only in main directory, not tune/)
        qrf_path_with = models_dir / f"rf_model_{carrier_tag}_with_proxy.joblib"
        qrf_path_no = models_dir / f"rf_model_{carrier_tag}.joblib"
        
        if qrf_path_with.exists():
            res = evaluate_qrf_carrier(qrf_path_with, df_test, carrier_id, quantiles)
            if res:
                res["train_mode"] = "with_proxy"
                results["qrf_with_proxy"].append(res)
        
        if qrf_path_no.exists():
            res = evaluate_qrf_carrier(qrf_path_no, df_test, carrier_id, quantiles)
            if res:
                res["train_mode"] = "no_proxy"
                results["qrf_no_proxy"].append(res)

        # XGBoost - check both versions (only in main directory, not tune/)
        xgb_path_with = models_dir / f"delivery_model_xgboost_{carrier_tag}_with_proxy.joblib"
        xgb_path_no = models_dir / f"delivery_model_xgboost_{carrier_tag}.joblib"
        
        if xgb_path_with.exists():
            res = evaluate_qr_family_carrier(xgb_path_with, df_test, carrier_id, quantiles, "XGBoost")
            if res:
                res["train_mode"] = "with_proxy"
                results["xgboost_with_proxy"].append(res)
        
        if xgb_path_no.exists():
            res = evaluate_qr_family_carrier(xgb_path_no, df_test, carrier_id, quantiles, "XGBoost")
            if res:
                res["train_mode"] = "no_proxy"
                results["xgboost_no_proxy"].append(res)

        # Sklearn - check both versions (only in main directory, not tune/)
        skl_path_with = models_dir / f"delivery_model_sklearn_{carrier_tag}_with_proxy.joblib"
        skl_path_no = models_dir / f"delivery_model_sklearn_{carrier_tag}.joblib"
        
        if skl_path_with.exists():
            res = evaluate_qr_family_carrier(skl_path_with, df_test, carrier_id, quantiles, "Sklearn")
            if res:
                res["train_mode"] = "with_proxy"
                results["sklearn_with_proxy"].append(res)
        
        if skl_path_no.exists():
            res = evaluate_qr_family_carrier(skl_path_no, df_test, carrier_id, quantiles, "Sklearn")
            if res:
                res["train_mode"] = "no_proxy"
                results["sklearn_no_proxy"].append(res)

    print("\n" + "=" * 70)
    print("Summary (mean across carriers with available models)")
    print("=" * 70)

    # Group results by model type and training mode
    model_types = ["dl", "qrf", "xgboost", "sklearn"]
    
    for model_type in model_types:
        with_proxy_key = f"{model_type}_with_proxy"
        no_proxy_key = f"{model_type}_no_proxy"
        
        with_proxy_items = results.get(with_proxy_key, [])
        no_proxy_items = results.get(no_proxy_key, [])
        
        if with_proxy_items:
            mean_pin = float(np.mean([r["pinball"] for r in with_proxy_items]))
            mean_crps = float(np.mean([r["crps"] for r in with_proxy_items]))
            print(f"{model_type.upper()} (with_proxy): mean pinball={mean_pin:.6f}, mean CRPS={mean_crps:.6f} over {len(with_proxy_items)} carriers")
        
        if no_proxy_items:
            mean_pin = float(np.mean([r["pinball"] for r in no_proxy_items]))
            mean_crps = float(np.mean([r["crps"] for r in no_proxy_items]))
            print(f"{model_type.upper()} (no_proxy): mean pinball={mean_pin:.6f}, mean CRPS={mean_crps:.6f} over {len(no_proxy_items)} carriers")
        
        if not with_proxy_items and not no_proxy_items:
            print(f"{model_type.upper()}: no carriers evaluated.")


if __name__ == "__main__":
    main()

