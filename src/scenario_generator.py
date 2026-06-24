import numpy as np
import pandas as pd
import torch
from pathlib import Path
import joblib
import time
import json
import warnings

import src.config as cfg
from src.model.mqrnn import MQRNN
from src.model.time_quantile_model import TimeQuantileModel
from src.utils import sample_paths, build_quantile_grid
from src.utils import calculate_lookahead_periods
from src.data_utils import load_precomputed_order_dc_features
from typing import Union, Dict, Tuple
from src.empirical_scenarios import _get_cached_training_data


# ──────────────────── GLOBAL CACHES ─────────────────────────────
_DEMAND_MODEL_CACHE: dict[str, dict] = {}  # Just one global model
_DELIVERY_MODEL_CACHE: dict[str, dict] = {} # Just one global model
_DEMAND_FEATURES_CACHE: dict[str, dict] = {}  # period -> features
_DC_FEATURES_CACHE: dict[tuple[str, str], pd.DataFrame] = {}  # (date, order_set) -> features
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache: (order_set, lookahead_L) -> {sku_idx:int -> hist_max_total:float}
_DEMAND_HIST_UPPER_CACHE: dict[tuple[str, int], dict[int, float]] = {}
_DEMAND_MEAN_TOTAL_CACHE: dict[tuple[str, int], dict[int, float]] = {}
_SKU_TO_IDX_CACHE: dict[str, int] | None = None


def _encode_with_unknown(encoder, values: list[str]) -> np.ndarray:
    """Label-encode values, mapping unseen tokens to the configured unknown token."""
    classes = getattr(encoder, "classes_", None)
    if classes is None or len(classes) == 0:
        raise ValueError("Label encoder has no classes_ defined.")
    class_str = np.asarray(classes, dtype=str)
    class_set = set(class_str.tolist())
    unk = getattr(cfg, "DELIVERY_DL_UNKNOWN_TOKEN", None)
    if unk and unk not in class_set:
        unk = None
    fallback = unk or class_str[0]
    normalized = [val if val in class_set else fallback for val in values]
    return encoder.transform(normalized)


def _build_feature_window_index(dt_series: pd.Series, sku_idx_array: np.ndarray) -> dict[tuple[int, int], int]:
    """Create a lookup from (sku_idx, bucket_start_ns) -> window row index."""
    dt_values = dt_series.values.astype("datetime64[ns]").astype("int64")
    window_index: dict[tuple[int, int], int] = {}
    for idx, (sku_idx, dt_ns) in enumerate(zip(sku_idx_array, dt_values, strict=False)):
        try:
            key = (int(sku_idx), int(dt_ns))
        except (TypeError, ValueError):
            continue
        window_index[key] = idx
    return window_index


def _get_hist_upper_by_sku(L: int, order_set: str) -> dict[int, float]:
    """Return per-SKU historical max cumulative demand over first L periods.

    The result is cached per (order_set, L) for efficient on-the-fly inference.
    """
    key = (order_set, int(L))
    if key in _DEMAND_HIST_UPPER_CACHE:
        return _DEMAND_HIST_UPPER_CACHE[key]

    cached = _get_cached_training_data(order_set, verbose=False)
    demand_samples = cached.get('demand_samples')
    sku_indices = cached.get('sku_indices')
    if demand_samples is None or sku_indices is None or len(demand_samples) == 0:
        _DEMAND_HIST_UPPER_CACHE[key] = {}
        return _DEMAND_HIST_UPPER_CACHE[key]

    L_eff = min(int(L), demand_samples.shape[1])
    totals = demand_samples[:, :L_eff].sum(axis=1).astype(np.float32)
    # Group max per SKU index
    df = pd.DataFrame({'sku': sku_indices, 'tot': totals})
    hist_max = df.groupby('sku', sort=False)['tot'].max()
    mapping = {int(k): float(v) for k, v in hist_max.items()}
    _DEMAND_HIST_UPPER_CACHE[key] = mapping
    return mapping


def _get_mean_total_by_sku(L: int, order_set: str) -> dict[int, float]:
    """Return per-SKU mean cumulative demand over first L periods.

    Cached per (order_set, L) for efficient hybrid classification.
    """
    key = (order_set, int(L))
    if key in _DEMAND_MEAN_TOTAL_CACHE:
        return _DEMAND_MEAN_TOTAL_CACHE[key]

    cached = _get_cached_training_data(order_set, verbose=False)
    demand_samples = cached.get('demand_samples')
    sku_indices = cached.get('sku_indices')
    if demand_samples is None or sku_indices is None or len(demand_samples) == 0:
        _DEMAND_MEAN_TOTAL_CACHE[key] = {}
        return _DEMAND_MEAN_TOTAL_CACHE[key]

    L_eff = min(int(L), demand_samples.shape[1])
    totals = demand_samples[:, :L_eff].sum(axis=1).astype(np.float32)
    df = pd.DataFrame({'sku': sku_indices, 'tot': totals})
    mean_by_sku = df.groupby('sku', sort=False)['tot'].mean()
    mapping = {int(k): float(v) for k, v in mean_by_sku.items()}
    _DEMAND_MEAN_TOTAL_CACHE[key] = mapping
    return mapping


def _get_sku_to_idx_map(project_root: Path) -> dict[str, int]:
    global _SKU_TO_IDX_CACHE
    if _SKU_TO_IDX_CACHE is not None:
        return _SKU_TO_IDX_CACHE
    mapping_path = project_root / "data" / "processed" / "mappings" / "sku_to_idx.json"
    if mapping_path.exists():
        try:
            _SKU_TO_IDX_CACHE = json.loads(mapping_path.read_text())
        except Exception:
            _SKU_TO_IDX_CACHE = {}
    else:
        _SKU_TO_IDX_CACHE = {}
    return _SKU_TO_IDX_CACHE

def _load_features_from_eval_set(
        project_root: Path,
        order_items: pd.DataFrame,
        period: str = "test",
        verbose: bool = False,
        pre_scale: bool = True
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """
    For every SKU present in `order_items`, locate the sliding window in
    the MQ‑RNN evaluation .npz whose *last history bucket* ends at the
    bucket that the incoming order falls into.  Return the tensors
    (xh, yh, xp, sku_idx, brand_idx) that a forward pass needs.

    Expected columns in `order_items`: 'sku_ID', 'order_time'.
    
    Args:
        pre_scale: If True, pre-scale the features using the demand model scalers
    """
    # Check cache first
    if period in _DEMAND_FEATURES_CACHE:
        feats_dict = _DEMAND_FEATURES_CACHE[period]
    else:
        required_cols = {"sku_ID", "order_time"}
        if not required_cols.issubset(order_items.columns):
            print(f"  Error: order_items must contain {required_cols}")
            return {}

        # Load the pre‑computed feature file
        npz_name = "mqrnn_test.npz" if period == "test" else "mqrnn_proxy_train.npz"
        features_path = project_root / "data" / "processed" / npz_name

        if not features_path.exists():
            print(f"  Error: Features file not found at {features_path}")
            return {}

        feats = np.load(features_path)
        feats_dict = {
            'dt_arr': pd.to_datetime(feats["dt"]),
            'xh_all': feats["xh"], 'yh_all': feats["yh"], 'xp_all': feats["xp"],
            'sku_idx_all': feats["sku_idx"], 'brand_idx_all': feats["brand_idx"],
            'hist_len': feats["xh"].shape[1],
            'freq_td': pd.Timedelta(1, unit=cfg.DEMAND_MODEL_TIME_PERIOD_FREQ)
        }
        feats_dict['window_index'] = _build_feature_window_index(feats_dict['dt_arr'], feats_dict['sku_idx_all'])
        _DEMAND_FEATURES_CACHE[period] = feats_dict

    # Load SKU mapping once
    if 'sku_to_idx' not in feats_dict:
        mapping_path = project_root / "data" / "processed" / "mappings" / "sku_to_idx.json"
        feats_dict['sku_to_idx'] = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}

    if 'window_index' not in feats_dict:
        feats_dict['window_index'] = _build_feature_window_index(feats_dict['dt_arr'], feats_dict['sku_idx_all'])

    # Load scalers if pre-scaling is requested
    past_scaler = None
    horizon_scaler = None
    if pre_scale:
        model_info = _get_demand_model(project_root, period=period, verbose=verbose)
        if model_info:
            past_scaler = model_info.get('past_scaler')
            horizon_scaler = model_info.get('horizon_scaler')

    order_features: Dict[str, Tuple[torch.Tensor, ...]] = {}

    # Process order items
    window_index = feats_dict.get('window_index', {})
    dt_arr = feats_dict['dt_arr']
    freq_td = feats_dict['freq_td']

    for _, row in order_items.iterrows():
        sku_id = row["sku_ID"]
        order_time_value = row.get("order_time")
        if pd.isna(order_time_value):
            continue
        order_time = pd.Timestamp(order_time_value)
        bucket_start = order_time.floor(cfg.DEMAND_MODEL_TIME_PERIOD_FREQ)

        idx_candidates: np.ndarray | None = None
        sku_idx_expected = feats_dict['sku_to_idx'].get(str(sku_id)) if feats_dict['sku_to_idx'] else None
        if window_index and sku_idx_expected is not None:
            bucket_key = int(bucket_start.value)
            idx = window_index.get((int(sku_idx_expected), bucket_key))
            if idx is not None:
                idx_candidates = np.array([idx], dtype=int)

        if idx_candidates is None:
            offset = ((bucket_start - dt_arr) // freq_td).astype("int64")
            mask = offset == 0

            if sku_idx_expected is not None:
                mask &= (feats_dict['sku_idx_all'] == sku_idx_expected)

            idx_candidates = np.where(mask)[0]

        if len(idx_candidates) == 0:
            if verbose:
                print(f"    ⚠️  No window found for SKU {sku_id} @ {bucket_start}")
            continue

        k = int(idx_candidates[0])
        
        # Load raw features
        past_feat = torch.from_numpy(feats_dict['xh_all'][k:k+1]).float()
        past_targets = torch.from_numpy(feats_dict['yh_all'][k:k+1]).float()
        horizon_feat = torch.from_numpy(feats_dict['xp_all'][k:k+1]).float()
        sku_idx = torch.tensor([int(feats_dict['sku_idx_all'][k])], dtype=torch.long)
        brand_idx = torch.tensor([int(feats_dict['brand_idx_all'][k])], dtype=torch.long)
        
        # Pre-scale features if requested and scalers are available
        if pre_scale and past_scaler and horizon_scaler:
            past_feat_flat = past_feat.view(-1, past_feat.shape[-1])
            horizon_feat_flat = horizon_feat.view(-1, horizon_feat.shape[-1])
            
            past_feat = torch.tensor(
                past_scaler.transform(past_feat_flat).reshape(past_feat.shape), 
                dtype=torch.float32
            )
            horizon_feat = torch.tensor(
                horizon_scaler.transform(horizon_feat_flat).reshape(horizon_feat.shape), 
                dtype=torch.float32
            )
        
        order_features[sku_id] = (past_feat, past_targets, horizon_feat, sku_idx, brand_idx)
        if verbose:
            print(f"    ✓ Features loaded for SKU {sku_id} (window #{k})")

    return order_features


def _get_demand_model(project_root: Path, period: str = 'test', verbose: bool = False):
    """Load (once) and return the global demand model."""
    mode_suffix = "_with_proxy" if period == 'test' else ""
    cache_key = f"global_demand{mode_suffix}"

    if cache_key in _DEMAND_MODEL_CACHE:
        return _DEMAND_MODEL_CACHE[cache_key]

    # Checkpoints from `src.training.demand.train_mqrnn` carry a "_tuned" suffix;
    # prefer that, then a plain name. Warn loudly (rather than silently) if neither
    # exists so users do not unknowingly fall back to the empirical sampler.
    demand_dir = project_root / 'data' / 'models' / 'demand'
    model_path = next(
        (demand_dir / name for name in
         (f"mqrnn_model{mode_suffix}_tuned.pt", f"mqrnn_model{mode_suffix}.pt")
         if (demand_dir / name).exists()),
        None,
    )
    if model_path is None:
        warnings.warn(
            f"Demand model not found in {demand_dir} (looked for "
            f"mqrnn_model{mode_suffix}_tuned.pt and mqrnn_model{mode_suffix}.pt); "
            "demand scenarios fall back to the empirical sampler. Train it with "
            "`python -m src.training.demand.train_mqrnn`.",
            stacklevel=2,
        )
        return None
        
    model_data = torch.load(model_path, map_location=_DEVICE, weights_only=False)
    hp = model_data['hyperparameters']
    meta = model_data['meta']
    
    model = MQRNN(
        num_cal=meta["num_cal"], num_ord=meta["num_ord"],
        num_skus=meta["num_skus"], num_brands=meta["num_brands"],
        sku_emb=hp.get("sku_emb", cfg.DEMAND_MODEL_SKU_EMBEDDING_DIM),
        brand_emb=hp.get("brand_emb", cfg.DEMAND_MODEL_BRAND_EMBEDDING_DIM),
        hidden=hp["hidden_dim"], 
        ctx=hp.get("context_dim", cfg.DEMAND_MODEL_CONTEXT_DIM),
        num_q=len(cfg.DEMAND_MODEL_QUANTILES),
        Lp=meta["Lp"], Lh=meta["Lh"],
        layers=hp["lstm_n_layers"],
        dropout=hp.get("dropout_p", 0) > 0,
        dropout_p=hp.get("dropout_p", 0.0),
        bidirectional=hp.get("bidirectional", cfg.DEMAND_MODEL_BIDIRECTIONAL)
    ).to(_DEVICE)
    
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    # Load scalers (same naming as the resolved model checkpoint)
    scalers_path = model_path.parent / model_path.name.replace("mqrnn_model", "mqrnn_scalers", 1)
    scalers_data = torch.load(scalers_path, weights_only=False) if scalers_path.exists() else {}

    _DEMAND_MODEL_CACHE[cache_key] = {
        'model': model,
        'past_scaler': scalers_data.get('scaler_hist'),
        'horizon_scaler': scalers_data.get('scaler_pred')
    }
    return _DEMAND_MODEL_CACHE[cache_key]


def _get_delivery_model(project_root: Path, period: str = 'test', carrier_service_id: int | None = None, verbose: bool = False):
    """Load (once) and return the delivery time model and scalers.
    
    If carrier_service_id is provided, loads per-carrier-service model.
    Otherwise, falls back to global model for backward compatibility.
    """
    mode_suffix = "_with_proxy" if period == 'test' else ""
    
    # Use per-carrier-service model if carrier_service_id is provided
    if carrier_service_id is not None:
        cache_key = f"cs_{carrier_service_id}_delivery{mode_suffix}"
        if cache_key in _DELIVERY_MODEL_CACHE:
            return _DELIVERY_MODEL_CACHE[cache_key]
        
        # Try per-cs models in order: DL, RF, then quantile models
        cs_models_dir = project_root / cfg.DELIVERY_MODELS_CS_DIR
        # Try with mode_suffix first, then fallback to without suffix for backward compatibility
        dl_model_path = cs_models_dir / f"dl_model_{carrier_service_id}{mode_suffix}.pt"
        if not dl_model_path.exists() and mode_suffix:
            dl_model_path = cs_models_dir / f"dl_model_{carrier_service_id}.pt"
        rf_model_path = cs_models_dir / f"rf_model_{carrier_service_id}{mode_suffix}.joblib"
        if not rf_model_path.exists() and mode_suffix:
            rf_model_path = cs_models_dir / f"rf_model_{carrier_service_id}.joblib"
        # Try common quantile model types (these may not have mode_suffix yet)
        xgb_model_path = cs_models_dir / f"delivery_model_xgboost_{carrier_service_id}{mode_suffix}.joblib"
        if not xgb_model_path.exists() and mode_suffix:
            xgb_model_path = cs_models_dir / f"delivery_model_xgboost_{carrier_service_id}.joblib"
        sklearn_model_path = cs_models_dir / f"delivery_model_sklearn_{carrier_service_id}{mode_suffix}.joblib"
        if not sklearn_model_path.exists() and mode_suffix:
            sklearn_model_path = cs_models_dir / f"delivery_model_sklearn_{carrier_service_id}.joblib"
        
        model_info = {'model': None, 'x_scaler': None, 'categorical_encoders': None}
        
        if dl_model_path.exists():
            model_data = torch.load(dl_model_path, map_location=_DEVICE, weights_only=False)
            hp = model_data.get('model_params', model_data)  # Support both old and new format
            
            model = TimeQuantileModel(
                numerical_dim=hp.get('numerical_dim', model_data.get('numerical_dim', len(cfg.DELIVERY_DL_NUMERICAL_FEATURES))),
                hidden_dim=hp.get('hidden_dim', model_data.get('hidden_dim', cfg.DELIVERY_DL_HIDDEN_DIM)),
                n_layers=hp.get('n_layers', model_data.get('n_layers', cfg.DELIVERY_DL_N_LAYERS)),
                dropout=hp.get('dropout', model_data.get('dropout', cfg.DELIVERY_DL_DROPOUT)),
                dropout_p=hp.get('dropout_p', model_data.get('dropout_p', cfg.DELIVERY_DL_DROPOUT_P)),
                dc_ori_vocab_size=model_data.get('vocab_sizes', {}).get('dc_ori') or hp.get('dc_ori_vocab_size'),
                dc_des_vocab_size=model_data.get('vocab_sizes', {}).get('dc_des') or hp.get('dc_des_vocab_size'),
                dc_ori_embedding_dim=hp.get('dc_ori_embedding_dim', model_data.get('dc_ori_embedding_dim', cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM)),
                dc_des_embedding_dim=hp.get('dc_des_embedding_dim', model_data.get('dc_des_embedding_dim', cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM))
            ).to(_DEVICE)
            
            # Support both 'model_state_dict' (new) and 'state_dict' (old) keys
            state_dict_key = 'model_state_dict' if 'model_state_dict' in model_data else 'state_dict'
            model.load_state_dict(model_data[state_dict_key])
            model.eval()
            
            model_info.update({
                'model': model, 
                'x_scaler': model_data.get('x_scaler'),
                'categorical_encoders': model_data.get('categorical_encoders')
            })
        elif rf_model_path.exists():
            model = joblib.load(rf_model_path)
            model_info['model'] = model
        elif xgb_model_path.exists():
            model = joblib.load(xgb_model_path)
            model_info['model'] = model
        elif sklearn_model_path.exists():
            model = joblib.load(sklearn_model_path)
            model_info['model'] = model
        else:
            if verbose:
                print(f"Warning: No per-cs delivery model found for carrier_service_id={carrier_service_id}. Falling back to global model.")
            # Fall through to global model loading below
            carrier_service_id = None
    
    # Fallback to global model (either no carrier_service_id provided, or per-cs model not found)
    if carrier_service_id is None:
        cache_key = f"global_delivery{mode_suffix}"
        if cache_key in _DELIVERY_MODEL_CACHE:
            return _DELIVERY_MODEL_CACHE[cache_key]

        dl_model_name = f"delivery_model_global{mode_suffix}.pt"
        qrf_model_name = f"delivery_model_global{mode_suffix}.joblib"
        
        dl_model_path = project_root / 'data' / 'models' / 'delivery_time' / dl_model_name
        qrf_model_path = project_root / 'data' / 'models' / 'delivery_time' / qrf_model_name

        model_info = {'model': None, 'x_scaler': None, 'categorical_encoders': None}

        if dl_model_path.exists():
            model_data = torch.load(dl_model_path, map_location=_DEVICE, weights_only=False)
            hp = model_data.get('model_params', model_data)  # Support both old and new format
            
            model = TimeQuantileModel(
                numerical_dim=hp.get('numerical_dim', model_data.get('numerical_dim', len(cfg.DELIVERY_DL_NUMERICAL_FEATURES))),
                hidden_dim=hp.get('hidden_dim', model_data.get('hidden_dim', cfg.DELIVERY_DL_HIDDEN_DIM)),
                n_layers=hp.get('n_layers', model_data.get('n_layers', cfg.DELIVERY_DL_N_LAYERS)),
                dropout=hp.get('dropout', model_data.get('dropout', cfg.DELIVERY_DL_DROPOUT)),
                dropout_p=hp.get('dropout_p', model_data.get('dropout_p', cfg.DELIVERY_DL_DROPOUT_P)),
                dc_ori_vocab_size=model_data.get('vocab_sizes', {}).get('dc_ori') or hp.get('dc_ori_vocab_size'),
                dc_des_vocab_size=model_data.get('vocab_sizes', {}).get('dc_des') or hp.get('dc_des_vocab_size'),
                dc_ori_embedding_dim=hp.get('dc_ori_embedding_dim', model_data.get('dc_ori_embedding_dim', cfg.DELIVERY_DL_DC_ORI_EMBEDDING_DIM)),
                dc_des_embedding_dim=hp.get('dc_des_embedding_dim', model_data.get('dc_des_embedding_dim', cfg.DELIVERY_DL_DC_DES_EMBEDDING_DIM))
            ).to(_DEVICE)
            
            # Support both 'model_state_dict' (new) and 'state_dict' (old) keys
            state_dict_key = 'model_state_dict' if 'model_state_dict' in model_data else 'state_dict'
            model.load_state_dict(model_data[state_dict_key])
            model.eval()
            
            model_info.update({
                'model': model, 
                'x_scaler': model_data.get('x_scaler'),
                'categorical_encoders': model_data.get('categorical_encoders')
            })

        elif qrf_model_path.exists():
            model = joblib.load(qrf_model_path)
            model_info['model'] = model
        
        else:
            if verbose:
                print(f"Warning: No global delivery time model found (tried {dl_model_name} and {qrf_model_name}).")
            return None

        _DELIVERY_MODEL_CACHE[cache_key] = model_info
        return model_info
    else:
        # Per-cs model was found, cache and return it
        _DELIVERY_MODEL_CACHE[cache_key] = model_info
        return model_info


def _generate_option_scenarios(order_items, options_df, customer_dc, promise_days, num_scenarios, 
                               seed=None, period='test', lookahead_periods=None, verbose=False,
                               demand_ml_mode: str = 'hybrid', hybrid_threshold: float | None = None,
                               dynamic_features: pd.DataFrame | None = None):
    """Generate demand + shipping cost scenarios for option-indexed fulfillment candidates.

    Falls back to empirical scenarios if ML models/features are unavailable.
    """
    project_root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(seed or cfg.RANDOM_SEED)
    order_id = order_items['order_ID'].iloc[0]
    order_date = pd.to_datetime(order_items['order_date'].iloc[0]).strftime('%Y-%m-%d')
    
    if verbose:
        print(f"--- Generating {num_scenarios} scenarios for order {order_id} ---")

    # Pre-allocate result dictionaries
    demand_scenarios = {}
    shipping_costs = {}
    unique_skus = [str(sku) for sku in order_items['sku_ID'].unique()]
    
    # 1. Demand Scenarios - Load model once (used for ML mode or hybrid-high SKUs)
    start_demand_gen = time.time()
    order_features = _load_features_from_eval_set(project_root, order_items, period, verbose=verbose)
    model_info = _get_demand_model(project_root, period=period, verbose=verbose)
    
    order_set = 'proxy_train' if period == 'proxy' else 'test'
    threshold = float(hybrid_threshold if hybrid_threshold is not None else getattr(cfg, 'DEMAND_HYBRID_THRESHOLD', 100.0))
    # Precompute mean totals for hybrid classification (if needed)
    if demand_ml_mode == 'hybrid':
        L_for_class = int(lookahead_periods if lookahead_periods is not None else cfg.DEMAND_FORECAST_HORIZON)
        mean_by_sku_map = _get_mean_total_by_sku(L_for_class, order_set)
    else:
        mean_by_sku_map = {}
    hist_upper_cache: dict[int, dict[int, float]] = {}

    # Cached empirical training data for per-SKU sampling
    cached_train = _get_cached_training_data(order_set, verbose=False)
    train_dem_samples = cached_train.get('demand_samples')
    train_sku_indices = cached_train.get('sku_indices')
    sku_to_idx_map = _get_sku_to_idx_map(project_root)

    ml_available = bool(model_info and order_features and model_info.get('model') is not None)
    model = model_info['model'] if ml_available else None
    if ml_available:
        model.eval()

    # Build per-SKU scenarios using selected mode
    for sku in unique_skus:
        use_ml = (demand_ml_mode == 'ml')
        sku_idx_int: int | None = None
        if demand_ml_mode == 'hybrid':
            # Determine sku_idx for classification
            if sku in order_features:
                sku_idx_tensor = order_features[sku][3]
                try:
                    sku_idx_int = int(sku_idx_tensor.item())
                except Exception:
                    sku_idx_int = int(sku_idx_tensor)
            elif sku_to_idx_map:
                sku_idx_int = int(sku_to_idx_map.get(str(sku), -1)) if str(sku) in sku_to_idx_map else None
            # Classify as high/slow
            if sku_idx_int is not None:
                mean_total = float(mean_by_sku_map.get(int(sku_idx_int), 0.0))
                use_ml = mean_total >= threshold
            else:
                use_ml = False  # missing mapping → treat as slow

        if use_ml and ml_available and sku in order_features:
            tensors = [t.to(_DEVICE) for t in order_features[sku]]
            with torch.no_grad():
                q_hat = model(*tensors).squeeze().cpu().numpy()
            actual_lookahead = int(min(lookahead_periods or q_hat.shape[0], q_hat.shape[0]))
            total_demand_quantiles = q_hat[:actual_lookahead].sum(axis=0)
            if num_scenarios == 1:
                q50_idx = int(np.argmin(np.abs(np.array(cfg.DEMAND_MODEL_QUANTILES) - 0.5)))
                val = int(round(float(total_demand_quantiles[q50_idx])))
                demand_scenarios[sku] = pd.Series([val], index=['scenario_0'])
            else:
                hist_upper_map = hist_upper_cache.get(actual_lookahead)
                if hist_upper_map is None:
                    hist_upper_map = _get_hist_upper_by_sku(actual_lookahead, order_set)
                    hist_upper_cache[actual_lookahead] = hist_upper_map
                if sku_idx_int is None:
                    sku_idx_tensor = order_features[sku][3]
                    try:
                        sku_idx_int = int(sku_idx_tensor.item())
                    except Exception:
                        sku_idx_int = int(sku_idx_tensor)
                hist_cap = float(hist_upper_map.get(int(sku_idx_int), 0.0))
                if total_demand_quantiles.shape[0] >= 2:
                    upper_extrap = float(total_demand_quantiles[-1] + (total_demand_quantiles[-1] - total_demand_quantiles[-2]))
                else:
                    upper_extrap = float(total_demand_quantiles[-1])
                upper_bound = max(upper_extrap, hist_cap)
                demand_scenarios[sku] = pd.Series(
                    sample_paths(total_demand_quantiles, num_scenarios, rng=rng, lower=0, upper=upper_bound).round().astype(int),
                    index=[f'scenario_{s}' for s in range(num_scenarios)]
                )
        else:
            # Empirical per-SKU sampling
            if train_dem_samples is None or train_sku_indices is None or len(train_dem_samples) == 0:
                draws = np.zeros(num_scenarios, dtype=int)
            else:
                # Map sku to sku_idx if needed
                if sku_idx_int is None:
                    if sku in order_features:
                        sku_idx_tensor = order_features[sku][3]
                        try:
                            sku_idx_int = int(sku_idx_tensor.item())
                        except Exception:
                            sku_idx_int = int(sku_idx_tensor)
                    elif sku_to_idx_map:
                        sku_idx_int = int(sku_to_idx_map.get(str(sku), -1)) if str(sku) in sku_to_idx_map else None
                if sku_idx_int is None or sku_idx_int == -1:
                    draws = np.zeros(num_scenarios, dtype=int)
                else:
                    mask = train_sku_indices == sku_idx_int
                    pool = train_dem_samples[mask]
                    if pool.size == 0:
                        draws = np.zeros(num_scenarios, dtype=int)
                    else:
                        L_eff = int(min(lookahead_periods if lookahead_periods is not None else pool.shape[1], pool.shape[1]))
                        totals = pool[:, :L_eff].sum(axis=1).astype(np.float32)
                        if num_scenarios == 1:
                            draws = np.array([int(max(0, round(float(np.median(totals)))))] , dtype=int)
                        else:
                            idx = rng.integers(0, len(totals), size=num_scenarios)
                            draws = np.rint(totals[idx]).astype(int)
            demand_scenarios[sku] = pd.Series(draws, index=[f'scenario_{s}' for s in range(num_scenarios)])

    ml_demand_ok = len(demand_scenarios) == len(unique_skus)

    if verbose:
        print(f"      - Demand Scenario Generation Time: {time.time() - start_demand_gen:.2f}s")

    # 2. Shipping Cost Scenarios - Process by carrier service (per-cs models)
    start_delivery_gen = time.time()
    option_ids = options_df['option_id'].to_numpy(dtype=int)
    dc_ids = options_df['dc_id'].to_numpy(dtype=int)
    carrier_ids = options_df['carrier_service_id'].to_numpy(dtype=int)
    base_costs = options_df['base_cost'].to_numpy(dtype=float)
    option_index = {int(opt): idx for idx, opt in enumerate(option_ids)}
    option_to_dc = {int(opt): int(dc_ids[idx]) for idx, opt in enumerate(option_ids)}
    cs_to_option_ids: dict[int, list[int]] = {}
    for carrier in np.unique(carrier_ids):
        mask = carrier_ids == carrier
        cs_to_option_ids[int(carrier)] = option_ids[mask].astype(int).tolist()
    dc_candidates = np.unique(dc_ids).astype(int).tolist()
    scenario_index = [f'scenario_{s}' for s in range(num_scenarios)]
    shipping_cost_matrix = np.full((num_scenarios, len(option_ids)), np.nan, dtype=float)

    # Load DC features once
    date_features_df = _DC_FEATURES_CACHE.get((order_date, order_set))
    if date_features_df is None:
        date_features_df = load_precomputed_order_dc_features(
            cfg.PRECOMPUTED_ORDER_DC_FEATURES_DIR,
            order_date,
            order_set,
        )
        _DC_FEATURES_CACHE[(order_date, order_set)] = date_features_df
    
    order_features_df = date_features_df[date_features_df['order_ID'] == order_id] if not date_features_df.empty else pd.DataFrame()
    if order_features_df.empty:
        raise RuntimeError(
            f"No precomputed order-DC features found for order {order_id}. "
            "Ensure build_sim_precompute has been run for this order set/date."
        )
    if 'candidate_dc' not in order_features_df.columns:
        raise RuntimeError("order_features_df missing 'candidate_dc' column.")
    order_features_df = order_features_df.drop(columns=cfg.DYNAMIC_FEATURES, errors='ignore')
    
    base_dynamic_cols = list(cfg.DYNAMIC_FEATURES)
    candidate_series = pd.to_numeric(order_features_df['candidate_dc'], errors='coerce')
    
    dyn_df: pd.DataFrame
    if dynamic_features is None or dynamic_features.empty:
        warnings.warn(
            "dynamic_features not provided; defaulting to zeros for all DCs.",
            RuntimeWarning,
        )
        dyn_df = pd.DataFrame({'dc_id': candidate_series.dropna().astype(int).unique()})
    else:
        dyn_df = dynamic_features.copy()
    
    if 'dc_id' not in dyn_df.columns:
        raise ValueError("dynamic_features DataFrame must include a 'dc_id' column.")
    
    dyn_df = dyn_df.copy()
    dyn_df['dc_id'] = pd.to_numeric(dyn_df['dc_id'], errors='coerce')
    dyn_df = dyn_df[dyn_df['dc_id'].notna()]
    dyn_df['dc_id'] = dyn_df['dc_id'].astype(int)
    
    for col in base_dynamic_cols:
        if col not in dyn_df.columns:
            dyn_df[col] = 0.0
    
    required_dc_ids = set(candidate_series.dropna().astype(int).unique().tolist())
    missing_dc_ids = required_dc_ids - set(dyn_df['dc_id'])
    if missing_dc_ids:
        dyn_df = pd.concat(
            [
                dyn_df,
                pd.DataFrame({'dc_id': list(missing_dc_ids)})
            ],
            ignore_index=True,
        )
        for col in base_dynamic_cols:
            dyn_df.loc[dyn_df['dc_id'].isin(missing_dc_ids), col] = 0.0
    
    dyn_map = dyn_df.set_index('dc_id')[base_dynamic_cols]
    order_features_df = order_features_df.copy()
    candidate_series_int = candidate_series.fillna(-1).astype(int)
    for col in base_dynamic_cols:
        mapped = candidate_series_int.map(dyn_map[col])
        order_features_df[col] = mapped.fillna(0.0).astype(float)
    
    ml_delivery_ok = False
    if not order_features_df.empty:
        # Filter to only candidate DCs that have features
        valid_dc_features = order_features_df[order_features_df['candidate_dc'].isin(dc_candidates)]
        valid_dcs = valid_dc_features['candidate_dc'].astype(int).values
        
        if len(valid_dcs) > 0:
            # Process each carrier service separately with its own model
            for carrier_service_id, option_ids_for_cs in cs_to_option_ids.items():
                # Get DCs for this carrier service
                mask = np.isin(option_ids, option_ids_for_cs, assume_unique=False)
                dcs_for_cs = np.unique(dc_ids[mask]).astype(int).tolist()
                valid_dcs_for_cs = [dc for dc in valid_dcs if dc in dcs_for_cs]
                
                if len(valid_dcs_for_cs) == 0:
                    continue
                
                # Load per-carrier-service model
                delivery_model_info = _get_delivery_model(
                    project_root, 
                    period=period, 
                    carrier_service_id=int(carrier_service_id),
                    verbose=verbose
                )
                
                if not delivery_model_info or not delivery_model_info.get('model'):
                    if verbose:
                        print(f"  Warning: No model available for carrier_service_id={carrier_service_id}, skipping options.")
                    continue
                
                model = delivery_model_info['model']
                is_dl_model = isinstance(model, TimeQuantileModel)
                
                # Filter features to DCs for this carrier service
                cs_dc_features = valid_dc_features[valid_dc_features['candidate_dc'].isin(valid_dcs_for_cs)]
                
                if len(cs_dc_features) == 0:
                    continue
                
                if is_dl_model:
                    # Batch process all DCs for this carrier service
                    model.eval()
                    X_numerical = cs_dc_features[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
                    X_categorical = cs_dc_features[cfg.DELIVERY_DL_CATEGORICAL_FEATURES]
                    
                    # Scale numerical features
                    x_scaler = delivery_model_info['x_scaler']
                    X_numerical_scaled = x_scaler.transform(X_numerical) if x_scaler else X_numerical.values
                    X_numerical_tensor = torch.tensor(X_numerical_scaled, dtype=torch.float32).to(_DEVICE)
                    
                    # Encode categorical features
                    encoders = delivery_model_info['categorical_encoders']
                    if encoders is None:
                        if verbose:
                            print(f"  Warning: No categorical encoders for carrier_service_id={carrier_service_id}, skipping.")
                        continue
                    dc_ori_encoded = _encode_with_unknown(
                        encoders['dc_ori'],
                        X_categorical['dc_ori'].astype(str).tolist(),
                    )
                    dc_ori_tensor = torch.tensor(dc_ori_encoded, dtype=torch.long).to(_DEVICE)
                    dest_values = [str(customer_dc)] * len(X_categorical)
                    dc_des_encoded = _encode_with_unknown(encoders['dc_des'], dest_values)
                    dc_des_tensor = torch.tensor(dc_des_encoded, dtype=torch.long).to(_DEVICE)
                    
                    # Single batch inference for all DCs in this carrier service
                    with torch.no_grad():
                        all_quantiles = model(X_numerical_tensor, dc_ori_tensor, dc_des_tensor).cpu().numpy()
                    
                    # Process results for each DC
                    d_q50_idx = int(np.argmin(np.abs(np.array(cfg.DELIVERY_TIME_QUANTILES) - 0.5)))
                    for quantile_idx, (_, row) in enumerate(cs_dc_features.iterrows()):
                        dc = int(row['candidate_dc'])
                        if dc not in valid_dcs_for_cs:
                            continue
                        predicted_quantiles = all_quantiles[quantile_idx]
                        if predicted_quantiles.ndim > 1:
                            predicted_quantiles = predicted_quantiles.squeeze()
                        if num_scenarios == 1:
                            delivery_times = np.array([float(predicted_quantiles[d_q50_idx])], dtype=float)
                        else:
                            delivery_times = sample_paths(predicted_quantiles, num_scenarios, rng=rng, lower=0, upper=5)
                        # Round ML delivery times to integer days
                        delivery_times = np.rint(delivery_times).astype(float)

                        # Calculate costs
                        delivery_deviations = delivery_times - promise_days
                        time_penalty = (
                            cfg.GAMMA_PLUS_LATE_PENALTY * np.maximum(0, delivery_deviations)
                            + cfg.GAMMA_MINUS_EARLY_PENALTY * np.maximum(0, -delivery_deviations)
                        )
                        time_penalty = np.atleast_1d(time_penalty).astype(float)

                        option_ids_for_dc = [opt_id for opt_id in option_ids_for_cs if option_to_dc.get(opt_id) == dc]
                        if not option_ids_for_dc:
                            continue
                        base_indices = [option_index[opt] for opt in option_ids_for_dc]
                        base_values = base_costs[base_indices]
                        costs_matrix = (time_penalty.reshape(-1, 1) + base_values.reshape(1, -1)).round(3)
                        shipping_cost_matrix[:, base_indices] = costs_matrix
                
                else:  # QRF or other quantile model
                    qs = [0.5] if num_scenarios == 1 else build_quantile_grid(num_scenarios, rng)
                    
                    # Batch predict for all DCs in this carrier service
                    X_batch = cs_dc_features[cfg.DELIVERY_TIME_FEATURES]
                    all_quantiles = model.predict(X_batch, quantiles=qs)
                    
                    # Process results for each DC
                    for quantile_idx, (_, row) in enumerate(cs_dc_features.iterrows()):
                        dc = int(row['candidate_dc'])
                        if dc not in valid_dcs_for_cs:
                            continue
                        predicted_quantiles = all_quantiles[quantile_idx]
                        if predicted_quantiles.ndim > 1:
                            predicted_quantiles = predicted_quantiles.squeeze()
                        if num_scenarios == 1:
                            delivery_times = np.array([float(np.squeeze(predicted_quantiles))], dtype=float)
                        else:
                            delivery_times = sample_paths(predicted_quantiles, num_scenarios, rng=rng, lower=0, upper=5)
                        # Round ML delivery times to integer days
                        delivery_times = np.rint(delivery_times).astype(float)

                        # Calculate costs
                        delivery_deviations = delivery_times - promise_days
                        time_penalty = (
                            cfg.GAMMA_PLUS_LATE_PENALTY * np.maximum(0, delivery_deviations)
                            + cfg.GAMMA_MINUS_EARLY_PENALTY * np.maximum(0, -delivery_deviations)
                        )
                        time_penalty = np.atleast_1d(time_penalty).astype(float)

                        option_ids_for_dc = [opt_id for opt_id in option_ids_for_cs if option_to_dc.get(opt_id) == dc]
                        if not option_ids_for_dc:
                            continue
                        base_indices = [option_index[opt] for opt in option_ids_for_dc]
                        base_values = base_costs[base_indices]
                        costs_matrix = (time_penalty.reshape(-1, 1) + base_values.reshape(1, -1)).round(3)
                        shipping_cost_matrix[:, base_indices] = costs_matrix
            
            ml_delivery_ok = True
    if verbose:
        print(f"      - Delivery Time Scenario Generation Time: {time.time() - start_delivery_gen:.2f}s")
        # for sku in unique_skus:
        #     print(f"      - Shipping Cost for SKU {sku}: {shipping_costs[sku]}")
        #     print(f"      - Demand for SKU {sku}: {demand_scenarios[sku]}")
        #     # Check structure
        #     print(f"Demand shape: {demand_scenarios[sku].shape}")
        #     print(f"Costs shape: {shipping_costs[sku].shape}")

        #     # Check first SKU quickly
        #     sku = list(demand_scenarios.keys())[0]
        #     print(f"\nSKU {sku}:")
        #     print(f"Demand: {demand_scenarios[sku].describe()}")
        #     print(f"Costs:\n{shipping_costs[sku].describe()}")

    if not ml_demand_ok:
        warnings.warn(
            f"Demand scenarios missing for order {order_id}; verify demand models and features.",
            RuntimeWarning,
        )
        raise RuntimeError("Failed to generate demand scenarios.")

    if not ml_delivery_ok:
        warnings.warn(
            f"Delivery scenarios missing for order {order_id}; using deterministic base costs.",
            RuntimeWarning,
        )
        fill_matrix = np.tile(base_costs, (num_scenarios, 1)) if num_scenarios > 0 else np.empty((0, len(option_ids)))
        shipping_cost_matrix[:, :] = fill_matrix

    nan_mask = np.isnan(shipping_cost_matrix).any(axis=0)
    nan_option_ids = option_ids[nan_mask]
    if nan_option_ids.size > 0:
        base_cost_lookup = {int(opt): float(base_costs[idx]) for idx, opt in enumerate(option_ids)}
        sample_preview = ", ".join(map(str, nan_option_ids[:5]))
        suffix = "..." if nan_option_ids.size > 5 else ""
        warnings.warn(
            f"Missing delivery estimates for {nan_option_ids.size} options "
            f"({sample_preview}{suffix}); using deterministic base costs.",
            RuntimeWarning,
        )
        for opt in nan_option_ids:
            shipping_cost_matrix[:, option_index[int(opt)]] = base_cost_lookup.get(int(opt), 0.0)

    shipping_cost_df = pd.DataFrame(
        shipping_cost_matrix,
        index=scenario_index,
        columns=option_ids.tolist(),
    )
    shipping_costs = {sku: shipping_cost_df for sku in unique_skus}

    return demand_scenarios, shipping_costs


def generate_scenarios(
    order_items,
    options_df=None,
    customer_dc=None,
    promise_days=None,
    num_scenarios=1,
    seed=None,
    period='test',
    lookahead_periods=None,
    verbose=False,
    demand_ml_mode: str = 'hybrid',
    hybrid_threshold: float | None = None,
    compatible_dcs: list | None = None,
    unit_costs_df: pd.DataFrame | None = None,
    dynamic_features: pd.DataFrame | None = None,
):
    """Public wrapper that supports both option-level and legacy DC-only inputs."""
    if options_df is None:
        if compatible_dcs is None or unit_costs_df is None or customer_dc is None or promise_days is None:
            raise ValueError(
                "generate_scenarios requires options_df or (compatible_dcs, unit_costs_df, customer_dc, promise_days)."
            )
        base_series = unit_costs_df.loc[compatible_dcs, str(customer_dc)].astype(float)
        options_df = pd.DataFrame({
            'option_id': np.arange(len(compatible_dcs), dtype=int),
            'dc_id': compatible_dcs,
            'carrier_service_id': 0,
            'base_cost': base_series.values,
            'distance_km': np.nan,
        })

    return _generate_option_scenarios(
        order_items=order_items,
        options_df=options_df.copy(),
        customer_dc=customer_dc,
        promise_days=promise_days,
        num_scenarios=num_scenarios,
        seed=seed,
        period=period,
        lookahead_periods=lookahead_periods,
        verbose=verbose,
        demand_ml_mode=demand_ml_mode,
        hybrid_threshold=hybrid_threshold,
        dynamic_features=dynamic_features,
    )


def generate_single_scenario(order_info: pd.Series,
                             order_items: pd.DataFrame,
                             options_df: pd.DataFrame | None,
                             period: str,
                             lookahead_periods: int,
                             verbose: bool = False,
                             demand_ml_mode: str = 'hybrid',
                             hybrid_threshold: float | None = None,
                             dynamic_features: pd.DataFrame | None = None) -> tuple[dict[str, int], dict[int, float]]:
    """Produce one deterministic (demand, cost) scenario concisely via generate_scenarios.

    Uses existing ML paths and their built-in fallbacks; then extracts the single scenario.
    """
    customer_dc = order_info['dc_des']
    unique_skus = [str(sku) for sku in order_items['sku_ID'].unique()]

    if options_df is None or options_df.empty:
        raise ValueError("generate_single_scenario requires options_df with distance-based costs.")
    
    dem_scen, cost_scen = generate_scenarios(
        order_items=order_items,
        options_df=options_df,
        customer_dc=customer_dc,
        promise_days=order_info['promise_delivery_days'],
        num_scenarios=1,
        seed=None,
        period=period,
        lookahead_periods=lookahead_periods,
        verbose=verbose,
        demand_ml_mode=demand_ml_mode,
        hybrid_threshold=hybrid_threshold,
        dynamic_features=dynamic_features,
    )

    # Demand: take the single value per SKU; default to 0 if missing
    demand_median = {str(sku): int(max(0, round(dem_scen.get(sku, pd.Series([0])).iloc[0]))) for sku in unique_skus}

    base_costs_by_option = options_df.set_index('option_id')['base_cost'].astype(float)
    if cost_scen:
        any_sku = next(iter(cost_scen))
        row = cost_scen[any_sku].reindex(columns=base_costs_by_option.index).iloc[0].fillna(base_costs_by_option)
    else:
        row = base_costs_by_option
    cost_median = {int(opt): float(row.loc[opt]) for opt in base_costs_by_option.index}

    return demand_median, cost_median
