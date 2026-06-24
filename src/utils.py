import numpy as np
from typing import Union, Optional
import torch
import src.config as cfg
import pandas as pd


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor):
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(taus * diff, (taus - 1) * diff).mean()



def sample_paths(
    q_hat: np.ndarray,          # [Q] sorted quantiles (e.g., 0.05..0.95)
    n_paths: int,
    rng: Union[np.random.Generator, None] = None,
    lower: Optional[float] = None, # τ=0 anchor (optional)
    upper: Optional[float] = None  # τ=1 anchor (optional)
) -> np.ndarray:                # → [n_paths] samples
    """
    Drop-in replacement that also samples the two tail intervals by adding τ=0/1 anchors.
    If lower/upper are not given, they are linearly extrapolated from end gaps.
    """
    if rng is None:
        rng = np.random.default_rng()

    q = np.asarray(q_hat, dtype=np.float32)
    if q.size < 2:
        raise ValueError("q_hat must have at least 2 quantiles.")

    # simple end extrapolation if bounds not provided
    if lower is None:
        lower = float(q[0] - (q[1] - q[0]))   # extend first gap
    if upper is None:
        upper = float(q[-1] + (q[-1] - q[-2]))  # extend last gap

    # keep monotone
    lower = min(lower, float(q[0]))
    upper = max(upper, float(q[-1]))

    q_ext = np.concatenate(([lower], q, [upper]))  # [Q+2]

    # uniformly pick intervals and interpolate within them (matches your original scheme)
    k = rng.integers(0, q_ext.size - 1, size=n_paths)
    u = rng.random(size=n_paths, dtype=np.float32)
    samples = q_ext[k] + u * (q_ext[k + 1] - q_ext[k])
    return samples.astype(np.float32)

# ──────────────────── Quantile Grid Builder ──────────────────────


def build_quantile_grid(n: int, rng: Union[np.random.Generator, None] = None) -> list[float]:
    """Return a sorted list of *n* quantiles in (0,1).

    Starts with the canonical 5%–95% grid (0.05,0.10,...,0.95) and, if more points
    are required, draws additional quantiles by uniformly interpolating between
    existing consecutive quantiles until the desired length is reached.
    """
    base = list(np.arange(0.05, 1, 0.05))
    if n <= len(base):
        if rng is None:
            rng = np.random.default_rng()
        return sorted(rng.choice(base, n, replace=False)) if n < len(base) else base

    if rng is None:
        rng = np.random.default_rng()

    qs = base.copy()
    num_to_generate = n - len(qs)

    # Iteratively insert samples into valid intervals [qs[k], qs[k+1]]
    for _ in range(num_to_generate):
        # valid interval indices are 0..len(qs)-2
        k = int(rng.integers(0, len(qs) - 1))
        u = float(rng.random())
        new_quantile = qs[k] + u * (qs[k + 1] - qs[k])
        # Insert and keep sorted order with minimal work
        qs.insert(k + 1, float(new_quantile))

    return qs[:n] 



def calculate_lookahead_periods(current_time: pd.Timestamp, simulation_date: str) -> int:
    """
    Calculate the number of time periods from current_time to the end of simulation_date.
    
    Args:
        current_time: Current order time
        simulation_date: Simulation date string (YYYY-MM-DD)
        time_freq: Time frequency from config (e.g., 'h' for hours, 'D' for days)
    
    Returns:
        Number of periods from current_time to end of simulation_date
    """
    time_freq = cfg.DEMAND_MODEL_TIME_PERIOD_FREQ
    
    simulation_end_time = pd.to_datetime(simulation_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    time_diff = simulation_end_time - current_time
    
    if time_freq == 'h':  # hours
        periods = int(time_diff.total_seconds() / 3600)
    elif time_freq == 'D':  # days
        periods = int(time_diff.total_seconds() / (24 * 3600))
    else:
        # Default to hours for unknown frequencies
        periods = int(time_diff.total_seconds() / 3600)
    
    return max(0, periods)


def load_proxy_model_and_data(model_path, order_set, simulation_date, device='cpu'):
    """
    Load proxy model checkpoint and prepare cached proxy inference data.

    Note: This lives in `src/utils.py` (module) rather than `src/utils/` (package) because
    this codebase also has a `src/utils.py` file, which prevents imports like
    `src.utils.proxy_loader` from resolving.
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # Local imports to avoid importing heavy deps at module import time
    from src.model.proxy_variants import build_proxy_model
    from src.algo.proxy import prepare_proxy_data

    model_path = Path(model_path) if not isinstance(model_path, Path) else model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Proxy model not found: {model_path}")

    logger.info(f"Loading proxy model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model parameters and determine architecture type
    model_params, architecture = _extract_proxy_model_params(checkpoint)

    logger.info(f"Loading proxy model architecture: {architecture}")
    model = build_proxy_model(model_params)

    # Load state dict with error handling
    state_dict = checkpoint['model']
    _load_proxy_state_dict_safe(model, state_dict)

    device_obj = torch.device(device)
    model.to(device_obj)

    # Load threshold from checkpoint (prioritize checkpoint over config)
    model_params_ckpt = checkpoint.get('model_params', {})
    hyperparams = checkpoint.get('hyperparams', {})
    threshold_ckpt = (
        model_params_ckpt.get('threshold') or
        hyperparams.get('loss', {}).get('threshold_on_sel')
    )
    
    if threshold_ckpt is not None:
        threshold = threshold_ckpt
        logger.info(f"[proxy] Using threshold={threshold} from checkpoint")
    else:
        threshold = cfg.PROXY_MODEL_THRESHOLD
        logger.info(f"[proxy] Using threshold={threshold} from config (checkpoint had no value)")

    feature_scalers = checkpoint.get('feature_scalers')
    if feature_scalers is None:
        global_scaler = checkpoint.get('scaler')
        if global_scaler is not None:
            feature_scalers = {'global': global_scaler}

    proxy_data = prepare_proxy_data(
        model=model,
        inference_params={'threshold': threshold, 'repair': True, 'debug': cfg.PROXY_INFERENCE_DEBUG},
        device=device_obj,
        model_info=checkpoint.get('info', {}),
        feature_scalers=feature_scalers,
        order_set=order_set,
        simulation_date=simulation_date,
    )

    logger.info(f"Proxy model loaded successfully for {order_set}/{simulation_date}")
    return proxy_data


def _extract_proxy_model_params(checkpoint: dict) -> tuple[dict, str]:
    """
    Extract and validate proxy model parameters from checkpoint.

    Returns:
        (model_params, architecture)
    """
    import logging
    logger = logging.getLogger(__name__)

    model_params = checkpoint.get('model_params', {})
    if not model_params:
        raise ValueError("Checkpoint missing 'model_params'. Cannot reconstruct model.")

    architecture = model_params.get('architecture', 'hierarchical_proxy_v2')
    if architecture not in {'hierarchical_proxy_v2', 'single_tower'}:
        raise ValueError(f"Unsupported architecture '{architecture}'.")

    if architecture == 'hierarchical_proxy_v2':
        valid_params = {
            'global_feature_dim', 'dc_feature_dim', 'option_feature_dim',
            'num_dcs', 'num_carriers',
            'sku_dim', 'brand_dim', 'sku_emb_dim', 'brand_emb_dim',
            'hidden_dim', 'n_layers', 'dropout_p', 'agg_type',
            'dc_embedding_dim', 'carrier_embedding_dim', 'option_proj_dim',
            'use_option_features_in_carrier', 'use_cost_summary',
            'use_scenario_module', 'use_dc_module',
            'use_dc_embedding', 'use_carrier_embedding', 'scenario_combine',
        }
    else:
        valid_params = {
            'global_feature_dim', 'dc_feature_dim', 'option_feature_dim',
            'num_dcs', 'num_carriers',
            'sku_dim', 'brand_dim', 'sku_emb_dim', 'brand_emb_dim',
            'hidden_dim', 'n_layers', 'dropout_p', 'agg_type', 'use_num_proj'
        }

    filtered_params = {k: v for k, v in model_params.items() if k in valid_params}

    model_info = checkpoint.get('info', {})
    if 'num_carriers' not in filtered_params:
        if 'num_carriers' in model_info:
            filtered_params['num_carriers'] = model_info['num_carriers']
            logger.info(f"[proxy] Using num_carriers={model_info['num_carriers']} from checkpoint info")
        else:
            raise ValueError("Checkpoint missing 'num_carriers' for proxy model.")

    return filtered_params, architecture


def _load_proxy_state_dict_safe(model, state_dict: dict):
    """Load proxy model weights, falling back to partial load on architecture mismatch."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.debug("[proxy] Model state dict loaded successfully (strict mode)")
    except RuntimeError as e:
        if 'Unexpected key(s)' in str(e) or 'size mismatch' in str(e):
            logger.warning(f"[proxy] Architecture mismatch detected, attempting partial load: {e}")
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {}
            model_state = model.state_dict()
            for k, v in state_dict.items():
                if k in model_keys:
                    model_param = model_state[k]
                    if model_param.shape == v.shape:
                        filtered_state_dict[k] = v
                    else:
                        logger.warning(
                            f"[proxy] Skipping {k}: shape mismatch (checkpoint: {getattr(v, 'shape', None)}, "
                            f"model: {getattr(model_param, 'shape', None)})"
                        )
                else:
                    logger.debug(f"[proxy] Skipping unexpected key: {k}")

            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"[proxy] Missing keys after partial load: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"[proxy] Unexpected keys after partial load: {unexpected_keys}")
        else:
            raise
