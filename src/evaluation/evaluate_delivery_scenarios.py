from __future__ import annotations

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt

import src.config as cfg

try:
    from scipy.stats import wasserstein_distance
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def plot_distribution(classes, true_dist, pred_dist, title, save_path):
    """
    Plot comparison between true and predicted PMF distributions.
    
    Args:
        classes: array of values (support)
        true_dist: true PMF values
        pred_dist: predicted/scenario PMF values
        title: plot title
        save_path: path to save the plot
    """
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(classes, true_dist, width, label='True Empirical Distribution', color='#4169E1', alpha=0.8)
    plt.bar(classes + width, pred_dist, width, label='Predicted Distribution', color='#B3A369', alpha=0.8)
    plt.xticks(classes + width/2, [str(int(c)) for c in classes], rotation=45, fontsize=12)
    plt.xlabel('Delivery Time (Days)')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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


def compute_interval_metrics(samples: np.ndarray, y_true: np.ndarray, alpha: float = 0.9) -> Tuple[float, float]:
    '''
    Coverage and sharpness for central alpha-interval.
    samples: [S, N], y_true: [N]
    Returns (coverage, sharpness)
    '''
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    lower = np.quantile(samples, lower_q, axis=0)
    upper = np.quantile(samples, upper_q, axis=0)
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = covered.mean()
    sharpness = np.mean(upper - lower)
    return float(coverage), float(sharpness)


def compute_wasserstein(samples: np.ndarray, y_true: np.ndarray) -> float | None:
    '''
    Wasserstein-1 between empirical sample distribution and degenerate at y_true, averaged over observations.
    If scipy is unavailable, return None.
    '''
    if not _HAVE_SCIPY:
        return None
    dists = []
    for i in range(y_true.shape[0]):
        d = wasserstein_distance(samples[:, i], np.array([y_true[i]]))
        dists.append(d)
    return float(np.mean(dists))


# ---------- Loaders ----------

def _dl_paths(project_root: Path):
    return project_root / 'data' / 'models' / 'delivery_time' / 'delivery_model_global_with_proxy.pt'

def _qrf_paths(project_root: Path):
    return project_root / 'data' / 'models' / 'delivery_time' / 'delivery_model_global_with_proxy.joblib'


def load_dl_bundle(project_root: Path):
    import torch
    from src.model.time_quantile_model import TimeQuantileModel

    path = _dl_paths(project_root)
    if not path.exists():
        return None, None, None
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    model = TimeQuantileModel(
        numerical_dim=bundle['numerical_dim'],
        hidden_dim=bundle['hidden_dim'],
        n_layers=bundle['n_layers'],
        dropout=True,
        dropout_p=bundle['dropout_p'],
        dc_ori_vocab_size=bundle['vocab_sizes'].get('dc_ori'),
        dc_des_vocab_size=bundle['vocab_sizes'].get('dc_des'),
        dc_ori_embedding_dim=bundle.get('dc_ori_embedding_dim', 8),
        dc_des_embedding_dim=bundle.get('dc_des_embedding_dim', 8),
    )
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    return model, bundle.get('x_scaler'), bundle.get('categorical_encoders')


def load_qrf_model(project_root: Path):
    import joblib
    path = _qrf_paths(project_root)
    if not path.exists():
        return None
    return joblib.load(path)


def load_simulator(model_name: str):
    import joblib
    import torch
    from src.model.time_quantile_model import TimeQuantileModel

    model_path = Path(cfg.DELIVERY_MODELS_DIR) / model_name
    if not model_path.exists():
        return None
    meta = joblib.load(model_path)
    # DL simulator pointer
    if isinstance(meta, dict) and meta.get('type') == 'dl':
        bundle_path = Path(meta['dl_model_path'])
        if not bundle_path.exists():
            return None
        bundle = torch.load(bundle_path, map_location='cpu', weights_only=False)
        model = TimeQuantileModel(
            numerical_dim=bundle['numerical_dim'],
            hidden_dim=bundle['hidden_dim'],
            n_layers=bundle['n_layers'],
            dropout=True,
            dropout_p=bundle['dropout_p'],
            dc_ori_vocab_size=bundle['vocab_sizes'].get('dc_ori'),
            dc_des_vocab_size=bundle['vocab_sizes'].get('dc_des'),
            dc_ori_embedding_dim=bundle.get('dc_ori_embedding_dim', 8),
            dc_des_embedding_dim=bundle.get('dc_des_embedding_dim', 8),
        )
        model.load_state_dict(bundle['state_dict'])
        model.eval()
        return {
            'type': 'dl',
            'model': model,
            'x_scaler': bundle.get('x_scaler'),
            'encoders': bundle.get('categorical_encoders'),
        }
    # CatBoost simulator bundle
    return {
        'type': 'catboost',
        'model': meta['model'],
        'label_encoder': meta['label_encoder'],
    }


# ---------- Samplers ----------

def sample_dl(N: int, df: pd.DataFrame, model, x_scaler, encoders) -> np.ndarray:
    import torch
    from src.utils import sample_paths

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_num = df[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
    X_num_scaled = x_scaler.transform(X_num) if x_scaler is not None else X_num.values
    X_num_tensor = np.asarray(X_num_scaled, dtype=np.float32)

    dc_ori_enc = encoders['dc_ori'].transform(df['dc_ori'].astype(str))
    dc_des_enc = encoders['dc_des'].transform(df['dc_des'].astype(str))

    preds = np.zeros((len(df), len(cfg.DELIVERY_TIME_QUANTILES)), dtype=np.float32)
    bs = 2048
    with torch.no_grad():
        for s in range(0, len(df), bs):
            e = min(len(df), s + bs)
            Xb = torch.tensor(X_num_tensor[s:e], dtype=torch.float32, device=device)
            dc_o = torch.tensor(dc_ori_enc[s:e], dtype=torch.long, device=device)
            dc_d = torch.tensor(dc_des_enc[s:e], dtype=torch.long, device=device)
            preds[s:e] = model(Xb, dc_o, dc_d).cpu().numpy()

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    out = np.zeros((N, len(df)), dtype=np.float32)
    for i in range(len(df)):
        out[:, i] = sample_paths(preds[i], N, rng=rng, lower=0, upper=5)
    return out


def sample_qrf(N: int, df: pd.DataFrame, model) -> np.ndarray:
    from src.utils import build_quantile_grid
    qs = build_quantile_grid(N, np.random.default_rng(cfg.RANDOM_SEED))
    X = df[cfg.DELIVERY_TIME_FEATURES]
    preds = model.predict(X, quantiles=qs)  # [Nobs, N]
    return preds.T.astype(np.float32)


def sample_empirical(N: int, df: pd.DataFrame, order_set: str = 'test') -> np.ndarray:
    from src.empirical_scenarios import _get_cached_training_data
    cached = _get_cached_training_data('test' if order_set == 'test' else 'proxy_train', verbose=False)
    pool = cached['delivery_times']
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    samples = rng.choice(pool, size=(N, len(df)), replace=True)
    return samples.astype(np.float32)


def sample_dl_per_carrier(
    N: int,
    df: pd.DataFrame,
    period: str = 'test',
    cs_models_dir: str | Path | None = None,
) -> np.ndarray:
    """
    Sample per-carrier DL delivery scenarios using the same loaders as scenario generation.
    Returns shape [S, Nobs].
    """
    import torch
    from src.scenario_generator import _get_delivery_model, _encode_with_unknown
    from src.model.time_quantile_model import TimeQuantileModel
    from src.utils import sample_paths, build_quantile_grid
    from src.training.delivery_time.common import format_carrier_id_for_path

    project_root = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = df.copy().reset_index(drop=True)
    if 'carrier_service_id' not in df.columns:
        if 'carrier_service_id_anon' in df.columns:
            df['carrier_service_id'] = df['carrier_service_id_anon']
        else:
            raise ValueError("carrier_service_id column missing; cannot sample per-carrier DL models.")

    samples = np.zeros((N, len(df)), dtype=np.float32)
    # Build candidate CS model directories (only for this evaluator; scenario_generator is untouched).
    if cs_models_dir is not None:
        base_dir = Path(cs_models_dir)
        if not base_dir.is_absolute():
            base_dir = project_root / base_dir
    else:
        base_dir = Path(cfg.DELIVERY_MODELS_CS_DIR)
        if not base_dir.is_absolute():
            base_dir = project_root / base_dir
    cs_dirs = [base_dir]
    if base_dir.name == "tune":
        cs_dirs.append(base_dir.parent)
    else:
        cs_dirs.append(base_dir / "tune")
    cs_dirs = [p for i, p in enumerate(cs_dirs) if p not in cs_dirs[:i]]

    for carrier_id, idx in df.groupby('carrier_service_id').groups.items():
        try:
            carrier_id_int = int(float(carrier_id))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid carrier_service_id value: {carrier_id!r}")
        idx = list(idx)
        df_c = df.loc[idx]
        model_info = None
        last_exc = None
        orig_dir = cfg.DELIVERY_MODELS_CS_DIR
        try:
            for cs_dir in cs_dirs:
                # Ensure cs_dir is absolute and convert to Path for consistency
                cs_dir_path = Path(cs_dir)
                if not cs_dir_path.is_absolute():
                    cs_dir_path = project_root / cs_dir_path
                
                # Set as relative to project_root if possible, otherwise use absolute
                try:
                    rel_path = cs_dir_path.relative_to(project_root)
                    cfg.DELIVERY_MODELS_CS_DIR = rel_path
                except ValueError:
                    # Not relative to project_root, use absolute
                    cfg.DELIVERY_MODELS_CS_DIR = cs_dir_path
                
                try:
                    import src.scenario_generator as sg
                    sg._DELIVERY_MODEL_CACHE.clear()
                except Exception:
                    pass
                try:
                    model_info = _get_delivery_model(
                        project_root,
                        period=period,
                        carrier_service_id=carrier_id_int,
                        verbose=False,
                    )
                except Exception as exc:
                    model_info = None
                    last_exc = exc
                if model_info and model_info.get('model'):
                    break
        finally:
            cfg.DELIVERY_MODELS_CS_DIR = orig_dir

        if not model_info or not model_info.get('model'):
            carrier_id_str = format_carrier_id_for_path(carrier_id_int)
            candidates = []
            for base in cs_dirs:
                candidates.append(base / f"dl_model_{carrier_id_str}_with_proxy.pt")
                candidates.append(base / f"dl_model_{carrier_id_str}.pt")
            existing = [str(p) for p in candidates if p.exists()]
            msg = (
                f"No delivery model found for carrier_service_id={carrier_id_int}. "
                f"Checked DELIVERY_MODELS_CS_DIR={cs_dirs[0]} and fallback={cs_dirs[1] if len(cs_dirs)>1 else None}. "
                f"Existing matches: {existing if existing else 'none'}."
            )
            if last_exc is not None:
                msg += f" Last exception: {last_exc}"
            raise ValueError(msg)

        model = model_info['model']
        if isinstance(model, TimeQuantileModel):
            model = model.to(device)
            X_num = df_c[cfg.DELIVERY_DL_NUMERICAL_FEATURES]
            X_num_scaled = model_info['x_scaler'].transform(X_num) if model_info.get('x_scaler') else X_num.values
            X_num_tensor = np.asarray(X_num_scaled, dtype=np.float32)

            encoders = model_info.get('categorical_encoders')
            if encoders is None:
                raise ValueError(f"Missing categorical encoders for carrier_service_id={carrier_id}.")
            dc_ori_enc = _encode_with_unknown(encoders['dc_ori'], df_c['dc_ori'].astype(str).tolist())
            dc_des_enc = _encode_with_unknown(encoders['dc_des'], df_c['dc_des'].astype(str).tolist())

            preds = np.zeros((len(df_c), len(cfg.DELIVERY_TIME_QUANTILES)), dtype=np.float32)
            bs = 2048
            with torch.no_grad():
                for s in range(0, len(df_c), bs):
                    e = min(len(df_c), s + bs)
                    Xb = torch.tensor(X_num_tensor[s:e], dtype=torch.float32, device=device)
                    dc_o = torch.tensor(dc_ori_enc[s:e], dtype=torch.long, device=device)
                    dc_d = torch.tensor(dc_des_enc[s:e], dtype=torch.long, device=device)
                    preds[s:e] = model(Xb, dc_o, dc_d).cpu().numpy()

            for j, row_idx in enumerate(idx):
                samples[:, row_idx] = sample_paths(preds[j], N, rng=rng, lower=0, upper=5)
        else:
            # QRF or other quantile model with .predict(quantiles=...)
            qs = [0.5] if N == 1 else build_quantile_grid(N, rng)
            X = df_c[cfg.DELIVERY_TIME_FEATURES]
            preds = model.predict(X, quantiles=qs)  # [Nobs, N]
            samples[:, idx] = preds.T.astype(np.float32)

    return samples.astype(np.float32)


def maybe_round_days(samples: np.ndarray, round_days: bool) -> np.ndarray:
    if not round_days:
        return samples
    return np.rint(samples).astype(np.float32)


# ---------- By-carrier evaluation ----------

def _load_cs_test_df() -> pd.DataFrame:
    """
    Load the carrier-service-level evaluation dataframe with engineered features.

    We intentionally reuse the delivery-time training data loader so that:
    - `carrier_service_id_anon` exists (delivery_time_test.csv does not include it)
    - all dynamic features in cfg.DELIVERY_TIME_FEATURES are present
    """
    from src.training.delivery_time.common import load_split_data

    _, df_eval = load_split_data(train_on_proxy=True, use_cs_data=True)
    return df_eval


def _get_carrier_col(df: pd.DataFrame, carrier_col: str | None) -> str:
    if carrier_col is not None:
        if carrier_col not in df.columns:
            raise ValueError(f"--carrier_col='{carrier_col}' not found in dataframe columns.")
        return carrier_col
    if 'carrier_service_id' in df.columns:
        return 'carrier_service_id'
    if 'carrier_service_id_anon' in df.columns:
        return 'carrier_service_id_anon'
    raise ValueError("Could not infer carrier column. Expected 'carrier_service_id' or 'carrier_service_id_anon'.")


def evaluate_simulator_by_carrier(
    N: int,
    alpha: float = 0.9,
    head: int | None = None,
    round_days: bool = False,
    exceed_thresholds: list[int] | None = None,
    quantiles: list[float] | None = None,
    reliability_bins: int = 10,
    late_threshold: float = 0.0,
    late_weight: float = 1.0,
    simulator_type: str = 'dl',
    carrier_col: str | None = None,
    min_rows_per_carrier: int = 0,
    plots_dir: Path | None = None,
    plot_pmf: bool = False,
) -> Tuple[Dict[int, Dict[str, object]], pd.DataFrame]:
    """
    Evaluate the *simulator* distribution vs true empirical test outcomes, per carrier.

    This uses the same per-carrier simulator artifacts used by the simulator
    (via OutcomeSampler), so results reflect what run_simulation will sample.
    """
    from src.simulator.delivery_sampler import OutcomeSampler

    df = _load_cs_test_df()
    df.columns = [c.strip() for c in df.columns]

    y = df[cfg.DELIVERY_TIME_TARGET].values
    mask = ~np.isnan(y)
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]

    if head is not None and head > 0:
        df = df.head(head)
        y = y[: len(df)]

    if exceed_thresholds is None:
        exceed_thresholds = [0, 1, 2, 3]
    if quantiles is None:
        quantiles = [0.8, 0.9, 0.95]

    ccol = _get_carrier_col(df, carrier_col)

    # OutcomeSampler expects `carrier_service_id` for internal grouping / loading.
    # We keep the original column name for grouping and create an alias.
    if 'carrier_service_id' not in df.columns and ccol in df.columns:
        df = df.copy()
        df['carrier_service_id'] = df[ccol]

    carriers = sorted([c for c in df[ccol].dropna().unique().tolist()])

    sampler = OutcomeSampler(simulator_type=simulator_type, scenario_source='simulator')

    results: Dict[int, Dict[str, object]] = {}
    rows = []

    # Aggregate "late" stats across all carriers. These mirror the simulation definition:
    # delivered_days > promise_delivery_days.
    total_units = 0.0
    total_expected_late_units = 0.0
    total_expected_late_rows = 0.0
    total_rows = 0
    order_units: dict[str, float] = {}
    order_expected_late_units: dict[str, float] = {}

    if plots_dir is not None and plot_pmf:
        plots_dir.mkdir(parents=True, exist_ok=True)

    for carrier_id in carriers:
        # Robustly coerce to int-like id for filenames / caching.
        try:
            cid_int = int(float(carrier_id))
        except Exception:
            # Skip non-numeric carriers; these won't match simulator artifacts anyway.
            continue

        df_c = df[df[ccol] == carrier_id].copy()
        if min_rows_per_carrier > 0 and len(df_c) < min_rows_per_carrier:
            continue

        y_c = df_c[cfg.DELIVERY_TIME_TARGET].values.astype(np.float32)

        # Prefer carrier-specific simulator, otherwise fall back to global.
        model_info = sampler._load_cs_simulator(cid_int)
        used_fallback = False
        if model_info is None:
            model_info = sampler._load_global_model()
            used_fallback = True
        if model_info is None:
            continue

        rng = np.random.default_rng(cfg.RANDOM_SEED + cid_int)

        if model_info['type'] == 'dl':
            # Match simulation sampling: per-carrier DL simulator + unknown-safe encoders + rounded day labels.
            samples_df = sampler._sample_dl(
                df_c,
                N,
                rng,
                model_info['model'],
                model_info.get('x_scaler'),
                model_info.get('encoders'),
            )  # [Nobs, S]
            sim_samples = samples_df.to_numpy().T.astype(np.float32)  # [S, Nobs]
        elif model_info['type'] == 'catboost':
            samples_df = sampler._sample_catboost(
                df_c[cfg.DELIVERY_TIME_FEATURES],
                N,
                rng,
                model_info['model'],
                model_info['label_encoder'],
            )  # [Nobs, S]
            sim_samples = samples_df.to_numpy().T.astype(np.float32)  # [S, Nobs]
        else:
            # QRF simulators are not expected here, but keep the branch safe.
            continue

        sim_samples = maybe_round_days(sim_samples, round_days)

        # Late stats (expected late probability per row, then aggregate).
        if 'promise_delivery_days' in df_c.columns:
            prom = df_c['promise_delivery_days'].to_numpy(dtype=np.float32, copy=False)
            p_late = (sim_samples > prom[None, :]).mean(axis=0)  # [Nobs]
            qty = df_c['quantity'].to_numpy(dtype=np.float32, copy=False) if 'quantity' in df_c.columns else None
            if qty is not None:
                total_units += float(qty.sum())
                total_expected_late_units += float((qty * p_late).sum())
            total_expected_late_rows += float(p_late.sum())
            total_rows += int(p_late.size)

            if 'order_ID' in df_c.columns and qty is not None:
                tmp = pd.DataFrame(
                    {
                        'order_ID': df_c['order_ID'].astype(str).to_numpy(),
                        'units': qty,
                        'late_units_exp': qty * p_late,
                    }
                )
                grp = tmp.groupby('order_ID', sort=False).sum(numeric_only=True)
                for oid, r in grp.iterrows():
                    order_units[oid] = order_units.get(oid, 0.0) + float(r['units'])
                    order_expected_late_units[oid] = order_expected_late_units.get(oid, 0.0) + float(r['late_units_exp'])

        cov, sharp = compute_interval_metrics(sim_samples, y_c, alpha)

        metrics = {
            'mean_crps': float(crps(y_c, sim_samples)),
            'coverage': cov,
            'sharpness': sharp,
            'wasserstein': compute_wasserstein(sim_samples, y_c),
            'decision_proxy': decision_proxy_loss(sim_samples, y_c, late_threshold, late_weight),
            'exceedance': exceedance_calibration(sim_samples, y_c, exceed_thresholds, reliability_bins),
            'quantiles': high_quantile_metrics(sim_samples, y_c, quantiles),
            'pmf': discrete_support_metrics(sim_samples, y_c),
            'num_obs': int(len(y_c)),
            'num_samples_per_obs': int(N),
            'used_global_fallback': bool(used_fallback),
        }
        results[cid_int] = metrics

        rows.append({
            'carrier_service_id': cid_int,
            'num_obs': int(len(y_c)),
            'mean_crps': metrics['mean_crps'],
            'pmf_l1': metrics['pmf']['l1'],
            'wasserstein': metrics['wasserstein'],
            'coverage': metrics['coverage'],
            'sharpness': metrics['sharpness'],
            'used_global_fallback': bool(used_fallback),
        })

        if plots_dir is not None and plot_pmf:
            pmf_data = metrics['pmf']
            support = np.array(pmf_data['support'])
            true_pmf = np.array(pmf_data['true_pmf'])
            scn_pmf = np.array(pmf_data['scn_pmf'])
            plot_path = plots_dir / f'pmf_distribution_simulator_{simulator_type}_carrier_{cid_int}.png'
            plot_distribution(
                classes=support,
                true_dist=true_pmf,
                pred_dist=scn_pmf,
                title=f'PMF Distribution: SIMULATOR({simulator_type}) vs True - Carrier {cid_int}',
                save_path=plot_path,
            )

    summary_df = pd.DataFrame(rows).sort_values(['carrier_service_id']).reset_index(drop=True)

    # Attach global late stats as a lightweight footer-like dict on the return value
    # (main prints it; callers can also read it).
    late_units_pct = (100.0 * total_expected_late_units / total_units) if total_units > 0 else float('nan')
    late_rows_pct = (100.0 * total_expected_late_rows / total_rows) if total_rows > 0 else float('nan')
    late_orders_pct = float('nan')
    if order_units:
        per_order = [
            100.0 * (order_expected_late_units.get(oid, 0.0) / units)
            for oid, units in order_units.items()
            if units > 0
        ]
        late_orders_pct = float(np.mean(per_order)) if per_order else float('nan')

    results['_global'] = {
        'late_delivery_pct_units_expected': late_units_pct,
        'late_delivery_pct_rows_expected': late_rows_pct,
        'late_delivery_pct_orders_expected': late_orders_pct,
        'num_orders': int(len(order_units)) if order_units else None,
        'num_rows': int(total_rows),
        'total_units': float(total_units) if total_units > 0 else None,
    }
    return results, summary_df


# ---------- Simulator helpers ----------

def simulator_probs(df: pd.DataFrame, model, label_encoder) -> Tuple[np.ndarray, np.ndarray]:
    feats = [f for f in cfg.DELIVERY_TIME_FEATURES if f in df.columns]
    X = df[feats].copy()
    # Fill any missing required features with 0
    for f in cfg.DELIVERY_TIME_FEATURES:
        if f not in X.columns:
            X[f] = 0
    probs = model.predict_proba(X[cfg.DELIVERY_TIME_FEATURES])  # [Nobs, K]
    labels = label_encoder.inverse_transform(np.arange(probs.shape[1]))
    labels = np.asarray(labels, dtype=float)
    # Ensure labels ascending and reorder probs
    order = np.argsort(labels)
    return probs[:, order], labels[order]


def simulator_samples(N: int, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    cum = np.cumsum(probs, axis=1)  # [Nobs, K]
    rand = rng.random((N, probs.shape[0]))  # [S, Nobs]
    cmp = cum[None, :, :] > rand[:, :, None]  # [S, Nobs, K]
    idx = cmp.argmax(axis=2)  # [S, Nobs]
    return labels[idx].T.astype(np.float32)


def simulator_exceedance_probs(probs: np.ndarray, labels: np.ndarray, thresholds: list[int]) -> Dict[int, np.ndarray]:
    out = {}
    for t in thresholds:
        mask = (labels > float(t)).astype(float)
        out[int(t)] = probs @ mask  # [Nobs]
    return out


def simulator_quantile(labels: np.ndarray, probs: np.ndarray, q: float) -> np.ndarray:
    cdf = np.cumsum(probs, axis=1)
    idx = (cdf >= q).argmax(axis=1)
    return labels[idx]


def pmf_from_samples(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s_int = np.rint(samples).astype(int)
    vals, counts = np.unique(s_int.ravel(), return_counts=True)
    pmf = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    return vals.astype(int), pmf


# ---------- Tail diagnostics ----------

def exceedance_calibration(samples: np.ndarray, y: np.ndarray, thresholds: list[int], n_bins: int = 10) -> dict:
    out = {}
    for t in thresholds:
        p_hat = (samples > t).mean(axis=0)
        o = (y > t).astype(float)
        brier = float(np.mean((p_hat - o) ** 2))
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(p_hat, bins) - 1
        p_mean = np.zeros(n_bins)
        o_mean = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            m = idx == b
            if np.any(m):
                p_mean[b] = float(np.mean(p_hat[m]))
                o_mean[b] = float(np.mean(o[m]))
                counts[b] = int(m.sum())
        out[int(t)] = {
            'brier': brier,
            'bin_centers': ((bins[:-1] + bins[1:]) / 2).tolist(),
            'p_mean': p_mean.tolist(),
            'o_mean': o_mean.tolist(),
            'counts': counts.tolist(),
        }
    return out


def high_quantile_metrics(samples: np.ndarray, y: np.ndarray, qs: list[float]) -> dict:
    out = {}
    for q in qs:
        qhat = np.quantile(samples, q, axis=0)
        out[q] = {
            'bias': float(np.mean(qhat - y)),
            'coverage': float(np.mean(y <= qhat)),
            'mae': float(np.mean(np.abs(y - qhat))),
        }
    return out


def discrete_support_metrics(samples: np.ndarray, y: np.ndarray) -> dict:
    y_int = np.rint(y).astype(int)
    s_int = np.rint(samples).astype(int)
    uy, cy = np.unique(y_int, return_counts=True)
    us, cs = np.unique(s_int.ravel(), return_counts=True)
    support = np.union1d(uy, us)
    mapy = {int(v): int(c) for v, c in zip(uy, cy)}
    maps = {int(v): int(c) for v, c in zip(us, cs)}
    true_counts = np.array([mapy.get(int(v), 0) for v in support], dtype=int)
    scn_counts = np.array([maps.get(int(v), 0) for v in support], dtype=int)
    true_p = true_counts / true_counts.sum() if true_counts.sum() > 0 else np.zeros_like(true_counts, dtype=float)
    scn_p = scn_counts / scn_counts.sum() if scn_counts.sum() > 0 else np.zeros_like(scn_counts, dtype=float)
    l1 = float(np.sum(np.abs(true_p - scn_p)))
    try:
        from scipy.special import rel_entr
        kl = float(np.sum(rel_entr(true_p + 1e-12, scn_p + 1e-12)))
    except Exception:
        kl = float('nan')
    return {
        'support': support.astype(int).tolist(),
        'true_counts': true_counts.tolist(),
        'scn_counts': scn_counts.tolist(),
        'true_pmf': true_p.tolist(),
        'scn_pmf': scn_p.tolist(),
        'l1': l1,
        'kl': kl,
    }


def decision_proxy_loss(samples: np.ndarray, y: np.ndarray, t_late: float = 0.0, lam: float = 1.0) -> float:
    lam = max(lam, 1.0)
    med = np.median(samples, axis=0)
    loss = np.abs(y - med) + lam * np.maximum(y - t_late, 0.0)
    return float(np.mean(loss))


# ---------- Evaluation ----------

def evaluate(
    N: int,
    alpha: float = 0.9,
    head: int | None = None,
    round_days: bool = False,
    exceed_thresholds: list[int] | None = None,
    quantiles: list[float] | None = None,
    reliability_bins: int = 10,
    late_threshold: float = 0.0,
    late_weight: float = 1.0,
    compare_to_simulator: bool = False,
    simulator_type: str = 'dl',
    eval_simulator: bool = False,
    cs_models_dir: str | Path | None = None,
    simulator_dir: str | Path | None = None,
) -> Dict[str, Dict[str, object]]:
    project_root = Path(__file__).resolve().parents[2]
    # Always use CS test data so per-carrier DL models can be evaluated.
    df = _load_cs_test_df()
    df.columns = [c.strip() for c in df.columns]
    y = df[cfg.DELIVERY_TIME_TARGET].values
    mask = ~np.isnan(y)
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]
    if head is not None and head > 0:
        df = df.head(head)
        y = y[:len(df)]

    if exceed_thresholds is None:
        exceed_thresholds = [0, 1, 2, 3]
    if quantiles is None:
        quantiles = [0.8, 0.9, 0.95]

    results: Dict[str, Dict[str, object]] = {}

    # Optional simulator reference (supports CatBoost or DL simulator)
    sim_ref = None
    if compare_to_simulator or eval_simulator:
        if simulator_type != 'dl':
            raise ValueError("Only simulator_type='dl' is supported for OutcomeSampler in this evaluation.")
        from src.simulator.delivery_sampler import OutcomeSampler

        if simulator_dir is not None:
            sim_dir = Path(simulator_dir)
            if not sim_dir.is_absolute():
                sim_dir = project_root / sim_dir
            cfg.DELIVERY_TIME_SIMULATOR_PATH = sim_dir
        sampler = OutcomeSampler(simulator_type='dl', scenario_source='simulator')
        df_sim = df.copy()
        if 'carrier_service_id' not in df_sim.columns:
            if 'carrier_service_id_anon' in df_sim.columns:
                df_sim['carrier_service_id'] = df_sim['carrier_service_id_anon']
            else:
                raise ValueError("carrier_service_id column missing; cannot use OutcomeSampler.")
        sim_samples_df = sampler.sample(df_sim, num_replications=N)  # [Nobs, S]
        sim_ref = sim_samples_df.to_numpy().astype(np.float32)  # [Nobs, S]

    def add_vs_sim(metrics: dict, samples: np.ndarray):
        if sim_ref is None:
            return metrics
        sim_samples = sim_ref  # [Nobs, S]
        # Wasserstein between sample sets per observation
        if _HAVE_SCIPY:
            dists = [wasserstein_distance(samples[:, i], sim_samples[i, :]) for i in range(samples.shape[1])]
            w1 = float(np.mean(dists))
        else:
            w1 = None
        # Exceedance Brier vs simulator (sample-based reference)
        ex_brier = {}
        for t in exceed_thresholds:
            p_hat = (samples > t).mean(axis=0)  # [Nobs]
            p_sim = (sim_samples > t).mean(axis=1)  # [Nobs]
            ex_brier[int(t)] = float(np.mean((p_hat - p_sim) ** 2))
        # High-quantile bias/coverage vs simulator (sample-based)
        qstats = {}
        for q in quantiles:
            qhat = np.quantile(samples, q, axis=0)  # [Nobs]
            qtrue = np.quantile(sim_samples, q, axis=1)  # [Nobs]
            cov_vals = [float(np.mean(sim_samples[i, :] <= qhat[i])) for i in range(len(qhat))]
            bias = float(np.mean(qhat - qtrue))
            cov = float(np.mean(np.array(cov_vals)))
            mae = float(np.mean(np.abs(qhat - qtrue)))
            qstats[q] = {'bias_vs_sim': bias, 'coverage_vs_sim': cov, 'mae_vs_sim': mae}
        # PMF L1/KL vs simulator pooled pmf
        vals_ml, pmf_ml = pmf_from_samples(samples)
        vals_sim, pmf_sim = pmf_from_samples(sim_samples)
        support = np.union1d(vals_ml, vals_sim)
        map_ml = {int(v): p for v, p in zip(vals_ml, pmf_ml)}
        map_sim = {int(v): p for v, p in zip(vals_sim, pmf_sim)}
        p_ml = np.array([map_ml.get(int(v), 0.0) for v in support])
        p_sim = np.array([map_sim.get(int(v), 0.0) for v in support])
        l1 = float(np.sum(np.abs(p_ml - p_sim)))
        try:
            from scipy.special import rel_entr
            kl = float(np.sum(rel_entr(p_sim + 1e-12, p_ml + 1e-12)))
        except Exception:
            kl = float('nan')
        metrics['vs_sim'] = {
            'wasserstein': w1,
            'exceedance_brier': {int(k): float(v) for k, v in ex_brier.items()},
            'high_quantiles': qstats,
            'pmf_l1': l1,
            'pmf_kl': kl,
        }
        return metrics

    # Simulator itself (vs true empirical distribution).
    if eval_simulator and sim_ref is not None:
        sim_samples = maybe_round_days(sim_ref.T, round_days)  # [S, Nobs]
        cov, sharp = compute_interval_metrics(sim_samples, y, alpha)
        results[f"simulator_{simulator_type}"] = {
            'mean_crps': float(crps(y, sim_samples)),
            'coverage': cov,
            'sharpness': sharp,
            'wasserstein': compute_wasserstein(sim_samples, y),
            'decision_proxy': decision_proxy_loss(sim_samples, y, late_threshold, late_weight),
            'exceedance': exceedance_calibration(sim_samples, y, exceed_thresholds, reliability_bins),
            'quantiles': high_quantile_metrics(sim_samples, y, quantiles),
            'pmf': discrete_support_metrics(sim_samples, y),
            'num_obs': int(len(y)),
            'num_samples_per_obs': int(N),
        }

    # DL (always per-carrier models, aligned with scenario generation)
    dl_samples = sample_dl_per_carrier(N, df, period='test', cs_models_dir=cs_models_dir)
    dl_samples = maybe_round_days(dl_samples, round_days)
    cov, sharp = compute_interval_metrics(dl_samples, y, alpha)
    results['dl'] = add_vs_sim({
        'mean_crps': float(crps(y, dl_samples)),
        'coverage': cov,
        'sharpness': sharp,
        'wasserstein': compute_wasserstein(dl_samples, y),
        'decision_proxy': decision_proxy_loss(dl_samples, y, late_threshold, late_weight),
        'exceedance': exceedance_calibration(dl_samples, y, exceed_thresholds, reliability_bins),
        'quantiles': high_quantile_metrics(dl_samples, y, quantiles),
        'pmf': discrete_support_metrics(dl_samples, y),
        'num_obs': int(len(y)),
        'num_samples_per_obs': int(N),
    }, dl_samples)

    # QRF
    qrf = load_qrf_model(project_root)
    if qrf is not None:
        qrf_samples = maybe_round_days(sample_qrf(N, df, qrf), round_days)
        cov, sharp = compute_interval_metrics(qrf_samples, y, alpha)
        results['qrf'] = add_vs_sim({
            'mean_crps': float(crps(y, qrf_samples)),
            'coverage': cov,
            'sharpness': sharp,
            'wasserstein': compute_wasserstein(qrf_samples, y),
            'decision_proxy': decision_proxy_loss(qrf_samples, y, late_threshold, late_weight),
            'exceedance': exceedance_calibration(qrf_samples, y, exceed_thresholds, reliability_bins),
            'quantiles': high_quantile_metrics(qrf_samples, y, quantiles),
            'pmf': discrete_support_metrics(qrf_samples, y),
            'num_obs': int(len(y)),
            'num_samples_per_obs': int(N),
        }, qrf_samples)

    # Empirical
    emp_samples = maybe_round_days(sample_empirical(N, df, order_set='test'), round_days)
    cov, sharp = compute_interval_metrics(emp_samples, y, alpha)
    results['empirical'] = add_vs_sim({
        'mean_crps': float(crps(y, emp_samples)),
        'coverage': cov,
        'sharpness': sharp,
        'wasserstein': compute_wasserstein(emp_samples, y),
        'decision_proxy': decision_proxy_loss(emp_samples, y, late_threshold, late_weight),
        'exceedance': exceedance_calibration(emp_samples, y, exceed_thresholds, reliability_bins),
        'quantiles': high_quantile_metrics(emp_samples, y, quantiles),
        'pmf': discrete_support_metrics(emp_samples, y),
        'num_obs': int(len(y)),
        'num_samples_per_obs': int(N),
    }, emp_samples)

    return results


def _parse_list(s: str, cast):
    return [cast(x) for x in s.split(',') if x.strip() != '']


def main():
    parser = argparse.ArgumentParser(description='Evaluate delivery-time scenario generators: DL vs QRF vs Empirical, incl. tail diagnostics.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Samples per observation')
    parser.add_argument('--alpha', type=float, default=0.95, help='Central interval level')
    parser.add_argument('--head', type=int, default=None, help='Limit number of test rows')
    parser.add_argument('--round_days', action='store_true', help='Round samples to nearest integer day')
    parser.add_argument('--exceed_thresholds', type=str, default='0,1,2,3', help='Comma list of thresholds for exceedance')
    parser.add_argument('--quantiles', type=str, default='0.8,0.9,0.95', help='Comma list of high quantiles')
    parser.add_argument('--reliability_bins', type=int, default=10, help='Number of bins for reliability curve')
    parser.add_argument('--late_threshold', type=float, default=0.0, help='Decision proxy late threshold')
    parser.add_argument('--late_weight', type=float, default=cfg.GAMMA_PLUS_LATE_PENALTY, help='Decision proxy late weight (>=1)')
    parser.add_argument('--compare_to_simulator', action='store_true', help='Also compare model scenarios to a simulator distribution')
    parser.add_argument('--simulator_type', type=str, default='dl', choices=['dl', 'catboost'], help='Simulator type (dl or catboost) to reference')
    parser.add_argument('--eval_simulator', action='store_true', help='Also evaluate simulator distribution vs true test empirical distribution')
    parser.add_argument('--by_carrier', action='store_true', help='Evaluate simulator distribution vs true test empirical distribution *per carrier* (uses CS loader)')
    parser.add_argument('--carrier_col', type=str, default=None, help="Carrier column name (default: auto-detect 'carrier_service_id' or 'carrier_service_id_anon')")
    parser.add_argument('--min_rows_per_carrier', type=int, default=0, help='Skip carriers with fewer than this many rows (default: 0)')
    parser.add_argument('--by_carrier_summary_csv', type=str, default=None, help='Optional path to save per-carrier simulator summary CSV')
    parser.add_argument('--cs_models_dir', type=str, default=None, help='Override carrier-specific model dir for per-carrier DL evaluation')
    parser.add_argument('--simulator_dir', type=str, default=None, help='Override simulator model dir (default: cfg.DELIVERY_TIME_SIMULATOR_PATH). E.g. data/models/delivery_time_cs/archive_simulator')
    parser.add_argument('--plot_pmf', action='store_true', help='Save PMF distribution plots for each method')
    parser.add_argument('--plots_dir', type=str, default=None, help='Directory to save plots (default: data/plots/delivery_time/pmf/)')
    args = parser.parse_args()

    if args.by_carrier:
        # By-carrier mode is intentionally focused on simulator-vs-true diagnostics.
        plots_dir = None
        if args.plot_pmf:
            plots_dir = Path(args.plots_dir) if args.plots_dir else Path('data/plots/delivery_time/pmf/by_carrier')

        by_carrier_results, summary_df = evaluate_simulator_by_carrier(
            N=args.num_samples,
            alpha=args.alpha,
            head=args.head,
            round_days=args.round_days,
            exceed_thresholds=_parse_list(args.exceed_thresholds, int),
            quantiles=_parse_list(args.quantiles, float),
            reliability_bins=args.reliability_bins,
            late_threshold=args.late_threshold,
            late_weight=args.late_weight,
            simulator_type=args.simulator_type,
            carrier_col=args.carrier_col,
            min_rows_per_carrier=args.min_rows_per_carrier,
            plots_dir=plots_dir,
            plot_pmf=args.plot_pmf,
        )

        if args.by_carrier_summary_csv:
            out_path = Path(args.by_carrier_summary_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(out_path, index=False)
            print(f"Saved by-carrier summary CSV: {out_path}")

        print('\n=== Simulator vs True (By Carrier) ===')
        if summary_df.empty:
            print("No carriers evaluated (check carrier column / min_rows_per_carrier).")
        else:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(summary_df)

        g = by_carrier_results.get('_global') or {}
        if g:
            print("\n=== Simulator Late Delivery (Expected, All Carriers) ===")
            print(f"late_delivery_pct_units_expected: {g.get('late_delivery_pct_units_expected')}")
            print(f"late_delivery_pct_rows_expected: {g.get('late_delivery_pct_rows_expected')}")
            print(f"late_delivery_pct_orders_expected: {g.get('late_delivery_pct_orders_expected')}")
            print(f"num_rows: {g.get('num_rows')} total_units: {g.get('total_units')} num_orders: {g.get('num_orders')}")

        # Avoid printing the full nested dict (can be large); exit early.
        return

    results = evaluate(
        N=args.num_samples,
        alpha=args.alpha,
        head=args.head,
        round_days=args.round_days,
        exceed_thresholds=_parse_list(args.exceed_thresholds, int),
        quantiles=_parse_list(args.quantiles, float),
        reliability_bins=args.reliability_bins,
        late_threshold=args.late_threshold,
        late_weight=args.late_weight,
        compare_to_simulator=args.compare_to_simulator,
        simulator_type=args.simulator_type,
        eval_simulator=args.eval_simulator,
        cs_models_dir=args.cs_models_dir,
        simulator_dir=args.simulator_dir,
    )

    # Generate PMF plots if requested
    if args.plot_pmf:
        plots_dir = Path(args.plots_dir) if args.plots_dir else Path('data/plots/delivery_time/pmf')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for method_name, method_results in results.items():
            pmf_data = method_results['pmf']
            support = np.array(pmf_data['support'])
            true_pmf = np.array(pmf_data['true_pmf'])
            scn_pmf = np.array(pmf_data['scn_pmf'])
            
            plot_path = plots_dir / f'pmf_distribution_{method_name}.png'
            plot_distribution(
                classes=support,
                true_dist=true_pmf,
                pred_dist=scn_pmf,
                title=f'PMF Distribution Comparison: {method_name.upper()}',
                save_path=plot_path
            )
            print(f"Saved PMF plot for {method_name}: {plot_path}")

    print('\n=== Delivery Scenario Evaluation (DL vs QRF vs Empirical) ===')
    for k, v in results.items():
        print(f"\n[{k}]")
        print(f"  mean_crps: {v['mean_crps']}")
        print(f"  coverage: {v['coverage']}")
        print(f"  sharpness: {v['sharpness']}")
        print(f"  wasserstein: {v['wasserstein']}")
        print(f"  decision_proxy: {v['decision_proxy']}")
        ex = v['exceedance']
        ex_summ = ', '.join([f"t={t}: brier={ex[t]['brier']:.4f}" for t in sorted(ex.keys())])
        print(f"  exceedance_brier: {ex_summ}")
        qd = v['quantiles']
        q_summ = ', '.join([f"q={q}: bias={qd[q]['bias']:.3f}, cov={qd[q]['coverage']:.3f}, mae={qd[q]['mae']:.3f}" for q in sorted(qd.keys())])
        print(f"  high_quantiles: {q_summ}")
        pmf = v['pmf']
        print(f"  pmf_l1: {pmf['l1']}, pmf_kl: {pmf['kl']}")
        if 'vs_sim' in v:
            vs = v['vs_sim']
            ex_sim = ', '.join([f"t={t}: brier={vs['exceedance_brier'][t]:.4f}" for t in sorted(vs['exceedance_brier'].keys())])
            q_sim = ', '.join([f"q={q}: bias={vs['high_quantiles'][q]['bias_vs_sim']:.3f}, cov={vs['high_quantiles'][q]['coverage_vs_sim']:.3f}, mae={vs['high_quantiles'][q]['mae_vs_sim']:.3f}" for q in sorted(vs['high_quantiles'].keys())])
            print(f"  vs_sim_wasserstein: {vs['wasserstein']}")
            print(f"  vs_sim_exceedance_brier: {ex_sim}")
            print(f"  vs_sim_high_quantiles: {q_sim}")
            print(f"  vs_sim_pmf_l1: {vs['pmf_l1']}, vs_sim_pmf_kl: {vs['pmf_kl']}")


if __name__ == '__main__':
    main()
