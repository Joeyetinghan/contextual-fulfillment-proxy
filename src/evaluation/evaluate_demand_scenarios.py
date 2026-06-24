import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
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

from src.scenario_generator import _get_demand_model
from src.scenario_generator import _get_hist_upper_by_sku
from src.empirical_scenarios import _get_cached_training_data
from src.utils import sample_paths


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
    plt.xlabel('Demand (Units)')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------- Metrics ----------

def crps(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None):
    """
    CRPS given samples y_pred (shape [S, N]) and true y_true (shape [N]).
    """
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
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    lower = np.quantile(samples, lower_q, axis=0)
    upper = np.quantile(samples, upper_q, axis=0)
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = covered.mean()
    sharpness = np.mean(upper - lower)
    return float(coverage), float(sharpness)


def compute_wasserstein(samples: np.ndarray, y_true: np.ndarray) -> float | None:
    if not _HAVE_SCIPY:
        return None
    dists = []
    for i in range(y_true.shape[0]):
        d = wasserstein_distance(samples[:, i], np.array([y_true[i]]))
        dists.append(d)
    return float(np.mean(dists))


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


def pmf_from_samples(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s_int = np.rint(samples).astype(int)
    vals, counts = np.unique(s_int.ravel(), return_counts=True)
    pmf = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    return vals.astype(int), pmf


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


# ---------- Samplers ----------

def _load_test_npz() -> dict:
    path = Path(cfg.DEMAND_TEST_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Test NPZ not found: {path}")
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def sample_mqrnn_total_demand(N: int, lookahead: int, device: torch.device, batch_size: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns samples [S, B] and y_true_total [B] over the first `lookahead` periods.
    Uses mini-batched inference to avoid GPU OOM.
    """
    test = _load_test_npz()
    xh = test['xh']; yh = test['yh']; xp = test['xp']
    yp = test['yp']  # [B, H]
    sku_idx = test['sku_idx']; brand_idx = test.get('brand_idx')

    B, H = yp.shape
    L = min(lookahead, H)
    y_true_total = yp[:, :L].sum(axis=1).astype(np.float32)

    # Load model + scalers
    project_root = Path(__file__).resolve().parents[1]
    model_info = _get_demand_model(project_root, period='test', verbose=False)
    if not model_info:
        return np.zeros((N, B), dtype=np.float32), y_true_total

    model = model_info['model']
    sc_h = model_info.get('past_scaler')
    sc_p = model_info.get('horizon_scaler')

    # Optional scaling (on CPU)
    xh_scaled = xh
    xp_scaled = xp
    if sc_h is not None and sc_p is not None:
        xh_scaled = sc_h.transform(xh.reshape(-1, xh.shape[-1])).reshape(xh.shape)
        xp_scaled = sc_p.transform(xp.reshape(-1, xp.shape[-1])).reshape(xp.shape)

    # Prepare tensors on CPU; move per-batch to device
    xh_t = torch.tensor(xh_scaled, dtype=torch.float32)
    yh_t = torch.tensor(yh, dtype=torch.float32)
    xp_t = torch.tensor(xp_scaled, dtype=torch.float32)
    sku_t = torch.tensor(sku_idx, dtype=torch.long)
    if brand_idx is None:
        brand_t = torch.zeros_like(sku_t)
    else:
        brand_t = torch.tensor(brand_idx, dtype=torch.long)

    # Inference in mini-batches
    bs = int(batch_size or 1024)
    Q = len(cfg.DEMAND_MODEL_QUANTILES)
    q_total = np.zeros((B, Q), dtype=np.float32)

    model = model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            for s in range(0, B, bs):
                e = min(B, s + bs)
                xb = xh_t[s:e].to(device, non_blocking=True)
                yb = yh_t[s:e].to(device, non_blocking=True)
                pb = xp_t[s:e].to(device, non_blocking=True)
                sk = sku_t[s:e].to(device, non_blocking=True)
                br = brand_t[s:e].to(device, non_blocking=True)
                qb = model(xb, yb, pb, sk, br)  # [b, H, Q]
                if qb.ndim == 2:
                    qb = qb[:, None, :]
                q_total[s:e] = qb[:, :L, :].sum(dim=1).float().cpu().numpy()
                del xb, yb, pb, sk, br, qb
                torch.cuda.empty_cache()
    except RuntimeError as err:
        if 'out of memory' in str(err).lower() and device.type == 'cuda':
            torch.cuda.empty_cache()
            # Fallback to CPU
            device = torch.device('cpu')
            model = model.to(device)
            q_total.fill(0.0)
            with torch.no_grad():
                for s in range(0, B, bs * 4):
                    e = min(B, s + bs * 4)
                    qb = model(xh_t[s:e], yh_t[s:e], xp_t[s:e], sku_t[s:e], brand_t[s:e])
                    if qb.ndim == 2:
                        qb = qb[:, None, :]
                    q_total[s:e] = qb[:, :L, :].sum(dim=1).float().cpu().numpy()
        else:
            raise

    # Build per-SKU historical upper bounds from training data (principled cap)
    # Uses the maximum cumulative demand over the first L periods in the training set for each SKU.
    # Cached per-lookahead historical upper bounds (efficient for on-the-fly inference)
    hist_upper_by_sku = _get_hist_upper_by_sku(L, order_set='test')
    
    # Sample per observation with lower bound 0 and principled upper bound
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    out = np.zeros((N, B), dtype=np.float32)
    for i in range(B):
        q = q_total[i]
        # Extrapolated upper from tail gap (fallback if quantiles are limited)
        if q.shape[0] >= 2:
            upper_extrap = float(q[-1] + (q[-1] - q[-2]))
        else:
            upper_extrap = float(q[-1])
        # Historical cap for this SKU
        hist_cap = float(hist_upper_by_sku.get(int(sku_idx[i]), 0.0))
        upper_bound = max(upper_extrap, hist_cap)
        out[:, i] = sample_paths(q, N, rng=rng, lower=0, upper=upper_bound)
    return out, y_true_total


def sample_empirical_total_demand(N: int, lookahead: int, sku_idx_test: np.ndarray) -> np.ndarray:
    """
    For each test observation with a SKU id in `sku_idx_test`, sample cumulative
    demand from training samples of the same SKU.
    Returns samples [S, B].
    """
    cached = _get_cached_training_data('test', verbose=False)
    demand_samples = cached['demand_samples']  # [M, H]
    sku_indices = cached['sku_indices']

    H = demand_samples.shape[1]
    L = min(lookahead, H)

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    B = len(sku_idx_test)
    out = np.zeros((N, B), dtype=np.float32)

    for i in range(B):
        sku = sku_idx_test[i]
        mask = sku_indices == sku
        pool = demand_samples[mask]
        if pool.size == 0:
            draws = np.zeros(N, dtype=np.float32)
        else:
            totals = pool[:, :L].sum(axis=1).astype(np.float32)
            idx = rng.integers(0, len(totals), size=N)
            draws = totals[idx]
        out[:, i] = draws
    return out


def sample_hybrid_total_demand(
    N: int,
    lookahead: int,
    device: torch.device,
    batch_size: int | None = None,
    threshold: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hybrid scenarios: high-volume SKUs via ML, slow-moving via empirical.

    High-volume defined by training mean cumulative demand over first L periods >= threshold.
    """
    test = _load_test_npz()
    yp = test['yp']
    sku_idx = test['sku_idx']

    B, H = yp.shape
    L = min(lookahead, H)
    y_true_total = yp[:, :L].sum(axis=1).astype(np.float32)

    cached = _get_cached_training_data('test', verbose=False)
    demand_samples = cached.get('demand_samples')
    sku_indices = cached.get('sku_indices')
    if demand_samples is None or sku_indices is None or len(demand_samples) == 0:
        # Fallback: all empirical
        emp = sample_empirical_total_demand(N, lookahead, sku_idx)
        return emp, y_true_total

    L_eff = min(L, demand_samples.shape[1])
    totals = demand_samples[:, :L_eff].sum(axis=1).astype(np.float32)
    df = pd.DataFrame({'sku': sku_indices, 'tot': totals})
    mean_by_sku = df.groupby('sku', sort=False)['tot'].mean().to_dict()
    high_mask = np.array([float(mean_by_sku.get(int(s), 0.0)) >= float(threshold) for s in sku_idx], dtype=bool)

    out = np.zeros((N, B), dtype=np.float32)
    any_high = bool(high_mask.any())
    any_slow = bool((~high_mask).any())

    if any_high:
        ml, _ = sample_mqrnn_total_demand(N, lookahead, device, batch_size=batch_size)
        out[:, high_mask] = ml[:, high_mask]
    if any_slow:
        emp = sample_empirical_total_demand(N, lookahead, sku_idx)
        out[:, ~high_mask] = emp[:, ~high_mask]

    return out, y_true_total

def maybe_round_int(samples: np.ndarray, do_round: bool) -> np.ndarray:
    return np.rint(samples).astype(np.float32) if do_round else samples


# ---------- Evaluation ----------

def evaluate(
    N: int,
    lookahead: int = 24,
    alpha: float = 0.9,
    head: int | None = None,
    round_int: bool = True,
    exceed_thresholds: list[int] | None = None,
    quantiles: list[float] | None = None,
    reliability_bins: int = 10,
    device: torch.device | None = None,
    batch_size: int = 1024,
    use_hybrid: bool = False,
    hybrid_threshold: float = 5.0,
) -> Dict[str, Dict[str, object]]:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = _load_test_npz()
    yp = test['yp']  # [B, H]
    sku_idx = test['sku_idx']

    B = yp.shape[0]
    if head is not None and head > 0:
        B = min(B, head)
        for k in list(test.keys()):
            if isinstance(test[k], np.ndarray) and test[k].shape[0] == yp.shape[0]:
                test[k] = test[k][:B]
        yp = yp[:B]
        sku_idx = sku_idx[:B]

    if exceed_thresholds is None:
        exceed_thresholds = [0, 1, 2, 5, 10]
    if quantiles is None:
        quantiles = [0.8, 0.9, 0.95]

    # Ground truth total demand over lookahead
    y_true_total = yp[:, :min(lookahead, yp.shape[1])].sum(axis=1).astype(np.float32)

    results: Dict[str, Dict[str, object]] = {}

    # ML (MQRNN) or Hybrid
    if use_hybrid:
        ml_samples, y_chk = sample_hybrid_total_demand(N, lookahead, device, batch_size=batch_size, threshold=hybrid_threshold)
    else:
        ml_samples, y_chk = sample_mqrnn_total_demand(N, lookahead, device, batch_size=batch_size)
    ml_samples = ml_samples[:, :B]
    if y_chk.shape[0] >= B:
        y_ml_true = y_chk[:B]
    else:
        y_ml_true = y_true_total
    ml_samples = maybe_round_int(ml_samples, round_int)
    cov, sharp = compute_interval_metrics(ml_samples, y_ml_true, alpha)
    results['ml'] = {
        'mean_crps': float(crps(y_ml_true, ml_samples)),
        'coverage': cov,
        'sharpness': sharp,
        'wasserstein': compute_wasserstein(ml_samples, y_ml_true),
        'exceedance': exceedance_calibration(ml_samples, y_ml_true, exceed_thresholds, reliability_bins),
        'quantiles': high_quantile_metrics(ml_samples, y_ml_true, quantiles),
        'pmf': discrete_support_metrics(ml_samples, y_ml_true),
        'num_obs': int(B),
        'num_samples_per_obs': int(N),
    }

    # Empirical
    emp_samples = sample_empirical_total_demand(N, lookahead, sku_idx[:B])
    emp_samples = maybe_round_int(emp_samples, round_int)
    cov, sharp = compute_interval_metrics(emp_samples, y_true_total[:B], alpha)
    results['empirical'] = {
        'mean_crps': float(crps(y_true_total[:B], emp_samples)),
        'coverage': cov,
        'sharpness': sharp,
        'wasserstein': compute_wasserstein(emp_samples, y_true_total[:B]),
        'exceedance': exceedance_calibration(emp_samples, y_true_total[:B], exceed_thresholds, reliability_bins),
        'quantiles': high_quantile_metrics(emp_samples, y_true_total[:B], quantiles),
        'pmf': discrete_support_metrics(emp_samples, y_true_total[:B]),
        'num_obs': int(B),
        'num_samples_per_obs': int(N),
    }

    return results


def _parse_list(s: str, cast):
    return [cast(x) for x in s.split(',') if x.strip() != '']


def main():
    parser = argparse.ArgumentParser(description='Evaluate demand scenario generators: ML (MQRNN), Empirical, or Hybrid.')
    parser.add_argument('--num_samples', type=int, default=1000, help='Samples per observation')
    parser.add_argument('--lookahead', type=int, default=24, help='Horizon length to sum over')
    parser.add_argument('--alpha', type=float, default=0.95, help='Central interval level')
    parser.add_argument('--head', type=int, default=None, help='Limit number of test rows')
    parser.add_argument('--round_int', action='store_true', help='Round samples to nearest integer')
    parser.add_argument('--exceed_thresholds', type=str, default='0,1,2,5,10', help='Comma list of thresholds for exceedance')
    parser.add_argument('--quantiles', type=str, default='0.8,0.9,0.95', help='Comma list of high quantiles')
    parser.add_argument('--reliability_bins', type=int, default=10, help='Number of bins for reliability curve')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size for ML inference')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Force device (default: auto)')
    parser.add_argument('--plot_pmf', action='store_true', help='Save PMF distribution plots for each method')
    parser.add_argument('--use_hybrid', action='store_true', help='Use hybrid scenarios: ML for high-volume, empirical for slow-moving')
    parser.add_argument('--hybrid_threshold', type=float, default=100.0, help='Threshold for high volume (mean cumulative demand over lookahead)')
    parser.add_argument('--plots_dir', type=str, default=None, help='Directory to save plots (default: data/plots/demand/pmf/)')
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else None

    results = evaluate(
        N=args.num_samples,
        lookahead=args.lookahead,
        alpha=args.alpha,
        head=args.head,
        round_int=args.round_int,
        exceed_thresholds=_parse_list(args.exceed_thresholds, int),
        quantiles=_parse_list(args.quantiles, float),
        reliability_bins=args.reliability_bins,
        device=dev,
        batch_size=args.batch_size,
        use_hybrid=args.use_hybrid,
        hybrid_threshold=args.hybrid_threshold,
    )

    # Generate PMF plots if requested
    if args.plot_pmf:
        plots_dir = Path(args.plots_dir) if args.plots_dir else Path('data/plots/demand/pmf')
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

    print('\n=== Demand Scenario Evaluation (ML vs Empirical) ===')
    for k, v in results.items():
        print(f"\n[{k}]")
        print(f"  mean_crps: {v['mean_crps']}")
        print(f"  coverage: {v['coverage']}")
        print(f"  sharpness: {v['sharpness']}")
        print(f"  wasserstein: {v['wasserstein']}")
        ex = v['exceedance']
        ex_summ = ', '.join([f"t={t}: brier={ex[t]['brier']:.4f}" for t in sorted(ex.keys())])
        print(f"  exceedance_brier: {ex_summ}")
        qd = v['quantiles']
        q_summ = ', '.join([f"q={q}: bias={qd[q]['bias']:.3f}, cov={qd[q]['coverage']:.3f}, mae={qd[q]['mae']:.3f}" for q in sorted(qd.keys())])
        print(f"  high_quantiles: {q_summ}")
        pmf = v['pmf']
        print(f"  pmf_l1: {pmf['l1']}, pmf_kl: {pmf['kl']}")


if __name__ == '__main__':
    main()
