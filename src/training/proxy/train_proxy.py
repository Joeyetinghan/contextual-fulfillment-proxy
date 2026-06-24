# ─────────────────────────────────────────────────────────────────────────────
# 1.  Train the proxy model + ProxyLoss on proxy_flat_*_dataset.pt
# 2.  Every N epochs print loss + classification metrics
# 3.  After training: visualise Actual vs Pred / (optionally) Repaired
#     for a few random samples.
# -----------------------------------------------------------------------------


import os
import time
import shutil
import torch, torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader
import numpy as np, random, json, gc
from pathlib import Path; from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

try:
    from torch import amp as torch_amp
except Exception:
    torch_amp = None

import src.config as cfg
from src.model.proxy_variants import build_proxy_model, canonicalize_architecture
from src.model.proxy_loss  import ProxyLoss
from src.model.proxy_inference import proxy_inference
from src.model.hierarchical_proxy_inference import hierarchical_proxy_inference
from src.training.proxy.cli import parse_train_proxy_args
from src.training.proxy.reporting import print_best_so_far, print_train_val_metrics, print_val_metrics
from src.training.proxy.ub_metrics import compute_proxy_ub_metrics, print_ub_metrics
from datetime import datetime, timezone

def _autocast_ctx(enabled: bool, device_type: str = 'cuda'):
    if not enabled:
        return nullcontext()
    if torch_amp is not None and hasattr(torch_amp, 'autocast'):
        try:
            return torch_amp.autocast(device_type=device_type, enabled=enabled)
        except TypeError:
            pass
    try:
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast(enabled=enabled)
    except Exception:
        return nullcontext()


def _grad_scaler(enabled: bool, device_type: str = 'cuda'):
    if not enabled:
        return None
    if torch_amp is not None and hasattr(torch_amp, 'GradScaler'):
        try:
            return torch_amp.GradScaler(device_type=device_type, enabled=enabled)
        except TypeError:
            pass
    try:
        from torch.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(enabled=enabled)
    except Exception:
        return None

# ────────── helpers: seed, load tensors, dataset, collate ────────── #
def seed_all(seed=cfg.RANDOM_SEED):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_flat(split: str = "proxy_train"):
    """
    Concatenate all proxy‑flat *.pt files for the requested split.
    Memory-efficient version:
      1. Pass 1: compute total size, collect metadata, read shapes.
      2. Pre-allocate exact output tensors.
      3. Pass 2: load and copy data into pre-allocated slots.
    
    This reduces peak memory from ~2x dataset size (concatenation)
    to ~1x dataset size + 1 file buffer.

    Returns
    -------
    tensors : dict[str, Tensor]   2‑D or 3‑D stacked data blocks
    info    : dict                scalar meta + per‑row metadata list
    """
    root = cfg.PROXY_TRAINING_DATA_DIR if split == "proxy_train" \
           else cfg.PROXY_TEST_DATA_DIR
    files = sorted(root.glob("proxy_flat_*_dataset.pt"))
    if not files:
        raise FileNotFoundError(f"No proxy_flat_*.pt in {root}")

    PROXY_FEATURE_KEYS = [
        "global_features", "dc_features", "option_features",
        "scenario_demand", "delivery_penalty",
        "targets", "quantity_vector",
        "sku_idx", "brand_idx", "eligibility_mask",
    ]
    
    # Determine which keys to load based on first file
    first_file = torch.load(files[0], map_location='cpu', weights_only=False)
    use_proxy_feature_format = 'global_features' in first_file
    
    if not use_proxy_feature_format:
        raise ValueError(
            "Old proxy datasets are no longer supported. "
            "Please regenerate features using: python -m src.proxy_feature_engineering"
        )
    
    del first_file
    gc.collect()
    
    T_KEYS = PROXY_FEATURE_KEYS

    # --- Pass 1: Compute total size & get shapes (minimal memory) ---
    print(f"Scanning {len(files)} files to compute dataset size...")
    total_rows = 0
    ref_info = None
    shapes = {}
    dtypes = {}
    file_row_counts = []

    # Iterate files to calculate total N using minimal memory
    for p in tqdm(files, desc="Pass 1/2 (Scanning)"):
        raw = torch.load(p, map_location='cpu', weights_only=False)
        
        # Number of rows B - always use proxy feature tensors.
        B = raw['global_features'].shape[0]
        file_row_counts.append(B)
        total_rows += B
        
        # Capture reference info from the first file
        if ref_info is None:
            ref_info = {
                "num_dcs":       raw.get("num_dcs"),
                "num_carriers":  raw.get("num_carriers", 1),
                "scenario_len":  raw.get("scenario_len"),
                "sku_dim":       raw.get("sku_dim"),
                "brand_dim":     raw.get("brand_dim"),
                "feature_names": raw.get("feature_names"),
                "dcs":           raw.get("dcs"),
                "carriers":      raw.get("carriers"),
                "use_proxy_feature_format": use_proxy_feature_format,
            }
            ref_info.update({
                "global_feature_dim": raw.get("global_feature_dim"),
                "dc_feature_dim": raw.get("dc_feature_dim"),
                "option_feature_dim": raw.get("option_feature_dim"),
                "global_feature_names": raw.get("global_feature_names"),
                "dc_feature_names": raw.get("dc_feature_names"),
                "option_feature_names": raw.get("option_feature_names"),
            })
            # Record shapes and dtypes for pre-allocation
            for k in T_KEYS:
                if k in raw:
                    shapes[k] = raw[k].shape[1:]
                    dtypes[k] = raw[k].dtype
        
        # Aggressively free memory
        del raw
        gc.collect()
        
    # Collect metadata in a separate lightweight pass
    print("Collecting metadata...")
    metadata_list = []
    for p in tqdm(files, desc="Metadata"):
        raw = torch.load(p, map_location='cpu', weights_only=False)
        if "metadata_rows" in raw:
            # Attach the originating date (from filename) so downstream tools can locate CSAA dumps.
            # Filename format: proxy_flat_YYYY-MM-DD_dataset.pt
            date = None
            for token in p.stem.split("_"):
                if len(token) == 10 and token[4] == "-" and token[7] == "-":
                    date = token
                    break
            rows = raw["metadata_rows"]
            if date and rows and isinstance(rows[0], dict):
                for r in rows:
                    r.setdefault("date", date)
                    r.setdefault("split", split)
            metadata_list.extend(rows)
        del raw
        gc.collect()

    # --- Pre-allocate final tensors ---
    print(f"Allocating memory for {total_rows} samples...")
    tensors = {}
    for k, shp in shapes.items():
        # Store scenarios (and delivery_penalty) in float16 for memory efficiency
        if k in ["scenario_demand", "scenario_cost", "delivery_penalty"]:
            dtype = torch.float16
        else:
            dtype = dtypes[k]
        # Allocating on CPU to be safe
        tensors[k] = torch.empty((total_rows, *shp), dtype=dtype, device='cpu')

    # --- Pass 2: Fill tensors ---
    start_idx = 0
    for p in tqdm(files, desc="Pass 2/2 (Loading)"):
        raw = torch.load(p, map_location='cpu', weights_only=False)
        B = raw['global_features'].shape[0]
        end_idx = start_idx + B
        
        for k in tensors:
            if k in raw:
                # Convert scenario tensors to float16 for memory efficiency
                if k in ["scenario_demand", "scenario_cost", "delivery_penalty"]:
                    tensors[k][start_idx:end_idx] = raw[k].half()
                else:
                    tensors[k][start_idx:end_idx] = raw[k]
        
        start_idx = end_idx
        del raw
        gc.collect()

    # Note: Files are loaded in chronological order (sorted YYYY-MM-DD filenames)
    # Within-date ordering follows directory iteration from feature engineering
    print(f"  Data loaded in chronological order by simulation date (file-based)")
    
    # Combine info
    info = ref_info
    info["metadata"] = metadata_list
    
    return tensors, info

class FlatDS(torch.utils.data.Dataset):
    def __init__(self,t): 
        self.t=t
        # Always use proxy feature tensors.
        self.size_key = 'global_features'
    def __len__(self): return self.t[self.size_key].size(0)
    def __getitem__(self,i): return {k:v[i] for k,v in self.t.items()}


def collate(b): return {k:torch.stack([d[k] for d in b]) for k in b[0]}



def plot_one(sample_idx, plan_pred, plan_rep, plan_true, save_dir, num_dcs=10):
    dcs = np.arange(1, num_dcs+1)
    plt.figure(figsize=(8,3))
    plt.bar(dcs-0.25, plan_true[:num_dcs], width=0.25, label='True', color='tab:blue')
    plt.bar(dcs      , plan_pred[:num_dcs], width=0.25, label='Pred', color='tab:orange')
    plt.bar(dcs+0.25 , plan_rep [:num_dcs], width=0.25, label='Repaired', color='tab:green')
    plt.xticks(dcs); plt.xlabel('DC'); plt.ylabel('Units')
    plt.title(f'Sample {sample_idx}')
    plt.legend(); plt.tight_layout()
    plt.savefig(save_dir/f"sample_{sample_idx}.png"); plt.close()

# ───────────── epoch loop with metrics ───────────── #
def run_epoch(
    model,
    loader,
    device,
    loss_fn,
    opt=None,
    amp_scaler=None,
    thr=0.5,
    compute_metrics=True,
    num_dcs=None,
    num_carriers=None,
    repair_strategy="default",
    non_blocking=False,
):
    num_batches = len(loader)
    if num_batches == 0:
        raise ValueError(
            "run_epoch received an empty DataLoader. "
            "Check split_ratio/refit_full_data and dataset size."
        )
    train = opt is not None
    model.train() if train else model.eval()
    if os.getenv("PROXY_LOSS_COMPONENTS") == "1":
        # Reset per-epoch so components print once per epoch (first batch).
        loss_fn._printed_components = False
    agg = dict(loss=0., rows=0)
    if compute_metrics:
        agg.update(dict(hit1_pre=0, hit5_pre=0, hit1_post=0, hit5_post=0, 
                   mrr_dc=0, mrr_carrier=0, mrr_joint=0,
                   joint_hit1=0, joint_hit5=0,
                   hit1_pre_repaired=0, hit1_post_repaired=0,
                   joint_hit1_repaired=0,
                   repair_changed=0,
                   repair_over_inv=0,
                   repair_over_inv_qty=0.0,
                   repair_raw_ineligible=0,
                   repair_raw_primary_oos=0,
                   repair_drop_primary=0,
                   repair_carrier_changed=0,
                   repair_raw_primary_ineligible=0,
                   repair_primary_changed=0,
                   repair_split=0,
                   repair_unfulfilled=0,
                   repair_unfulfilled_qty=0.0,
                   repair_moved_qty=0.0,
                   demand_total=0.0))
    
    ctx = torch.enable_grad() if train else torch.inference_mode()
    example_logs = 0
    example_limit = 3
    with ctx:
        for batch in tqdm(loader, desc="Processing batch"):
            # Structure-aware forward pass
            use_amp_inputs = (amp_scaler is not None and device.type == 'cuda')
            input_dtype = torch.float16 if use_amp_inputs else torch.float32
            batch['global_features'] = batch['global_features'].to(device, non_blocking=non_blocking)
            batch['dc_features'] = batch['dc_features'].to(device, non_blocking=non_blocking)
            batch['option_features'] = batch['option_features'].to(device, non_blocking=non_blocking)
            batch['scenario_demand'] = batch['scenario_demand'].to(
                device, dtype=input_dtype, non_blocking=non_blocking
            )
            batch['delivery_penalty'] = batch['delivery_penalty'].to(
                device, dtype=input_dtype, non_blocking=non_blocking
            )
            batch['sku_idx'] = batch['sku_idx'].to(device, non_blocking=non_blocking)
            batch['brand_idx'] = batch['brand_idx'].to(device, non_blocking=non_blocking)
            batch['targets'] = batch['targets'].to(device, non_blocking=non_blocking)
            batch['quantity_vector'] = batch['quantity_vector'].to(device, non_blocking=non_blocking)
            batch['base_cost_raw'] = batch['base_cost_raw'].to(
                device, dtype=input_dtype, non_blocking=non_blocking
            )
            
            # Mixed precision forward pass
            # Optional NaN/Inf checks (toggle via env or argparse if needed)
            debug_nan = getattr(run_epoch, "_debug_nan", False)
            debug_loss_stats = getattr(run_epoch, "_debug_loss_stats", False)
            debug_batch_eligibility = getattr(run_epoch, "_debug_batch_eligibility", False)
            if debug_nan:
                for k in ['global_features', 'dc_features', 'option_features', 'scenario_demand', 'delivery_penalty', 'targets', 'quantity_vector', 'base_cost_raw']:
                    t = batch.get(k)
                    if t is None:
                        continue
                    if torch.isnan(t).any() or torch.isinf(t).any():
                        raise ValueError(f"[nan-check] {k} has NaN/Inf")

            with _autocast_ctx(enabled=(amp_scaler is not None), device_type='cuda'):
                output = model(
                    global_feats=batch['global_features'],
                    dc_feats=batch['dc_features'],
                    option_feats=batch['option_features'],
                    demand_scenarios=batch['scenario_demand'],
                    delivery_penalty=batch['delivery_penalty'],
                    sku_idx=batch['sku_idx'],
                    brand_idx=batch['brand_idx']
                )
                
                # Compute loss (unified logits [B, D*C])
                # Extract inventory from dc_features: [B, D, K_dc] -> first feature is inventory_level [B, D]
                inventory_vec = batch['dc_features'][:, :, 0]  # [B, D] - inventory_level per DC
                
                # Get eligibility mask if available
                eligibility_mask = batch.get('eligibility_mask', None)
                if eligibility_mask is not None:
                    eligibility_mask = eligibility_mask.to(device, non_blocking=non_blocking)

                # Expected cost scenarios: raw base_cost + delivery_penalty
                # Use in-place add to avoid an additional [B,S,D,C] allocation.
                base_cost = batch['base_cost_raw']  # [B, D, C]
                scenario_costs = batch['delivery_penalty']  # [B, S, D, C]
                scenario_costs.add_(base_cost.unsqueeze(1))
                scenario_costs = scenario_costs.view(scenario_costs.size(0), scenario_costs.size(1), -1)  # [B, S, D*C]

                if debug_loss_stats:
                    with torch.no_grad():
                        sc_min = scenario_costs.min().item()
                        sc_max = scenario_costs.max().item()
                        sc_mean = scenario_costs.mean().item()
                        if isinstance(output, tuple):
                            logits_dc_dbg, logits_carrier_dbg = output
                            log_min = min(logits_dc_dbg.min().item(), logits_carrier_dbg.min().item())
                            log_max = max(logits_dc_dbg.max().item(), logits_carrier_dbg.max().item())
                        else:
                            log_min = output.min().item()
                            log_max = output.max().item()
                        print(f"[loss-stats] scenario_costs min/mean/max: {sc_min:.4f}/{sc_mean:.4f}/{sc_max:.4f}")
                        print(f"[loss-stats] logits min/max: {log_min:.4f}/{log_max:.4f}")
                    run_epoch._debug_loss_stats = False

                if debug_batch_eligibility:
                    elig = batch.get('eligibility_mask')
                    if elig is None:
                        print("[eligibility-batch] no eligibility_mask in batch")
                    else:
                        targets = batch.get('targets')
                        if targets is not None and elig.device != targets.device:
                            elig = elig.to(targets.device)
                        elig = elig.float()
                        if elig.ndim == 3:
                            elig_flat = elig.view(elig.size(0), -1)
                        elif elig.ndim == 2:
                            elig_flat = elig
                        else:
                            elig_flat = elig.view(elig.size(0), -1)
                        mismatch_rows = 0
                        mismatch_entries = 0
                        if targets is not None:
                            targets_pos = targets > 0
                            mismatch = targets_pos & (elig_flat <= 0)
                            mismatch_rows = mismatch.any(dim=1).sum().item()
                            mismatch_entries = mismatch.sum().item()
                        elig_counts = elig_flat.sum(dim=1)
                        print(
                            "[eligibility-batch] "
                            f"eligible_per_row min/mean/max: "
                            f"{elig_counts.min().item():.1f}/"
                            f"{elig_counts.mean().item():.1f}/"
                            f"{elig_counts.max().item():.1f} "
                            f"mismatch_rows={int(mismatch_rows)} "
                            f"mismatch_entries={int(mismatch_entries)}"
                        )
                    run_epoch._debug_batch_eligibility = False
                
                # ProxyLoss.forward(sel_logits, qty_true, inventory_vec, demand_scalar, ..., scenario_costs, eligibility_mask)
                loss = loss_fn(
                    output,
                    batch['targets'],
                    inventory_vec=inventory_vec,
                    demand_scalar=batch['quantity_vector'],
                    scenario_costs=scenario_costs,
                    eligibility_mask=eligibility_mask,
                )
            
            if debug_nan:
                if isinstance(output, tuple):
                    out_nan = any(torch.isnan(t).any() or torch.isinf(t).any() for t in output)
                else:
                    out_nan = torch.isnan(output).any() or torch.isinf(output).any()
                if out_nan:
                    raise ValueError("[nan-check] model output has NaN/Inf")
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    raise ValueError("[nan-check] loss has NaN/Inf")

            if train:
                opt.zero_grad(set_to_none=True)
                if amp_scaler is not None:
                    amp_scaler.scale(loss).backward()
                    try:
                        amp_scaler.step(opt)
                        amp_scaler.update()
                    except AssertionError:
                        # Fallback: if AMP scaler didn't record inf checks, do a normal step
                        if not getattr(run_epoch, "_amp_step_warned", False):
                            print("[amp] Warning: GradScaler had no inf checks; falling back to opt.step()")
                            run_epoch._amp_step_warned = True
                        opt.step()
                else:
                    loss.backward()
                    opt.step()
            
            # Handle both unified [B, D*C] and hierarchical (logits_dc, logits_carrier) outputs
            is_hierarchical = isinstance(output, tuple)
            
            if is_hierarchical:
                logits_dc, logits_carrier = output
                B = logits_dc.size(0)
            else:
                logits_flat = output
                B = logits_flat.size(0)
            
            agg['loss'] += loss.item()
            agg['rows'] += B
            
            if compute_metrics:
                D, C = num_dcs, num_carriers
                targets = batch['targets'].cpu()
                
                # Get eligibility mask if available
                elig_mask_cpu = None
                if 'eligibility_mask' in batch and batch['eligibility_mask'] is not None:
                    elig_mask_cpu = batch['eligibility_mask'].detach().cpu()  # [B, D*C]
                
                if is_hierarchical:
                    # Hierarchical: already have DC and Carrier logits
                    logits_dc_cpu = logits_dc.detach().cpu().float()  # [B, D]
                    logits_carrier_cpu = logits_carrier.detach().cpu().float()  # [B, D, C]
                    logits_grid = logits_carrier_cpu  # Use carrier logits for grid
                else:
                    # Unified: reshape to grid for metrics
                    logits_flat_cpu = logits_flat.detach().cpu().float()
                    logits_grid = logits_flat_cpu.view(B, D, C)
                
                # Apply eligibility mask to logits before softmax
                if elig_mask_cpu is not None:
                    elig_grid = elig_mask_cpu.view(B, D, C)  # [B, D, C]
                    neg_inf = torch.finfo(logits_grid.dtype).min
                    
                    if is_hierarchical:
                        # Mask DC logits: DC is eligible if any carrier is eligible
                        elig_dc = (elig_grid.sum(dim=2) > 0).float()  # [B, D]
                        logits_dc_cpu = torch.where(elig_dc > 0, logits_dc_cpu, torch.full_like(logits_dc_cpu, neg_inf))
                        
                        # Mask carrier logits per DC
                        logits_carrier_cpu = torch.where(elig_grid > 0, logits_carrier_cpu, torch.full_like(logits_carrier_cpu, neg_inf))
                        logits_grid = logits_carrier_cpu
                    else:
                        # Unified: mask grid logits directly
                        logits_grid = torch.where(elig_grid > 0, logits_grid, torch.full_like(logits_grid, neg_inf))
                
                # Targets [B, D*C] -> [B, D, C]
                targets_grid = targets.view(B, D, C)
                
                # Find true (DC, Carrier) pair
                qty_per_dc = targets_grid.sum(dim=2)  # [B, D]
                true_dc_idx = qty_per_dc.argmax(dim=1)  # [B]
                batch_indices = torch.arange(B)
                true_carrier_idx = targets_grid[batch_indices, true_dc_idx, :].argmax(dim=1)  # [B]
                
                # DC marginal probabilities
                if is_hierarchical:
                    # Hierarchical: use DC logits directly
                    probs_dc = torch.softmax(logits_dc_cpu, dim=1)  # [B, D]
                    # Carrier probabilities are already conditioned on DC in the model
                    # For joint probability: P(dc, carrier) = P(dc) * P(carrier|dc)
                    probs_carrier_per_dc = torch.softmax(logits_carrier_cpu, dim=2)  # [B, D, C]
                    probs_grid = probs_dc.unsqueeze(2) * probs_carrier_per_dc  # [B, D, C]
                else:
                    # Unified: marginalize over carriers
                    probs_grid = torch.softmax(logits_grid.view(B, -1), dim=1).view(B, D, C)
                    probs_dc = probs_grid.sum(dim=2)  # [B, D]
                
                # DC metrics
                best1_dc = probs_dc.topk(1, dim=1).indices
                best5_dc = probs_dc.topk(min(5, D), dim=1).indices
                targets_dc = (qty_per_dc > 0).float()
                hit1_dc = targets_dc.gather(1, best1_dc).any(1).float()
                hit5_dc = targets_dc.gather(1, best5_dc).any(1).float()
                agg['hit1_pre'] += hit1_dc.sum().item()
                agg['hit5_pre'] += hit5_dc.sum().item()
                
                sorted_dc = probs_dc.argsort(dim=1, descending=True)
                dc_matches = (sorted_dc == true_dc_idx.unsqueeze(1)).nonzero(as_tuple=False)
                if dc_matches.size(0) == B:
                    agg['mrr_dc'] += (1.0 / (dc_matches[:, 1] + 1).float()).sum().item()
                
                # Carrier metrics (DC-independent): marginalize over DCs
                probs_carrier_marginal = probs_grid.sum(dim=1)  # [B, C]
                best1_carrier = probs_carrier_marginal.topk(1, dim=1).indices
                best5_carrier = probs_carrier_marginal.topk(min(5, C), dim=1).indices
                hit1_carrier = (best1_carrier == true_carrier_idx.unsqueeze(1)).any(1).float()
                hit5_carrier = (best5_carrier == true_carrier_idx.unsqueeze(1)).any(1).float()
                agg['hit1_post'] += hit1_carrier.sum().item()
                agg['hit5_post'] += hit5_carrier.sum().item()
                
                sorted_carrier = probs_carrier_marginal.argsort(dim=1, descending=True)
                carrier_matches = (sorted_carrier == true_carrier_idx.unsqueeze(1)).nonzero(as_tuple=False)
                if carrier_matches.size(0) == B:
                    agg['mrr_carrier'] += (1.0 / (carrier_matches[:, 1] + 1).float()).sum().item()
                
                # Joint (DC, Carrier) metrics
                probs_flat = probs_grid.view(B, -1)
                true_joint_idx = true_dc_idx * C + true_carrier_idx
                topk_joint_1 = probs_flat.topk(1, dim=1).indices
                topk_joint_5 = probs_flat.topk(min(5, D*C), dim=1).indices
                joint_hit1 = (topk_joint_1 == true_joint_idx.unsqueeze(1)).any(1).float()
                joint_hit5 = (topk_joint_5 == true_joint_idx.unsqueeze(1)).any(1).float()
                agg['joint_hit1'] += joint_hit1.sum().item()
                agg['joint_hit5'] += joint_hit5.sum().item()
                
                sorted_joint = probs_flat.argsort(dim=1, descending=True)
                joint_matches = (sorted_joint == true_joint_idx.unsqueeze(1)).nonzero(as_tuple=False)
                if joint_matches.size(0) == B:
                    agg['mrr_joint'] += (1.0 / (joint_matches[:, 1] + 1).float()).sum().item()
                
                # ===== Repaired Metrics (with constraint repair) =====
                inventory_cpu = inventory_vec.detach().cpu()
                demand_cpu = batch['quantity_vector'].detach().cpu()
                
                # Run inference with repair=True
                if is_hierarchical:
                    plan_raw, plan_repaired, carrier_repaired, carrier_raw = hierarchical_proxy_inference(
                        (logits_dc_cpu, logits_carrier_cpu),
                        inventory_cpu,
                        demand_cpu,
                        repair=True,
                        eligibility_mask=elig_mask_cpu,
                        debug=False,
                        return_raw_carrier=True,
                        repair_strategy=repair_strategy
                    )
                else:
                    logits_unified = logits_grid.view(B, -1)
                    _, plan_repaired, carrier_repaired = proxy_inference(
                        logits_unified,
                        inventory_cpu,
                        demand_cpu,
                        repair=True,
                        eligibility_mask=elig_mask_cpu,
                        num_dcs=D,
                        num_carriers=C,
                        debug=False,
                        repair_strategy=repair_strategy
                    )
                    _, plan_raw, carrier_raw = proxy_inference(
                        logits_unified,
                        inventory_cpu,
                        demand_cpu,
                        repair=False,
                        eligibility_mask=elig_mask_cpu,
                        num_dcs=D,
                        num_carriers=C,
                        debug=False,
                        repair_strategy=repair_strategy
                    )
                
                # Extract repaired (DC, Carrier) from plan
                repaired_dc_mask = (plan_repaired > 0)  # [B, D]
                repaired_dc_idx = repaired_dc_mask.float().argmax(dim=1)  # Primary DC
                repaired_carrier_idx = carrier_repaired[batch_indices, repaired_dc_idx]  # [B]
                
                # DC metrics (repaired)
                hit1_dc_rep = (repaired_dc_idx == true_dc_idx).float()
                agg['hit1_pre_repaired'] += hit1_dc_rep.sum().item()
                
                # DC allocation available in plan_repaired; keep only hit@1 for repaired metrics
                
                # Carrier metrics (repaired, at true DC)
                # Carrier metrics (repaired, DC-independent): compare chosen carrier to true carrier
                hit1_carrier_rep = (repaired_carrier_idx == true_carrier_idx).float()
                agg['hit1_post_repaired'] += hit1_carrier_rep.sum().item()
                                
                # Joint metrics (repaired)
                true_joint_rep = repaired_dc_idx * C + repaired_carrier_idx
                joint_hit1_rep = (true_joint_rep == true_joint_idx).float()
                agg['joint_hit1_repaired'] += joint_hit1_rep.sum().item()

                # ===== Repair diagnostics =====
                demand_vec = demand_cpu.view(-1)
                inv_cpu = inventory_cpu
                plan_raw_f = plan_raw.float()
                plan_rep_f = plan_repaired.float()
                raw_total = plan_raw_f.sum(dim=1)
                rep_total = plan_rep_f.sum(dim=1)

                raw_over = (plan_raw_f - inv_cpu.float()).clamp_min(0).sum(dim=1)
                agg['repair_over_inv'] += (raw_over > 0).sum().item()
                agg['repair_over_inv_qty'] += raw_over.sum().item()

                changed = (plan_raw != plan_repaired).any(dim=1)
                agg['repair_changed'] += changed.sum().item()

                raw_any = raw_total > 0
                rep_any = rep_total > 0
                primary_raw = plan_raw_f.argmax(dim=1)
                primary_rep = plan_rep_f.argmax(dim=1)
                primary_changed = (primary_raw != primary_rep) & raw_any & rep_any
                agg['repair_primary_changed'] += primary_changed.sum().item()

                rep_split = (plan_repaired > 0).sum(dim=1) > 1
                agg['repair_split'] += rep_split.sum().item()

                unfulfilled = (demand_vec - rep_total).clamp_min(0)
                agg['repair_unfulfilled'] += (unfulfilled > 0).sum().item()
                agg['repair_unfulfilled_qty'] += unfulfilled.sum().item()

                moved_qty = (plan_rep_f - plan_raw_f).abs().sum(dim=1) * 0.5
                agg['repair_moved_qty'] += moved_qty.sum().item()
                agg['demand_total'] += demand_vec.sum().item()

                # Primary DC diagnostics
                raw_primary_alloc = plan_raw_f[batch_indices, primary_raw]
                raw_primary_inv = inv_cpu[batch_indices, primary_raw].float()
                primary_oos = raw_any & (raw_primary_alloc > raw_primary_inv)
                agg['repair_raw_primary_oos'] += primary_oos.sum().item()

                drop_primary = raw_any & (plan_rep_f[batch_indices, primary_raw] <= 0)
                agg['repair_drop_primary'] += drop_primary.sum().item()

                carrier_raw_primary = carrier_raw[batch_indices, primary_raw]
                carrier_rep_at_raw = carrier_repaired[batch_indices, primary_raw]
                carrier_changed = (
                    raw_any
                    & (plan_rep_f[batch_indices, primary_raw] > 0)
                    & (carrier_raw_primary >= 0)
                    & (carrier_rep_at_raw >= 0)
                    & (carrier_raw_primary != carrier_rep_at_raw)
                )
                agg['repair_carrier_changed'] += carrier_changed.sum().item()

                # Eligibility violations in raw plan (if available)
                raw_ineligible_any = None
                raw_primary_ineligible = None
                if elig_mask_cpu is not None:
                    elig_grid = elig_mask_cpu.view(B, D, C) if elig_mask_cpu.ndim == 2 else elig_mask_cpu
                    carrier_raw_clamped = carrier_raw.clamp(min=0)
                    elig_for_raw = torch.gather(elig_grid, 2, carrier_raw_clamped.unsqueeze(2)).squeeze(2)
                    raw_alloc_mask = plan_raw_f > 0
                    raw_ineligible = raw_alloc_mask & ((carrier_raw < 0) | (elig_for_raw <= 0))
                    raw_ineligible_any = raw_ineligible.any(dim=1)
                    agg['repair_raw_ineligible'] += raw_ineligible_any.sum().item()

                    carrier_raw_primary_clamped = carrier_raw_primary.clamp(min=0)
                    raw_primary_elig = elig_grid[batch_indices, primary_raw, carrier_raw_primary_clamped]
                    raw_primary_ineligible = raw_any & ((carrier_raw_primary < 0) | (raw_primary_elig <= 0))
                    agg['repair_raw_primary_ineligible'] += raw_primary_ineligible.sum().item()

                if compute_metrics and example_logs < example_limit:
                    interesting = (
                        (raw_over > 0)
                        | primary_changed
                        | rep_split
                        | (unfulfilled > 0)
                    )
                    if raw_ineligible_any is not None:
                        interesting = interesting | raw_ineligible_any
                    idxs = interesting.nonzero(as_tuple=True)[0][: (example_limit - example_logs)]
                    for idx in idxs.tolist():
                        raw_dc = int(primary_raw[idx].item())
                        rep_dc = int(primary_rep[idx].item())
                        raw_car = int(carrier_raw[idx, raw_dc].item())
                        rep_car = int(carrier_repaired[idx, rep_dc].item())
                        msg = (
                            "[repair-example] "
                            f"i={idx} demand={float(demand_vec[idx]):.1f} "
                            f"raw_total={float(raw_total[idx]):.1f} rep_total={float(rep_total[idx]):.1f} "
                            f"raw_dc={raw_dc} rep_dc={rep_dc} raw_car={raw_car} rep_car={rep_car} "
                            f"raw_over={float(raw_over[idx]):.1f} "
                            f"primary_oos={bool(primary_oos[idx])} "
                            f"drop_primary={bool(drop_primary[idx])} "
                            f"split={bool(rep_split[idx])} "
                            f"unfulfilled={float(unfulfilled[idx]):.1f}"
                        )
                        if raw_ineligible_any is not None:
                            msg += f" raw_ineligible={bool(raw_ineligible_any[idx])}"
                        if raw_primary_ineligible is not None:
                            msg += f" raw_primary_ineligible={bool(raw_primary_ineligible[idx])}"
                        print(msg)
                        example_logs += 1
                                
    loss = agg['loss'] / num_batches
    if compute_metrics:
        r = agg['rows'] or 1
        metrics = dict(
            hit1_dc=agg['hit1_pre'] / r,
            hit5_dc=agg['hit5_pre'] / r,
            mrr_dc=agg['mrr_dc'] / r,
            hit1_carrier=agg['hit1_post'] / r,
            hit5_carrier=agg['hit5_post'] / r,
            mrr_carrier=agg.get('mrr_carrier', 0) / r,
            joint_hit1=agg.get('joint_hit1', 0) / r,
            joint_hit5=agg.get('joint_hit5', 0) / r,
            mrr_joint=agg.get('mrr_joint', 0) / r,
            hit1_dc_repaired=agg['hit1_pre_repaired'] / r,
            hit1_carrier_repaired=agg['hit1_post_repaired'] / r,
            joint_hit1_repaired=agg.get('joint_hit1_repaired', 0) / r,
            repair_changed_rate=agg['repair_changed'] / r,
            repair_over_inv_rate=agg['repair_over_inv'] / r,
            repair_over_inv_qty_frac=(agg['repair_over_inv_qty'] / agg['demand_total']) if agg['demand_total'] > 0 else 0.0,
            repair_raw_ineligible_rate=agg['repair_raw_ineligible'] / r,
            repair_raw_primary_oos_rate=agg['repair_raw_primary_oos'] / r,
            repair_drop_primary_rate=agg['repair_drop_primary'] / r,
            repair_carrier_changed_rate=agg['repair_carrier_changed'] / r,
            repair_raw_primary_ineligible_rate=agg['repair_raw_primary_ineligible'] / r,
            repair_primary_changed_rate=agg['repair_primary_changed'] / r,
            repair_split_rate=agg['repair_split'] / r,
            repair_unfulfilled_rate=agg['repair_unfulfilled'] / r,
            repair_unfulfilled_qty_frac=(agg['repair_unfulfilled_qty'] / agg['demand_total']) if agg['demand_total'] > 0 else 0.0,
            repair_moved_qty_frac=(agg['repair_moved_qty'] / agg['demand_total']) if agg['demand_total'] > 0 else 0.0,
        )
    else:
        metrics = None
    return loss, metrics

# ───────────────────────── main ───────────────────────── #
def main():
    args = parse_train_proxy_args()
    run_epoch._debug_nan = args.debug_nan
    run_epoch._debug_loss_stats = args.debug_loss_stats
    run_epoch._debug_batch_eligibility = args.debug_batch_eligibility

    seed_all()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")

    tensors,info = load_flat()
    if args.debug_eligibility and 'eligibility_mask' in tensors:
        with torch.no_grad():
            targets_pos = tensors['targets'] > 0
            elig_pos = tensors['eligibility_mask'] > 0
            mismatch = targets_pos & ~elig_pos
            mismatch_rows = mismatch.any(dim=1).sum().item()
            mismatch_entries = mismatch.sum().item()
            print(f"[eligibility-mismatch] rows={int(mismatch_rows)} entries={int(mismatch_entries)}")
            if mismatch_rows > 0:
                idxs = mismatch.any(dim=1).nonzero(as_tuple=True)[0][:5].tolist()
                metadata = info.get('metadata')
                if metadata and len(metadata) > max(idxs, default=-1):
                    examples = [metadata[i] for i in idxs]
                    print(f"[eligibility-mismatch] example_rows={examples}")
    # Keep scenarios in float16 for memory efficiency
    # Will convert to float32 during batch processing
    
    # Preserve raw base costs for expected-cost loss
    tensors['base_cost_raw'] = tensors['option_features'][..., 0].float()

    num_samples = int(tensors['global_features'].size(0))
    if num_samples <= 0:
        raise ValueError("Empty proxy training dataset.")

    # Optional refit mode: train on the full dataset after model/hparam selection.
    refit_full_data = bool(args.refit_full_data or args.split_ratio == 0.0)
    if refit_full_data:
        tr_indices = list(range(num_samples))
        va_indices = list(range(num_samples))
        print(
            "Refit mode enabled: training on full dataset "
            f"({num_samples} samples); no held-out validation split."
        )
    else:
        # Sequential split based on order creation time (most realistic for time-series)
        # Metadata should already be sorted by creation time from file loading order.
        val_len = int(num_samples * args.split_ratio)
        train_len = num_samples - val_len
        if train_len <= 0 or val_len <= 0:
            raise ValueError(
                f"Invalid split produced empty train/val set: "
                f"num_samples={num_samples}, split_ratio={args.split_ratio}, "
                f"train_len={train_len}, val_len={val_len}."
            )
        tr_indices = list(range(train_len))
        va_indices = list(range(train_len, num_samples))
        print(
            f"Sequential split: Train={len(tr_indices)} samples (earliest), "
            f"Val={len(va_indices)} samples (latest)"
        )

    tr_indices_np = np.asarray(tr_indices, dtype=np.int64)

    # Scale global, DC, and option features (independently configurable).
    # Fit scalers on the training subset to avoid validation leakage.
    global_scaler = None
    if args.normalize_global_features:
        global_scaler = StandardScaler()
        global_np = tensors['global_features'].numpy()
        global_scaler.fit(global_np[tr_indices_np])
        if hasattr(global_scaler, 'scale_'):
            global_scaler.scale_ = np.where(global_scaler.scale_ < 1e-8, 1.0, global_scaler.scale_)
        tensors['global_features'] = torch.tensor(
            global_scaler.transform(global_np), dtype=torch.float32
        )
    else:
        tensors['global_features'] = tensors['global_features'].float()

    dc_scaler = None
    dc_scale_indices = None
    if args.normalize_dc_features:
        dc_np_full = tensors['dc_features'].numpy()
        dc_feat_dim = dc_np_full.shape[-1]
        dc_scale_indices = list(range(dc_feat_dim))
        if dc_feat_dim >= 5:
            # Keep inventory_level (idx 0) and region_match (idx 2) in raw scale
            dc_scale_indices = [i for i in range(dc_feat_dim) if i not in (0, 2)]
        dc_scaler = StandardScaler()
        dc_np = dc_np_full.reshape(-1, dc_feat_dim)
        if dc_scale_indices:
            dc_train_np = dc_np_full[tr_indices_np].reshape(-1, dc_feat_dim)
            dc_scaler.fit(dc_train_np[:, dc_scale_indices])
            if hasattr(dc_scaler, 'scale_'):
                dc_scaler.scale_ = np.where(dc_scaler.scale_ < 1e-8, 1.0, dc_scaler.scale_)
            dc_np[:, dc_scale_indices] = dc_scaler.transform(dc_np[:, dc_scale_indices])
            dc_np_full = dc_np.reshape(dc_np_full.shape)
        tensors['dc_features'] = torch.tensor(dc_np_full, dtype=torch.float32)
    else:
        tensors['dc_features'] = tensors['dc_features'].float()

    option_scaler = None
    if args.normalize_option_features:
        option_scaler = StandardScaler()
        option_np_full = tensors['option_features'].numpy()
        option_feat_dim = option_np_full.shape[-1]
        option_train_np = option_np_full[tr_indices_np].reshape(-1, option_feat_dim)
        option_scaler.fit(option_train_np)
        if hasattr(option_scaler, 'scale_'):
            option_scaler.scale_ = np.where(option_scaler.scale_ < 1e-8, 1.0, option_scaler.scale_)
        option_np = option_np_full.reshape(-1, option_feat_dim)
        option_np = option_scaler.transform(option_np).reshape(option_np_full.shape)
        tensors['option_features'] = torch.tensor(option_np, dtype=torch.float32)
    else:
        tensors['option_features'] = tensors['option_features'].float()

    feature_scalers = {}
    if global_scaler is not None:
        feature_scalers['global'] = {'mean': global_scaler.mean_, 'scale': global_scaler.scale_}
    if dc_scaler is not None:
        feature_scalers['dc'] = {
            'mean': dc_scaler.mean_,
            'scale': dc_scaler.scale_,
            'indices': dc_scale_indices,
        }
    if option_scaler is not None:
        feature_scalers['option'] = {'mean': option_scaler.mean_, 'scale': option_scaler.scale_}

    ds = FlatDS(tensors)
    tr_ds = torch.utils.data.Subset(ds, tr_indices)
    va_ds = torch.utils.data.Subset(ds, va_indices)
    
    use_pin_memory = bool(args.pin_memory and device.type == 'cuda')
    dl_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=use_pin_memory,
    )
    if dl_kwargs['num_workers'] > 0:
        dl_kwargs['persistent_workers'] = bool(args.persistent_workers)
        dl_kwargs['prefetch_factor'] = max(1, int(args.prefetch_factor))

    tr_ld = DataLoader(tr_ds, shuffle=True, **dl_kwargs)
    va_ld = DataLoader(va_ds, shuffle=False, **dl_kwargs)
    non_blocking_transfer = bool(use_pin_memory)
    print(
        f"DataLoader: workers={dl_kwargs['num_workers']} "
        f"pin_memory={use_pin_memory} "
        f"persistent_workers={dl_kwargs.get('persistent_workers', False)} "
        f"prefetch_factor={dl_kwargs.get('prefetch_factor', 'n/a')}"
    )

    model_params = {
        'architecture': canonicalize_architecture(args.model_variant),
        'global_feature_dim': info['global_feature_dim'],
        'dc_feature_dim': info['dc_feature_dim'],
        'option_feature_dim': info['option_feature_dim'],
        'num_dcs': info['num_dcs'],
        'num_carriers': info['num_carriers'],
        'sku_dim': info['sku_dim'],
        'brand_dim': info['brand_dim'],
        'sku_emb_dim': args.sku_emb_dim,
        'brand_emb_dim': args.brand_emb_dim,
        'dc_embedding_dim': args.dc_embedding_dim,
        'carrier_embedding_dim': args.carrier_emb_dim,
        'option_proj_dim': args.option_proj_dim,
        'use_option_features_in_carrier': args.use_option_features_in_carrier,
        'use_scenario_module': args.use_scenario_module,
        'use_dc_module': args.use_dc_module,
        'use_dc_embedding': args.use_dc_embedding,
        'use_carrier_embedding': args.use_carrier_embedding,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'dropout_p': args.dropout_p,
        'agg_type': args.agg_type,
        'scenario_combine': args.scenario_combine,
        'use_num_proj': args.use_num_proj,
        'use_cost_summary': args.use_cost_summary,
        'repair_strategy': args.repair_strategy,
    }

    print("\n" + "="*70)
    print(f"PROXY ARCHITECTURE: {model_params['architecture']} (requested: {args.model_variant})")
    print("="*70)
    model = build_proxy_model(model_params).to(device)
    
    # Wrap model with DataParallel if multiple GPUs available
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        effective_batch_per_gpu = args.batch_size // num_gpus
        print(f"Model wrapped with DataParallel (using {num_gpus} GPUs)")
        print(f"Effective batch size per GPU: {effective_batch_per_gpu} (total batch: {args.batch_size})")
        if effective_batch_per_gpu < 1:
            print(f"WARNING: Batch size {args.batch_size} is smaller than number of GPUs {num_gpus}")
            print(f"Consider increasing --batch_size to at least {num_gpus}")
    
    # Mixed precision training (safe fallback)
    use_amp = torch.cuda.is_available()
    amp_scaler = _grad_scaler(enabled=use_amp, device_type='cuda')
    if amp_scaler is not None:
        print("Mixed precision training (AMP) enabled")
    
    print(f"Global features: {info['global_feature_dim']}")
    print(f"DC features: {info['dc_feature_dim']} per DC ({info['num_dcs']} DCs)")
    option_space = info['num_dcs'] * info['num_carriers']
    if option_space == 55 * 17:
        print(f"Option space: {info['num_dcs']} dcs x {info['num_carriers']} carriers = {option_space}")
    else:
        print(
            f"Option space: {info['num_dcs']} dcs x {info['num_carriers']} carriers = {option_space} "
            "(expected 55x17=935)"
        )
    print(f"Option features: {info['option_feature_dim']} deterministic per option ({option_space} options)")
    print(f"  + delivery_penalty scenarios: {info.get('scenario_len', 'N/A')} scenarios per option")
    print(f"Hidden dim: {args.hidden_dim}")
    print("="*70 + "\n")

    train_targets_for_weights = (
        tensors['targets']
        if refit_full_data
        else tensors['targets'][:len(tr_indices)]
    )

    # Compute inverse-frequency class weights (per option) on the training subset only.
    if args.disable_class_weights:
        class_weights = None
    else:
        with torch.no_grad():
            y_sum = train_targets_for_weights.sum(1)
            active = (y_sum > 0)
            y_idx = train_targets_for_weights.argmax(1)
            num_options = train_targets_for_weights.size(1)  # D*C (unified)
            counts = torch.bincount(y_idx[active], minlength=num_options).float().clamp_min(1.0)
            inv_freq = 1.0 / counts
            class_weights = inv_freq.pow(args.class_weight_power)
            if args.class_weight_max and args.class_weight_max > 0:
                class_weights = class_weights.clamp_max(args.class_weight_max)
            class_weights = class_weights / class_weights.mean()
            # class_weights is [D*C] (per option)

    dc_class_weights = None
    if args.use_dc_class_weights:
        with torch.no_grad():
            targets_grid = train_targets_for_weights.view(-1, info['num_dcs'], info['num_carriers'])
            qty_per_dc = targets_grid.sum(dim=2)
            y_dc = qty_per_dc.argmax(dim=1)
            active_dc = qty_per_dc.sum(dim=1) > 0
            dc_counts = torch.bincount(y_dc[active_dc], minlength=info['num_dcs']).float().clamp_min(1.0)
            dc_weights = (1.0 / dc_counts).pow(args.dc_class_weight_power)
            if args.dc_class_weight_max and args.dc_class_weight_max > 0:
                dc_weights = dc_weights.clamp_max(args.dc_class_weight_max)
            dc_class_weights = (dc_weights / dc_weights.mean()).float()

    carrier_class_weights = None
    if args.use_carrier_class_weights:
        with torch.no_grad():
            targets_grid = train_targets_for_weights.view(-1, info['num_dcs'], info['num_carriers'])
            qty_per_dc = targets_grid.sum(dim=2)
            y_dc = qty_per_dc.argmax(dim=1)
            active = qty_per_dc.sum(dim=1) > 0
            batch_indices = torch.arange(targets_grid.size(0))
            carrier_qty = targets_grid[batch_indices, y_dc, :]
            y_carrier = carrier_qty.argmax(dim=1)
            carrier_counts = torch.bincount(y_carrier[active], minlength=info['num_carriers']).float().clamp_min(1.0)
            carrier_weights = (1.0 / carrier_counts).pow(args.carrier_class_weight_power)
            if args.carrier_class_weight_max and args.carrier_class_weight_max > 0:
                carrier_weights = carrier_weights.clamp_max(args.carrier_class_weight_max)
            carrier_class_weights = (carrier_weights / carrier_weights.mean()).float()

    loss_fn=ProxyLoss(class_weights=class_weights,
                      selection_weight=args.selection_weight,
                      carrier_loss_weight=args.carrier_loss_weight,
                      constraint_weight=args.constraint_loss_weight,
                      cardinality_weight=args.cardinality_penalty_weight,
                      entropy_weight=args.entropy_weight,
                      gumbel_tau=args.gumbel_tau,
                      cost_loss_weight=args.cost_loss_weight,
                      label_smoothing=args.label_smoothing,
                      aux_dc_weight=args.aux_dc_weight,
                      aux_carrier_weight=args.aux_carrier_weight,
                      dc_class_weights=dc_class_weights,
                      carrier_class_weights=carrier_class_weights,
                      use_eligibility_mask=args.use_eligibility_mask)

    opt=optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched=optim.lr_scheduler.ReduceLROnPlateau(opt,'min',
           patience=cfg.PROXY_MODEL_LR_SCHEDULER_PATIENCE,
           factor=cfg.PROXY_MODEL_LR_SCHEDULER_FACTOR)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    model_dir_name = f"{args.model_name}_{timestamp}"
    out_dir=Path(cfg.PROXY_MODELS_DIR)/model_dir_name; out_dir.mkdir(parents=True,exist_ok=True)
    best_ckpt_path = out_dir / 'best.pt'
    print(f"\n{'='*70}")
    print(f"Model Name: {args.model_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Output Directory: {out_dir}")
    print(f"{'='*70}\n")

    # Save concise hyperparameters snapshot
    hparams = {
        'model_name': args.model_name,
        'run': {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'device': str(device)
        },
        'data': {
            'num_dcs': info['num_dcs'],
            'scenario_len': info['scenario_len'],
            'sku_dim': info['sku_dim'],
            'brand_dim': info['brand_dim']
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'split_ratio': args.split_ratio,
            'refit_full_data': refit_full_data,
            'early_stopping_patience': args.early_stopping_patience,
            'print_every': args.print_every,
            'final_eval_on_best': args.final_eval_on_best,
            'final_ub_eval_orders': args.final_ub_eval_orders,
            'final_ub_eval_scenarios': args.final_ub_eval_scenarios,
            'final_ub_eval_batch_size': args.final_ub_eval_batch_size,
        },
        'loss': {
            'selection_weight': args.selection_weight,
            'carrier_loss_weight': args.carrier_loss_weight,
            'constraint_loss_weight': args.constraint_loss_weight,
            'cardinality_penalty_weight': args.cardinality_penalty_weight,
            'entropy_weight': args.entropy_weight,
            'cost_loss_weight': args.cost_loss_weight,
            'threshold_on_sel': args.threshold_on_sel,
            'aux_dc_weight': args.aux_dc_weight,
            'aux_carrier_weight': args.aux_carrier_weight,
            'class_weight_power': args.class_weight_power,
            'class_weight_max': args.class_weight_max,
            'use_dc_class_weights': args.use_dc_class_weights,
            'dc_class_weight_power': args.dc_class_weight_power,
            'dc_class_weight_max': args.dc_class_weight_max,
            'use_eligibility_mask': args.use_eligibility_mask,
        },
        'inference': {
            'repair_strategy': args.repair_strategy
        },
        'architecture': {
            'model_variant': args.model_variant,
            'agg_type': args.agg_type,
            'scenario_combine': args.scenario_combine,
            'hidden_dim': args.hidden_dim,
            'n_layers': args.n_layers,
            'dropout_p': args.dropout_p,
            'sku_emb_dim': args.sku_emb_dim,
            'brand_emb_dim': args.brand_emb_dim,
            'dc_embedding_dim': args.dc_embedding_dim,
            'carrier_emb_dim': args.carrier_emb_dim,
            'option_proj_dim': args.option_proj_dim,
            'use_option_features_in_carrier': args.use_option_features_in_carrier,
            'use_cost_summary': args.use_cost_summary,
            'use_scenario_module': args.use_scenario_module,
            'use_dc_module': args.use_dc_module,
            'use_dc_embedding': args.use_dc_embedding,
            'use_carrier_embedding': args.use_carrier_embedding,
        }
    }
    with open(out_dir/'hyperparams.json', 'w') as f:
        json.dump(hparams, f, indent=2)
    best=float('inf')
    best_epoch = 0
    best_joint_hit1_repaired = float("-inf")
    best_joint_hit1_repaired_epoch = 0
    patience_counter = 0
    train_curve=[]; val_curve=[]
    all_metrics = []  # Store all epoch metrics for JSON
    for ep in range(1,args.epochs+1):
        epoch_start = time.perf_counter()
        should_compute_metrics = (ep % args.print_every == 1) or (ep == args.epochs)
        
        # Compute metrics on print epochs for efficiency
        tr_loss, m_tr = run_epoch(
            model, tr_ld, device, loss_fn, opt, amp_scaler, args.threshold_on_sel,
            compute_metrics=should_compute_metrics,
            num_dcs=info['num_dcs'],
            num_carriers=info['num_carriers'],
            repair_strategy=args.repair_strategy,
            non_blocking=non_blocking_transfer,
        )
        if refit_full_data:
            va_loss, m_va = tr_loss, None
            sched.step(tr_loss)
        else:
            va_loss, m_va = run_epoch(
                model, va_ld, device, loss_fn, None, None, args.threshold_on_sel,
                compute_metrics=should_compute_metrics,
                num_dcs=info['num_dcs'],
                num_carriers=info['num_carriers'],
                repair_strategy=args.repair_strategy,
                non_blocking=non_blocking_transfer,
            )
            sched.step(va_loss)
        epoch_time = time.perf_counter() - epoch_start
        lr_now = float(opt.param_groups[0]["lr"])
        
        # Print loss every epoch
        if refit_full_data:
            print(
                f'E{ep:02d} tr_loss:{tr_loss:.4f} va_loss:NA '
                f'lr:{lr_now:.2e} t:{epoch_time:.1f}s refit_full_data=1'
            )
        else:
            print(
                f'E{ep:02d} tr_loss:{tr_loss:.4f} va_loss:{va_loss:.4f} '
                f'lr:{lr_now:.2e} t:{epoch_time:.1f}s '
                f'patience:{patience_counter}/{args.early_stopping_patience}'
            )
        
        # Print detailed metrics on designated epochs
        if should_compute_metrics and m_tr is not None:
            ub_metrics = None

            if not refit_full_data and m_va is not None:
                print_train_val_metrics(m_tr, m_va)

                ub_every = max(1, int(args.ub_eval_every))
                ub_window = max(1, int(args.print_every)) * ub_every
                should_eval_ub = (
                    int(args.ub_eval_orders) > 0 and
                    (ep == args.epochs or (ep % ub_window == 1))
                )
                if should_eval_ub:
                    ub_metrics = compute_proxy_ub_metrics(
                        model=model,
                        tensors=tensors,
                        info=info,
                        metadata_rows=info.get("metadata", []) or [],
                        val_indices=va_indices,
                        device=device,
                        repair_strategy=args.repair_strategy,
                        csaa_root=Path(args.csaa_solutions_root),
                        max_orders=int(args.ub_eval_orders),
                        n2_eval=int(args.ub_eval_scenarios),
                        forward_batch_size=int(args.ub_eval_batch_size),
                    )
                    if ub_metrics:
                        print_ub_metrics(ub_metrics, ub_every)
                elif int(args.ub_eval_orders) > 0:
                    print_ub_metrics(ub_metrics, ub_every)
            
            # Store metrics for JSON
            epoch_metrics = {
                'epoch': ep,
                'train_loss': tr_loss,
                'train_metrics': m_tr,
            }
            if not refit_full_data and m_va is not None:
                epoch_metrics['val_loss'] = va_loss
                epoch_metrics['val_metrics'] = m_va
            if ub_metrics:
                epoch_metrics['ub_metrics'] = ub_metrics
            all_metrics.append(epoch_metrics)

            if not refit_full_data and m_va is not None:
                if m_va["joint_hit1_repaired"] > best_joint_hit1_repaired:
                    best_joint_hit1_repaired = m_va["joint_hit1_repaired"]
                    best_joint_hit1_repaired_epoch = ep
                best_loss_for_print = min(best, va_loss)
                print_best_so_far(best_loss_for_print, best_joint_hit1_repaired, best_joint_hit1_repaired_epoch)
            
        if refit_full_data:
            if ep == args.epochs:
                best = tr_loss
                best_epoch = ep
                model_params['threshold'] = args.threshold_on_sel
                model_params['carriers'] = info.get('carriers', [])
                torch.save({
                    'model': model.state_dict(),
                    'feature_scalers': feature_scalers,
                    'info': info,
                    'model_params': model_params,
                    'hyperparams': hparams
                }, best_ckpt_path)
        else:
            if va_loss < best:
                best = va_loss
                best_epoch = ep
                patience_counter = 0
                model_params['threshold'] = args.threshold_on_sel
                model_params['carriers'] = info.get('carriers', [])
                torch.save({
                    'model': model.state_dict(),
                    'feature_scalers': feature_scalers,
                    'info': info,
                    'model_params': model_params,
                    'hyperparams': hparams
                }, best_ckpt_path)
            else:
                patience_counter += 1
            
        if (not refit_full_data) and patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {ep} (no improvement for {args.early_stopping_patience} epochs)")
            break
            
        train_curve.append(tr_loss); val_curve.append(va_loss)

    # Save metrics to JSON
    with open(out_dir / 'training_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Optional final validation pass on best checkpoint (for robust model selection).
    if refit_full_data and args.final_eval_on_best:
        print("Refit mode: skipping final held-out evaluation (no held-out split).")
    if (not refit_full_data) and args.final_eval_on_best and best_ckpt_path.exists():
        print("\n" + "=" * 70)
        print("Final Validation on Best Checkpoint")
        print("=" * 70)
        print(f"Best checkpoint: {best_ckpt_path}")
        print(f"Best epoch (by val loss): E{best_epoch:02d} | best_va_loss={best:.4f}")
        try:
            best_blob = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_blob['model'])

            final_va_loss, final_va_metrics = run_epoch(
                model, va_ld, device, loss_fn, None, None, args.threshold_on_sel,
                compute_metrics=True,
                num_dcs=info['num_dcs'],
                num_carriers=info['num_carriers'],
                repair_strategy=args.repair_strategy,
                non_blocking=non_blocking_transfer,
            )
            print(f"Final Best Eval: va_loss:{final_va_loss:.4f}")
            print_val_metrics(final_va_metrics)

            final_ub = None
            final_ub_orders = int(args.final_ub_eval_orders)
            if final_ub_orders > 0:
                final_n2 = int(args.final_ub_eval_scenarios) if int(args.final_ub_eval_scenarios) > 0 else int(args.ub_eval_scenarios)
                final_ub = compute_proxy_ub_metrics(
                    model=model,
                    tensors=tensors,
                    info=info,
                    metadata_rows=info.get("metadata", []) or [],
                    val_indices=va_indices,
                    device=device,
                    repair_strategy=args.repair_strategy,
                    csaa_root=Path(args.csaa_solutions_root),
                    max_orders=final_ub_orders,
                    n2_eval=final_n2,
                    forward_batch_size=int(args.final_ub_eval_batch_size),
                )
                print_ub_metrics(final_ub, ub_every=1)

            final_report = {
                'model_name': args.model_name,
                'model_dir_name': model_dir_name,
                'best_checkpoint': str(best_ckpt_path),
                'best_epoch_by_val_loss': int(best_epoch),
                'best_val_loss_during_training': float(best),
                'final_best_eval_val_loss': float(final_va_loss),
                'final_best_eval_val_metrics': final_va_metrics,
                'final_best_eval_ub_metrics': final_ub,
            }
            with open(out_dir / 'final_best_eval.json', 'w') as f:
                json.dump(final_report, f, indent=2)
            print(f"Final best-eval report saved to {out_dir / 'final_best_eval.json'}")
        except Exception as exc:
            print(f"Warning: final best-checkpoint evaluation failed: {exc}")

    plot_dir=Path(cfg.PROXY_PLOTS_DIR)/args.model_name; plot_dir.mkdir(parents=True,exist_ok=True)
    # ------- LOSS CURVE ------------------------------------------- #
    plt.figure(figsize=(10,4))
    plt.plot(train_curve, label='Train')
    plt.plot(val_curve,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss curve')
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_dir/'loss_curve.png', dpi=300)
    plt.close()
    print(f"Loss curve saved to {plot_dir/'loss_curve.png'}")

    # Publish a stable alias at the documented, non-timestamped path so the README
    # and scripts/reproduce/ wrappers (data/models/proxy/<model_name>/best.pt)
    # resolve to the latest run of this model_name.
    if best_ckpt_path.exists():
        alias_dir = Path(cfg.PROXY_MODELS_DIR) / args.model_name
        alias_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt_path, alias_dir / 'best.pt')
        print(f"Published stable alias: {alias_dir/'best.pt'}  <-  {best_ckpt_path}")

    print("\n=== Training complete ===")
    
if __name__=="__main__":
    main()
