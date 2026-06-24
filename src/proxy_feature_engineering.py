import pandas as pd
import numpy as np
import torch
import json
import pickle
import argparse
from multiprocessing import Pool, cpu_count
from typing import Optional

import src.config as cfg
from src.data_utils import (
    load_data,
    preprocess_proxy_features, load_dc_carrier_metadata,
    compute_base_costs_for_order,
    compute_global_features_for_order,
    compute_dc_features_for_order,
    get_observed_dcs_from_preprocessed,
)
from src.simulator.precompute import PrecomputeStore
from src.simulator.catalog import OptionsCatalog

# ---------------------------- helper ---------------------------- #
def build_target(fp_df: pd.DataFrame, sku_id: str, dcs: list[str], carriers: list[int]) -> np.ndarray:
    """
    Build target tensor from fulfillment plan.
    
    Returns:
        (num_dcs, num_carriers) array with quantities allocated to each (DC, Carrier) option.
    """
    target_grid = np.zeros((len(dcs), len(carriers)), dtype=np.float32)
    
    if fp_df is not None:
        sub = fp_df[fp_df['sku_ID'] == sku_id]
        for _, row in sub.iterrows():
            try:
                dc_idx = dcs.index(row['dc_ori'])
                carrier_val = int(row['carrier_service_id'])
                if carrier_val in carriers:
                    c_idx = carriers.index(carrier_val)
                    target_grid[dc_idx, c_idx] += row['quantity']
            except ValueError:
                continue # DC or Carrier not in our universe
                
    return target_grid


# compute_base_costs moved to data_utils.py as compute_base_costs_for_order


def load_delivery_scenarios(order_dir) -> Optional[np.ndarray]:
    """
    Attempt to load raw delivery time scenarios if they were stored during CSAA collection.
    
    Args:
        order_dir: Path to order directory
        
    Returns:
        delivery_time_scenarios: (NumSKUs, S, NumOptions) array of delivery days, or None if not available
    """
    # Check for raw delivery time scenarios (future enhancement)
    delivery_path = order_dir / "delivery_time_scenarios.npz"
    if delivery_path.exists():
        try:
            data = np.load(delivery_path)
            # Expected format: delivery_time_{sku} for each SKU
            delivery_keys = sorted([k for k in data.files if k.startswith('delivery_time_')])
            if delivery_keys:
                return np.stack([data[k] for k in delivery_keys], axis=0)
        except Exception:
            pass
    return None


def disentangle_cost_scenarios(
    cost_scenarios: np.ndarray,
    base_costs: np.ndarray
) -> np.ndarray:
    """
    Separate stochastic delivery penalty from total cost scenarios.
    
    Args:
        cost_scenarios: (S, D, C) array of total costs (base + delivery penalty)
        base_costs: (D, C) array of deterministic base shipping costs
        
    Returns:
        delivery_penalties: (S, D, C) array of stochastic delivery time penalties
    """
    # Subtract base costs to isolate delivery penalty component
    # cost_scenarios = base_cost + (late_days * penalty_per_day)
    return cost_scenarios - base_costs[np.newaxis, :, :]


# Feature engineering functions moved to data_utils.py:
# - compute_base_costs_for_order
# - compute_global_features_for_order
# - compute_dc_features_for_order


# ----------------------- main per‑date function ----------------------- #
def engineer_proxy_features(
    date: str,
    split: str,
    peak_only: bool = False,
    scenario_subset: str = "all",
    use_float16: bool = True,
    use_precomputed_base_cost: bool = False,
    debug_eligibility: bool = False,
):
    print(f"[{split}]  {date}")
    if peak_only:
        root = cfg.DATA_DIR / "peak" / "csaa_solutions" / split / date
    else:
        root = cfg.DATA_DIR / "csaa_solutions" / split / date
    if not root.is_dir():
        print("  dir missing → skip"); return

    out_dir = (cfg.PROXY_TRAINING_DATA_DIR if split == "proxy_train"
               else cfg.PROXY_TEST_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- shared data -------- #
    orders_df, network_df, sku_df = load_data(split)
    
    # Load DC-Region mapping
    dc_region_map = network_df.set_index('dc_ID')['region_ID'].to_dict()

    pre = pd.read_csv(cfg.PREPROCESSED_PATH)
    pre['order_time'] = pd.to_datetime(pre['order_time'])
    pre = preprocess_proxy_features(pre)
    
    # Load precomputed base cost tensor if requested
    base_cost_precomputed = None
    node_to_idx = None
    precomp_dc_index = None  # maps current all_dcs to indices in precomputed tensor
    precomp_carriers = None
    if use_precomputed_base_cost:
        precomp_path = cfg.PROXY_DATA_DIR / "precomputed_base_cost_tensor.pt"
        if precomp_path.exists():
            print(f"  Loading precomputed base costs from {precomp_path}")
            # Safe load: file contains only tensors/arrays, no code objects
            precomp = torch.load(precomp_path, map_location='cpu', weights_only=False)
            base_cost_precomputed = precomp['base_cost_tensor'].numpy()  # [N_nodes, D_pre, C]
            precomp_nodes = precomp['nodes']
            precomp_dcs = [int(dc) for dc in precomp['dcs']]
            precomp_carriers = [int(c) for c in precomp.get('carriers', [])]
            node_to_idx = {node: i for i, node in enumerate(precomp_nodes)}
            precomp_dc_to_idx = {dc: i for i, dc in enumerate(precomp_dcs)}
            print(f"    Loaded tensor shape: {base_cost_precomputed.shape}")
        else:
            print(f"  WARNING: Precomputed base costs not found at {precomp_path}")
            print(f"           Run: python -m scripts.precompute.precompute_base_cost_tensor")
            print(f"           Falling back to on-the-fly computation")
            use_precomputed_base_cost = False

    # Load carrier metadata to define the universe of carriers and cost coefficients
    dc_metadata_full, cost_models = load_dc_carrier_metadata()
    # Align proxy universe with simulation/CSAA by using the options catalog.
    precompute_store = PrecomputeStore()
    catalog = OptionsCatalog(precompute_store=precompute_store)
    catalog_dcs = sorted({int(opt.dc_id) for opt in catalog.all_options})
    catalog_carriers = sorted({int(opt.carrier_service_id) for opt in catalog.all_options})
    all_carriers = catalog_carriers
    num_carriers = len(all_carriers)

    observed_dcs = get_observed_dcs_from_preprocessed()
    if observed_dcs:
        filtered_dcs = [dc for dc in catalog_dcs if dc in observed_dcs]
        dropped = [dc for dc in catalog_dcs if dc not in observed_dcs]
        if dropped:
            print(
                f"  DC universe filtered to observed preprocessed DCs: "
                f"{len(filtered_dcs)}/{len(catalog_dcs)} (dropped {dropped})"
            )
        catalog_dcs = filtered_dcs
    
    # Prepare DC metadata for cost computation (dc_id, lat, lon)
    dc_metadata = dc_metadata_full[['dc_id', 'lat', 'lon']].drop_duplicates().dropna()
    
    # Compute SKU average daily demand using ONLY data prior to current date (no leakage)
    current_date_dt = pd.to_datetime(date)
    historical_data = pre[pre['order_time'] < current_date_dt].copy()
    
    if not historical_data.empty:
        sku_daily_demand = historical_data.groupby(cfg.SKU_COL)['quantity'].sum()
        date_range_days = (historical_data['order_time'].max() - historical_data['order_time'].min()).days
        if date_range_days > 0:
            sku_daily_demand = sku_daily_demand / date_range_days
        else:
            sku_daily_demand = sku_daily_demand * 0  # Fallback: all zero
        sku_daily_demand_dict = sku_daily_demand.to_dict()
    else:
        sku_daily_demand_dict = {}
    
    # Compute global order volume per hour (for peak detection)
    if not historical_data.empty:
        historical_data['order_hour'] = historical_data['order_time'].dt.hour
        hourly_volume = historical_data.groupby('order_hour').size().to_dict()
    else:
        hourly_volume = {}

    # categorical factorisation
    for col in cfg.PROXY_CATEGORICAL_ORDER_FEATURES:
        pre[col], _ = pd.factorize(pre[col], sort=True)
    pre["sku_idx"],   _ = pd.factorize(pre[cfg.SKU_COL],   sort=True)
    pre["brand_idx"], _ = pd.factorize(pre[cfg.BRAND_COL], sort=True)

    all_dcs = catalog_dcs
    # If we have precomputed base costs, align DC dimension/order with the precomputed tensor
    if base_cost_precomputed is not None:
        try:
            precomp_dc_index = [precomp_dc_to_idx[int(dc)] for dc in all_dcs]
        except KeyError as e:
            missing_dc = int(getattr(e, 'args', ['?'])[0])
            print(f"  WARNING: DC {missing_dc} not found in precomputed base costs; "
                  f"falling back to on-the-fly base cost computation.")
            base_cost_precomputed = None
            node_to_idx = None
            precomp_dc_index = None
            use_precomputed_base_cost = False
        else:
            if precomp_carriers is not None:
                precomp_carriers_int = [int(c) for c in precomp_carriers]
                if precomp_carriers_int != all_carriers:
                    print(
                        "  WARNING: Precomputed base cost carriers do not match the options catalog; "
                        "falling back to on-the-fly base cost computation."
                    )
                    base_cost_precomputed = None
                    node_to_idx = None
                    precomp_dc_index = None
                    use_precomputed_base_cost = False

    num_dcs   = len(all_dcs)
    option_space = num_dcs * num_carriers
    expected_space = len(observed_dcs) * num_carriers if observed_dcs else None
    if expected_space and option_space == expected_space:
        print(f"  option space: {num_dcs} dcs x {num_carriers} carriers = {option_space}")
    else:
        expected_note = f"{len(observed_dcs)}x{num_carriers}={expected_space}" if expected_space else "55x17=935"
        print(
            f"  option space: {num_dcs} dcs x {num_carriers} carriers = {option_space} "
            f"(expected {expected_note})"
        )
    sku_dim   = pre["sku_idx"].nunique()
    brand_dim = pre["brand_idx"].nunique()
    
    # Create DC metadata lookup for distance computation
    dc_meta_dict = {}
    for _, row in dc_metadata.iterrows():
        dc_id = int(row['dc_id'])
        dc_meta_dict[dc_id] = {
            'lat': float(row['lat']),
            'lon': float(row['lon']),
        }
    
    # Scenario alignment with SAA stages:
    # - Stage 1 (Candidate Generation): Uses first N1*Q scenarios, divided into Q candidates of N1 each
    # - Stage 2 (Evaluation): Uses next N2 scenarios to evaluate candidates
    # "candidate_only" trains on Stage 1 scenarios only, reserving N2 for evaluation/bounds
    num_candidate_scenarios = cfg.SAA_N1 * cfg.SAA_Q  # Stage 1: candidate generation
    num_evaluation_scenarios = cfg.SAA_N2              # Stage 2: candidate evaluation
    
    expected_scenarios = num_candidate_scenarios
    if scenario_subset == "all":
        expected_scenarios += num_evaluation_scenarios

    # -------- storages -------- #
    # Structure-aware architecture only: separate global, DC, and option features
    global_feats_list, dc_feats_list, option_feats_list = [], [], []
    dem, tgt = [], []
    qtys, sku_i, brand_i, meta_rows = [], [], [], []
    eligibility_masks = []  # Track eligible (DC, Carrier) options
    delivery_penalties_list = []  # Store stochastic delivery penalties [S, D, C]
    mismatch_rows = 0
    mismatch_entries = 0
    mismatch_examples = 0
    max_mismatch_examples = 5
    orders_total = 0
    orders_skipped_exception = 0
    orders_missing_preprocessed = 0
    orders_with_rows = 0
    skus_total = 0
    skus_skipped_shape = 0
    rows_added = 0
    error_examples = []

    # -------- iterate orders -------- #
    for o_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        orders_total += 1
        try:
            scen = np.load(o_dir / "candidate_scenarios.npz")

            # Load per-SKU demand scenarios (format: demand_{sku})
            demand_keys = sorted([k for k in scen.files if k.startswith('demand_')])
            if not demand_keys:
                raise ValueError("No demand scenario data found")
            demand_arr = np.stack([scen[k] for k in demand_keys], axis=0)  # (NumSKUs, S)

            # Prefer separated base costs + delivery penalties if present
            base_costs_path = o_dir / 'base_costs.npy'
            penalties_path = o_dir / 'delivery_penalties.npz'
            if base_costs_path.exists() and penalties_path.exists():
                base_costs = np.load(base_costs_path)
                penalties_npz = np.load(penalties_path)
                penalties = penalties_npz['penalty']  # (S, NumOptions)
                cost_matrix = penalties.astype(np.float32) + base_costs.astype(np.float32).reshape(1, -1)
                cost_arr = np.repeat(cost_matrix[None, ...], demand_arr.shape[0], axis=0)
            else:
                # Fallback: load per-SKU shipping scenario data (format: shipping_{sku})
                shipping_keys = sorted([k for k in scen.files if k.startswith('shipping_')])
                if not shipping_keys:
                    raise ValueError("No shipping scenario data found; expected base_costs.npy + delivery_penalties.npz or shipping_* keys")
                cost_arr = np.stack([scen[k] for k in shipping_keys], axis=0)  # (NumSKUs, S, NumOptions)

            # Optionally restrict to Stage 1 (candidate generation) scenarios only
            # This matches what SAA uses in run_saa_procedure's candidate generation phase
            if scenario_subset == "candidate_only":
                num_candidate_scenarios = cfg.SAA_N1 * cfg.SAA_Q
                demand_arr = demand_arr[:, :num_candidate_scenarios]
                cost_arr = cost_arr[:, :num_candidate_scenarios, ...]
            
            # Read order data (prefer parquet, fallback to CSV for backward compatibility)
            items_pq = o_dir/'order_items.parquet'
            items = pd.read_parquet(items_pq) if items_pq.exists() else pd.read_csv(o_dir/'order_items.csv')
            
            inv_pq = o_dir/'on_hand_inventory_pivot.parquet'
            inv_piv = pd.read_parquet(inv_pq) if inv_pq.exists() else pd.read_csv(o_dir/'on_hand_inventory_pivot.csv', index_col=0)
            inv_piv.columns = inv_piv.columns.astype(int) 
            inv_piv = inv_piv.reindex(columns=all_dcs).fillna(0)
            
            fp_pq = o_dir/'fulfillment_plan.parquet'
            fp_path = o_dir/'fulfillment_plan.csv'
            if fp_pq.exists():
                fp_df = pd.read_parquet(fp_pq)
            elif fp_path.exists():
                fp_df = pd.read_csv(fp_path)
            else:
                fp_df = None
            
            # Load option metadata: option_index -> (dc_id, carrier_id)
            opt_map = {}
            metadata_pkl = o_dir / "metadata.pkl"
            metadata_json = o_dir / "metadata.json"
            dynamic_path = o_dir / "dynamic_features.parquet"
            
            if metadata_pkl.exists():
                try:
                    with open(metadata_pkl, 'rb') as f:
                        meta = pickle.load(f)
                    option_snapshot = meta.get('option_snapshot', [])
                    opt_map = {i: (o['dc_id'], o['carrier_service_id']) for i, o in enumerate(option_snapshot)}
                except Exception:
                    pass
            elif metadata_json.exists():
                try:
                    with open(metadata_json) as f:
                        meta = json.load(f)
                    option_snapshot = meta.get('option_snapshot', [])
                    opt_map = {i: (o['dc_id'], o['carrier_service_id']) for i, o in enumerate(option_snapshot)}
                except Exception:
                    pass
            
        except Exception as e:
            orders_skipped_exception += 1
            if len(error_examples) < 5:
                error_examples.append((o_dir.name, str(e)))
            print("  skip order", o_dir.name, "err:", e); continue

        dynamic_df = None
        if dynamic_path.exists():
            try:
                dynamic_df = pd.read_parquet(dynamic_path)
            except Exception:
                dynamic_df = None

        order_id    = o_dir.name
        cust_dc     = items['dc_des'].iloc[0] if 'dc_des' in items.columns else None
        skus        = items['sku_ID'].unique().tolist()
            
        order_pre = pre[pre['order_ID'] == order_id]
        if order_pre.empty:
            orders_missing_preprocessed += 1
            continue
        order_feat_vec = order_pre[cfg.PROXY_ORDER_FEATURES].iloc[0].to_dict()
        inv_mat = inv_piv.loc[skus].values  # (NumSKUs, NumDCs)
        qty_vec = items.set_index('sku_ID')['quantity'].loc[skus].values
        sku_idx_v = order_pre['sku_idx'].values
        brand_idx_v = order_pre['brand_idx'].values
        
        # Compute deterministic base costs for this order
        customer_lat = order_pre['customer_lat'].iloc[0] if 'customer_lat' in order_pre.columns else 0.0
        customer_lon = order_pre['customer_lon'].iloc[0] if 'customer_lon' in order_pre.columns else 0.0
        
        if use_precomputed_base_cost and base_cost_precomputed is not None and precomp_dc_index is not None:
            # Fast path: lookup precomputed base costs using dc_des as node id,
            # then align DC dimension/order with current all_dcs via precomp_dc_index.
            customer_dc = order_pre['dc_des'].iloc[0] if 'dc_des' in order_pre.columns else None
            if customer_dc is not None and customer_dc in node_to_idx:
                node_idx = node_to_idx[customer_dc]
                full_base_cost_grid = base_cost_precomputed[node_idx]  # [D_pre, C]
                base_cost_grid = full_base_cost_grid[precomp_dc_index, :]  # [D, C]
            else:
                # Fallback: compute on the fly
                base_cost_grid = compute_base_costs_for_order(
                    customer_lat, customer_lon,
                    all_dcs, all_carriers,
                    dc_metadata, cost_models
                )
        else:
            # Slow path: compute on the fly
            base_cost_grid = compute_base_costs_for_order(
                customer_lat, customer_lon,
                all_dcs, all_carriers,
                dc_metadata, cost_models
            )
        
        # Check if raw delivery time scenarios are available (future enhancement)
        raw_delivery_scenarios = load_delivery_scenarios(o_dir)
        
        # Compute region match features (binary vector: 1 if same region as customer)
        cust_region = dc_region_map.get(cust_dc, None) if cust_dc is not None else None
        region_match_vec = np.zeros(num_dcs, dtype=np.float32)
        if cust_region is not None:
            for dc_idx, dc_id in enumerate(all_dcs):
                if dc_region_map.get(dc_id, None) == cust_region:
                    region_match_vec[dc_idx] = 1.0
        
        # Compute basket consolidation features (count of other items at each DC)
        # For current SKU i, count how many OTHER SKUs in the order are in stock at each DC
        inv_binary = (inv_mat > 0).astype(np.float32)  # (NumSKUs, NumDCs)
        total_items_at_dc = inv_binary.sum(axis=0)  # (NumDCs,)
        
        # Get global order volume for this hour (peak detection)
        order_hour = order_pre['order_hour'].iloc[0] if 'order_hour' in order_pre.columns else 0
        global_volume = hourly_volume.get(order_hour, 0)
        if dynamic_df is not None and not dynamic_df.empty and 'dc_id' in dynamic_df.columns:
            dyn_df = dynamic_df.copy()
            dyn_df['dc_id'] = pd.to_numeric(dyn_df['dc_id'], errors='coerce').astype('Int64')
            dyn_df = dyn_df.dropna(subset=['dc_id']).drop_duplicates('dc_id')
            dyn_df = dyn_df.set_index('dc_id')
            dyn_cols = []
            for feat in cfg.DYNAMIC_FEATURES:
                if feat in dyn_df.columns:
                    vals = dyn_df[feat].reindex(all_dcs).fillna(0.0).astype(float).to_numpy()
                else:
                    vals = np.zeros(num_dcs, dtype=np.float32)
                dyn_cols.append(vals)
            dynamic_dc_feats = np.stack(dyn_cols, axis=1).astype(np.float32)
        else:
            dynamic_dc_feats = np.zeros((num_dcs, len(cfg.DYNAMIC_FEATURES)), dtype=np.float32)

        rows_before = rows_added
        for i, sku in enumerate(skus):
            skus_total += 1
            actual_S = len(demand_arr[i])
            if actual_S == 0:
                print(f"  drop {order_id}/{sku}: no scenarios")
                continue

            inv_vec = inv_mat[i]
            
            # Basket consolidation: count OTHER items available at each DC
            consolidation_potential = total_items_at_dc - inv_binary[i]  # (NumDCs,)
            
            # Days of Supply calculation
            avg_demand = sku_daily_demand_dict.get(sku, 0.0)
            dos_vec = inv_vec / (avg_demand + 1e-6)  # Add epsilon to avoid division by zero
            
            # Restructure cost array: map from option-based to (DC, Carrier) grid
            current_cost_S_Opts = cost_arr[i]  # (S, NumOptions)
            S_dim = current_cost_S_Opts.shape[0]
            
            # Normalize cost array shape
            if current_cost_S_Opts.ndim == 3:
                current_cost_S_Opts = current_cost_S_Opts.mean(axis=1)  # Aggregate middle dimension
            
            if current_cost_S_Opts.ndim != 2:
                # Unknown shape, skip this sample
                print(f"  skip {order_id}/{sku}: unexpected cost shape {current_cost_S_Opts.shape}")
                skus_skipped_shape += 1
                continue
            
            num_options = current_cost_S_Opts.shape[1]
            
            # ========== STRUCTURE-AWARE FEATURES ==========
            # Global features
            global_feats = compute_global_features_for_order(
                order_feat_vec=order_feat_vec,
                sku_quantity=qty_vec[i],
                global_order_volume=global_volume,
                base_cost_grid=base_cost_grid,
            )
            
            # DC features
            dc_features = compute_dc_features_for_order(
                inv_vec=inv_vec,
                sku_daily_demand=avg_demand,
                region_match_vec=region_match_vec,
                consolidation_potential=consolidation_potential,
                customer_lat=customer_lat,
                customer_lon=customer_lon,
                dc_meta_dict=dc_meta_dict,
                all_dcs=all_dcs,
            )
            if dynamic_dc_feats.size:
                dc_features = np.concatenate([dc_features, dynamic_dc_feats], axis=1)
            
            global_feats_list.append(list(global_feats.values()))
            dc_feats_list.append(dc_features)
            dem.append(demand_arr[i])  # (S,)
            
            # Map option-based costs to (DC, Carrier) grid
            dense_cost = np.zeros((S_dim, num_dcs, num_carriers), dtype=np.float32)
            eligibility_mask = np.zeros((num_dcs, num_carriers), dtype=np.float32)
            
            if opt_map:
                for opt_idx, (d_id, c_id) in opt_map.items():
                    if opt_idx < num_options and d_id in all_dcs and c_id in all_carriers:
                        d_idx = all_dcs.index(d_id)
                        c_idx = all_carriers.index(c_id)
                        dense_cost[:, d_idx, c_idx] = current_cost_S_Opts[:, int(opt_idx)]
                        eligibility_mask[d_idx, c_idx] = 1.0
            else:
                # Fallback: assume uniform costs across carriers
                if current_cost_S_Opts.shape[-1] == num_dcs:
                    for c_idx in range(num_carriers):
                        dense_cost[:, :, c_idx] = current_cost_S_Opts[:, :num_dcs]
                    eligibility_mask[:, :] = 1.0
            
            # Build target before logging/append
            target_grid = build_target(fp_df, sku, all_dcs, all_carriers)

            if debug_eligibility:
                mismatch = (target_grid > 0) & (eligibility_mask <= 0)
                if mismatch.any():
                    mismatch_rows += 1
                    mismatch_entries += int(mismatch.sum())
                    if mismatch_examples < max_mismatch_examples:
                        idxs = np.argwhere(mismatch)
                        example_pairs = [
                            (all_dcs[i], all_carriers[j]) for i, j in idxs[:5]
                        ]
                        print(
                            f"[eligibility-mismatch] order={order_id} sku={sku} "
                            f"mismatches={int(mismatch.sum())} examples={example_pairs}"
                        )
                        mismatch_examples += 1

            eligibility_masks.append(eligibility_mask.flatten())
            
            # Disentangle cost into base_cost and delivery penalty
            # If raw delivery scenarios are available, use them directly
            # Otherwise, reconstruct by subtracting base costs
            if raw_delivery_scenarios is not None and i < len(raw_delivery_scenarios):
                # Use raw delivery time scenarios (in days)
                # Note: This assumes delivery penalties are already separated
                delivery_penalty = raw_delivery_scenarios[i]  # (S, NumOptions)
                # Map to dense grid
                delivery_penalty_dense = np.zeros((S_dim, num_dcs, num_carriers), dtype=np.float32)
                if opt_map:
                    for opt_idx, (d_id, c_id) in opt_map.items():
                        if opt_idx < delivery_penalty.shape[-1] and d_id in all_dcs and c_id in all_carriers:
                            d_idx = all_dcs.index(d_id)
                            c_idx = all_carriers.index(c_id)
                            delivery_penalty_dense[:, d_idx, c_idx] = delivery_penalty[:, int(opt_idx)]
            else:
                # Reconstruct delivery penalty by subtracting base costs
                delivery_penalty_dense = disentangle_cost_scenarios(
                    cost_scenarios=dense_cost,
                    base_costs=base_cost_grid
                )
            
            # Option features [D, C, K_opt]
            penalty_mean = delivery_penalty_dense.mean(axis=0)
            penalty_std = delivery_penalty_dense.std(axis=0)
            penalty_p90 = np.quantile(delivery_penalty_dense, 0.9, axis=0)
            option_features = np.stack(
                [base_cost_grid, penalty_mean, penalty_std, penalty_p90],
                axis=-1
            ).astype(np.float32)

            option_feats_list.append(option_features)
            delivery_penalties_list.append(delivery_penalty_dense)
            tgt.append(target_grid.flatten())
            qtys.append([qty_vec[i]])
            sku_i.append([int(sku_idx_v[i])])
            brand_i.append([int(brand_idx_v[i])])
            rows_added += 1
            meta_rows.append({'order_id': order_id, 'sku_id': sku})

        if rows_added > rows_before:
            orders_with_rows += 1

    if not global_feats_list:
        print("  no valid rows -> skip save")
        print(
            f"  summary: orders_total={orders_total} "
            f"orders_skipped_exception={orders_skipped_exception} "
            f"orders_missing_preprocessed={orders_missing_preprocessed} "
            f"orders_with_rows={orders_with_rows} "
            f"skus_total={skus_total} "
            f"skus_skipped_shape={skus_skipped_shape} "
            f"rows_added={rows_added}"
        )
        if error_examples:
            print("  sample_errors:")
            for order_id, err in error_examples:
                print(f"    - {order_id}: {err}")
        return

    global_feature_names = list(global_feats.keys())
    dc_feature_names = ['inventory_level', 'days_of_supply', 'region_match', 'consolidation_potential', 'distance_km'] + cfg.DYNAMIC_FEATURES
    option_feature_names = ['base_cost', 'delivery_penalty_mean', 'delivery_penalty_std', 'delivery_penalty_p90']
    
    # Use float16 for all float tensors to reduce storage by 50%
    # Model will cast to float32 during forward pass for numerical stability
    tensor_dtype = torch.float16 if use_float16 else torch.float32
    
    # Structure-Aware Architecture Tensors:
    #   global_features: [N, F_global] - order-level features
    #   dc_features: [N, D, K_dc] - per-DC features (inventory, DOS, region match, consolidation, distance)
    #   option_features: [N, D, C, K_opt] - per-option features (base_cost)
    #   scenario_demand: [N, S] - demand scenarios (processed by internal branch)
    #   delivery_penalty: [N, S, D, C] - delivery penalty scenarios (processed by internal branch)
    #   targets: [N, D*C] - target allocations
    #   eligibility_mask: [N, D*C] - eligible options
    
    dataset = {
        # ========== STRUCTURE-AWARE FEATURES ==========
        'global_features': torch.tensor(np.asarray(global_feats_list), dtype=tensor_dtype),  # [N, F_global]
        'dc_features': torch.tensor(np.asarray(dc_feats_list), dtype=tensor_dtype),  # [N, D, K_dc]
        'option_features': torch.tensor(np.asarray(option_feats_list), dtype=tensor_dtype),  # [N, D, C, K_opt]
        'scenario_demand': torch.tensor(np.asarray(dem), dtype=tensor_dtype),  # [N, S]
        'delivery_penalty': torch.tensor(np.asarray(delivery_penalties_list), dtype=tensor_dtype),  # [N, S, D, C]
        
        # ========== SHARED METADATA ==========
        'targets': torch.tensor(np.asarray(tgt), dtype=tensor_dtype),  # [N, D*C]
        'eligibility_mask': torch.tensor(np.asarray(eligibility_masks), dtype=tensor_dtype),  # [N, D*C]
        'quantity_vector': torch.tensor(np.asarray(qtys), dtype=tensor_dtype),  # [N, 1]
        'sku_idx': torch.tensor(np.asarray(sku_i), dtype=torch.long),  # [N, 1]
        'brand_idx': torch.tensor(np.asarray(brand_i), dtype=torch.long),  # [N, 1]
        
        # Feature metadata
        'global_feature_names': global_feature_names,
        'dc_feature_names': dc_feature_names,
        'option_feature_names': option_feature_names,
        
        # Dimensions
        'num_dcs': num_dcs,
        'num_carriers': num_carriers,
        'scenario_len': dem[0].shape[0] if len(dem) > 0 else expected_scenarios,
        'sku_dim': sku_dim,
        'brand_dim': brand_dim,
        'global_feature_dim': len(global_feature_names),
        'dc_feature_dim': len(dc_feature_names),
        'option_feature_dim': len(option_feature_names),
        
        # Order metadata
        'metadata_rows': meta_rows,
        'dcs': all_dcs,
        'carriers': all_carriers,
        'architecture': 'hierarchical_proxy_v2',
    }
    
    torch.save(dataset, out_dir / f"proxy_flat_{date}_dataset.pt")
    if debug_eligibility and mismatch_rows > 0:
        print(f"[eligibility-mismatch] total_rows={mismatch_rows} total_entries={mismatch_entries}")
    print(f"  saved {len(global_feats_list)} rows (proxy feature tensors) -> {out_dir}")
    print(
        f"  summary: orders_total={orders_total} "
        f"orders_skipped_exception={orders_skipped_exception} "
        f"orders_missing_preprocessed={orders_missing_preprocessed} "
        f"orders_with_rows={orders_with_rows} "
        f"skus_total={skus_total} "
        f"skus_skipped_shape={skus_skipped_shape} "
        f"rows_added={rows_added}"
    )
    if error_examples:
        print("  sample_errors:")
        for order_id, err in error_examples:
            print(f"    - {order_id}: {err}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_split", default="proxy_train", choices=["proxy_train", "test"])
    p.add_argument("--start_date")
    p.add_argument("--end_date")
    p.add_argument("--num_jobs", type=int, default=cpu_count())
    p.add_argument("--peak-only", action="store_true", help="Read from data/peak/csaa_solutions/ instead of data/csaa_solutions/")
    p.add_argument("--scenario_subset", choices=["all", "candidate_only"], default="all",
                   help="Scenario selection: 'candidate_only' uses only Stage 1 scenarios (SAA_N1 * SAA_Q) "
                        "matching SAA candidate generation; 'all' includes Stage 2 evaluation scenarios (+SAA_N2)")
    p.add_argument("--use-float16", action="store_true", default=True,
                   help="Store scenarios in float16 instead of float32 (50%% size reduction, default: True)")
    p.add_argument("--use-float32", dest="use_float16", action="store_false",
                   help="Store scenarios in float32 for full precision")
    p.add_argument("--use-precomputed-base-cost", action="store_true", default=True,
                   help="Use precomputed base cost tensor (fast) instead of on-the-fly computation")
    p.add_argument("--debug-eligibility", action="store_true", default=False,
                   help="Log when targets use (DC, carrier) pairs not marked eligible")
    args = p.parse_args()

    if args.peak_only:
        root = cfg.DATA_DIR / "peak" / "csaa_solutions" / args.data_split
    else:
        root = cfg.DATA_DIR / "csaa_solutions" / args.data_split
    dates = sorted(d.name for d in root.iterdir() if d.is_dir())
    if args.start_date: dates = [d for d in dates if d >= args.start_date]
    if args.end_date: dates = [d for d in dates if d <= args.end_date]

    print(f"{len(dates)} dates → {args.num_jobs} workers")
    print(f"Storage: {'float16 (50% compression)' if args.use_float16 else 'float32 (full precision)'}")
    print(f"Base cost: {'Precomputed lookup (fast)' if args.use_precomputed_base_cost else 'On-the-fly computation'}")
    
    with Pool(processes=args.num_jobs) as pool:
        pool.starmap(
            engineer_proxy_features,
            [(d, args.data_split, args.peak_only, args.scenario_subset, args.use_float16, args.use_precomputed_base_cost, args.debug_eligibility) for d in dates]
        )

    print("=== done ===")
