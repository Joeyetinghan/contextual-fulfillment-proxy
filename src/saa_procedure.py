import time
from functools import lru_cache
import numpy as np
import pandas as pd
import src.config as cfg
from src.optimization_models import solve_stochastic_lookahead_model, GUROBI_AVAILABLE
from src.data_structures.options import ensure_option_matrix
from src.empirical_scenarios import load_carrier_delivery_samples


@lru_cache(maxsize=2)
def _carrier_penalty_samples(order_set: str):
    per_carrier, global_arr = load_carrier_delivery_samples(order_set)
    per_carrier = {int(k): v for k, v in per_carrier.items()}
    return per_carrier, global_arr


def _sample_penalty_matrix(
    option_ids,
    option_to_carrier,
    order_set: str,
    promise_days: float,
    num_scenarios: int,
    rng: np.random.Generator,
):
    """
    Build a (num_scenarios x num_options) penalty matrix by sampling delivery penalties
    from cached carrier-specific empirical distributions.
    """
    per_carrier, global_arr = _carrier_penalty_samples(order_set)
    penalty_matrix = np.zeros((num_scenarios, len(option_ids)), dtype=float)
    
    for idx, opt_id in enumerate(option_ids):
        carrier_val = option_to_carrier.get(opt_id, -1)
        carrier = int(carrier_val) if carrier_val is not None else -1
        samples = per_carrier.get(carrier)
        arr = samples if samples is not None and len(samples) > 0 else global_arr
        if arr.size == 0:
            arr = np.array([promise_days], dtype=float)
        draws = rng.choice(arr, size=num_scenarios, replace=True)
        deviation = draws - promise_days
        penalty_matrix[:, idx] = (
            cfg.GAMMA_PLUS_LATE_PENALTY * np.maximum(0.0, deviation)
            + cfg.GAMMA_MINUS_EARLY_PENALTY * np.maximum(0.0, -deviation)
        )
    return penalty_matrix


def _coerce_inventory_matrix(on_hand_inventory, order_skus, dc_ids):
    """
    Normalize the on-hand inventory input (Series, MultiIndex DF, etc.) into a
    dense DataFrame indexed by SKU (str) and columns of DC IDs. Missing entries
    are filled with zeros so downstream lookups never KeyError.
    """
    inv = on_hand_inventory.copy()
    if isinstance(inv, pd.Series):
        inv = inv.to_frame()
    if isinstance(inv.index, pd.MultiIndex):
        base = inv
        if 'onhand_inventory' in base.columns:
            base = base['onhand_inventory']
        elif base.shape[1] == 1:
            base = base.iloc[:, 0]
        else:
            base = base.iloc[:, 0]
        inv_matrix = base.unstack(level=-1)
    else:
        inv_matrix = inv
    inv_matrix = inv_matrix.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    inv_matrix.index = inv_matrix.index.map(str)
    inv_matrix = inv_matrix.reindex(order_skus, fill_value=0.0)
    for dc in dc_ids:
        if dc not in inv_matrix.columns and str(dc) not in inv_matrix.columns:
            inv_matrix[dc] = 0.0
    return inv_matrix


def _get_inventory_value(row: pd.Series, dc_id: int) -> float:
    if dc_id in row.index:
        return float(row[dc_id])
    str_dc = str(dc_id)
    if str_dc in row.index:
        return float(row[str_dc])
    return 0.0


def _compute_residual_inventory(z_candidate, order_skus, dc_to_option_ids, inventory_matrix):
    """
    Compute per-SKU/DC residual inventory after subtracting first-stage shipments.
    """
    residual = {}
    for sku in order_skus:
        row = inventory_matrix.loc[sku]
        residual[sku] = {}
        for dc, option_ids in dc_to_option_ids.items():
            inventory = _get_inventory_value(row, dc)
            shipped = sum(z_candidate.get((sku, opt_id), 0.0) for opt_id in option_ids)
            residual[sku][dc] = max(0.0, inventory - shipped)
    return residual


def _compute_second_stage_cost(
    future_demand,
    residual_inventory,
    sorted_options,
    option_cost_map,
    option_to_dc_map,
    stockout_penalty,
):
    """
    Greedy optimal solver for the second-stage fulfillment cost.

    For each SKU, we walk the fulfillment lanes (options) in increasing cost order,
    consume any residual inventory at that lane, and accrue cost. A virtual lane
    with cost = stockout penalty is implicitly modeled by the final `remaining`
    check, so any leftover demand pays the penalty.
    """
    scenario_cost = 0.0
    for sku, demand in future_demand.items():
        remaining = float(demand)
        if remaining <= 0:
            continue

        dc_available = residual_inventory.get(sku, {}).copy()
        for opt_id in sorted_options:
            dc = option_to_dc_map.get(opt_id)
            if dc is None or dc == -1:
                continue
            available = dc_available.get(dc, 0.0)
            if available <= 0:
                continue

            ship = min(remaining, available)
            scenario_cost += option_cost_map[opt_id] * ship
            dc_available[dc] = available - ship
            remaining -= ship

            if remaining <= 0:
                break

        if remaining > 0:
            scenario_cost += stockout_penalty * remaining

    return scenario_cost


def calculate_current_stockout_cost(z_candidate, order_items, order_skus, option_ids, stockout_penalty):
    """Optimized stockout cost calculation."""
    # Pre-compute order quantities for all SKUs
    order_quantities = order_items.set_index('sku_ID')['quantity'].to_dict()
    order_quantities = {str(k): v for k, v in order_quantities.items()}
    
    total_cost = 0.0
    for sku in order_skus:
        order_qty = order_quantities.get(sku, 0)
        shipped_qty = sum(z_candidate.get((sku, opt_id), 0) for opt_id in option_ids)
        total_cost += stockout_penalty * max(0, order_qty - shipped_qty)
    
    return total_cost


def evaluate_candidate_solution(
    z_candidate,
    on_hand_inventory,
    demand_scenarios_N2,
    shipping_costs_N2,
    stockout_penalty,
    base_costs_by_option,
    dc_to_option_ids,
    option_to_dc,
    option_to_carrier,
    order_set,
    promise_days,
    future_penalties,
    order_items,
    return_stats: bool = False,
    return_components: bool = False,
):
    """Optimized solution evaluation."""
    # Evaluation does not require the solver; allow use in environments without Gurobi.
    if z_candidate is None:
        if return_components:
            return {
                "objective_mean": float("inf"),
                "objective_std": float("inf"),
                "current_stage_mean": float("inf"),
                "current_stage_std": float("inf"),
                "future_recourse_mean": float("inf"),
                "future_recourse_std": float("inf"),
                "num_scenarios": 0,
            }
        return (float('inf'), float('inf')) if return_stats else float('inf')
    
    order_skus = [str(k) for k in demand_scenarios_N2.keys()]
    option_ids = list(base_costs_by_option.index)
    num_scenarios = len(list(demand_scenarios_N2.values())[0])
    base_cost_vector = base_costs_by_option.reindex(option_ids).to_numpy(dtype=float)
    option_to_dc_map = {int(opt): int(option_to_dc.get(opt, -1)) for opt in option_ids}
    
    normalized_z = {(str(i), j): val for (i, j), val in z_candidate.items()}
    z_positive = {k: v for k, v in normalized_z.items() if v > 0}
    
    # Pre-compute consolidation discount
    w_sum = 0.0
    for opt_id in option_ids:
        option_total = sum(normalized_z.get((sku, opt_id), 0.0) for sku in order_skus)
        if option_total >= 2:
            w_sum += base_costs_by_option[opt_id] * option_total
    
    consolidation_discount = cfg.BETA_DISCOUNT * w_sum
    
    # Calculate current stockout cost
    current_stockout = calculate_current_stockout_cost(
        normalized_z, order_items, order_skus, option_ids, stockout_penalty
    )
    
    # Pre-compute z_candidate items for faster lookup
    z_items = {(i, j): val for (i, j), val in z_positive.items()}
    
    dc_ids = list(dc_to_option_ids.keys())
    inventory_matrix = _coerce_inventory_matrix(on_hand_inventory, order_skus, dc_ids)
    residual_inventory = _compute_residual_inventory(normalized_z, order_skus, dc_to_option_ids, inventory_matrix)
    
    static_sorted_options = None
    static_option_cost_map = None
    if future_penalties is None:
        sorted_indices = np.argsort(base_cost_vector, kind='mergesort')
        static_sorted_options = [option_ids[idx] for idx in sorted_indices]
        static_option_cost_map = {option_ids[idx]: float(base_cost_vector[idx]) for idx in range(len(option_ids))}
    else:
        pen = np.asarray(future_penalties, dtype=float)
        if pen.shape != (num_scenarios, len(option_ids)):
            raise ValueError(
                f"future_penalties must have shape (num_scenarios, num_options)=({num_scenarios},{len(option_ids)}); "
                f"got {getattr(pen, 'shape', None)}"
            )
        future_penalties = pen
    
    # Evaluate scenarios
    total_scenario_cost = 0.0
    total_scenario_cost_sq = 0.0
    total_shipping_cost = 0.0
    total_shipping_cost_sq = 0.0
    total_future_cost = 0.0
    total_future_cost_sq = 0.0
    for s in range(num_scenarios):
        scenario_idx = f"scenario_{s}"
        
        # Calculate shipping cost using pre-filtered z_items
        shipping_cost = sum(
            shipping_costs_N2[i].loc[scenario_idx, j] * val
            for (i, j), val in z_items.items()
        )
        shipping_cost = float(shipping_cost)
        total_shipping_cost += shipping_cost
        total_shipping_cost_sq += shipping_cost * shipping_cost
        
        # Prepare future demand
        future_demand = {i: demand_scenarios_N2[i].loc[scenario_idx] for i in order_skus}
        
        if future_penalties is None:
            sorted_options = static_sorted_options
            option_cost_map = static_option_cost_map
        else:
            penalty_row = future_penalties[s]
            option_costs = base_cost_vector + penalty_row
            sorted_indices = np.argsort(option_costs, kind='mergesort')
            sorted_options = [option_ids[idx] for idx in sorted_indices]
            option_cost_map = {option_ids[idx]: float(option_costs[idx]) for idx in range(len(option_ids))}
        
        # Second-stage (recourse) cost using deterministic base costs unless
        # scenario-dependent penalties are explicitly supplied.
        future_cost = _compute_second_stage_cost(
            future_demand,
            residual_inventory,
            sorted_options,
            option_cost_map,
            option_to_dc_map,
            stockout_penalty,
        )
        future_cost = float(future_cost)
        total_future_cost += future_cost
        total_future_cost_sq += future_cost * future_cost

        scenario_cost = float(shipping_cost + future_cost)
        total_scenario_cost += scenario_cost
        total_scenario_cost_sq += scenario_cost * scenario_cost

    mean_scenario_cost = float(total_scenario_cost / num_scenarios)
    mean_shipping_cost = float(total_shipping_cost / num_scenarios)
    mean_future_cost = float(total_future_cost / num_scenarios)
    current_stage_mean = mean_shipping_cost + current_stockout - consolidation_discount
    future_recourse_mean = mean_future_cost
    obj_mean = current_stage_mean + future_recourse_mean

    if not return_stats and not return_components:
        return obj_mean

    def _sample_std(total: float, total_sq: float, count: int) -> float:
        if count <= 1:
            return 0.0
        n = float(count)
        var = max(0.0, (total_sq - (total * total) / n) / (n - 1.0))
        return float(np.sqrt(var))

    objective_std = _sample_std(total_scenario_cost, total_scenario_cost_sq, num_scenarios)
    current_stage_std = _sample_std(total_shipping_cost, total_shipping_cost_sq, num_scenarios)
    future_recourse_std = _sample_std(total_future_cost, total_future_cost_sq, num_scenarios)

    if return_components:
        return {
            "objective_mean": float(obj_mean),
            "objective_std": float(objective_std),
            "current_stage_mean": float(current_stage_mean),
            "current_stage_std": float(current_stage_std),
            "future_recourse_mean": float(future_recourse_mean),
            "future_recourse_std": float(future_recourse_std),
            "num_scenarios": int(num_scenarios),
        }

    return obj_mean, float(objective_std)


def run_saa_procedure(
    order_items,
    options_df,
    on_hand_inventory_pivot,
    customer_dc,
    promise_days,
    stockout_penalty,
    order_set: str = 'test',
    master_seed: int = cfg.RANDOM_SEED,
    demand_scenarios=None,
    shipping_costs=None,
    verbose: bool = True,
    return_candidates: bool = False,
    use_future_penalties: bool | None = None,
):
    """Runs the full Sample Average Approximation procedure using (dc, carrier) options."""
    if use_future_penalties is None:
        use_future_penalties = bool(getattr(cfg, "SAA_USE_FUTURE_PENALTIES", False))
    if verbose:
        print("\n--- Starting SAA Procedure ---")
    start_time_saa = time.time()
    total_model_solve_time = 0
    
    # Master random number generator for reproducibility
    rng = np.random.default_rng(master_seed)

    options_matrix = ensure_option_matrix(options_df)
    option_ids_np = options_matrix.option_ids.astype(int)
    option_ids = option_ids_np.tolist()
    option_to_dc = pd.Series(options_matrix.dc_ids, index=option_ids_np)
    option_to_carrier = pd.Series(options_matrix.carrier_service_ids, index=option_ids_np)
    base_costs_by_option = pd.Series(options_matrix.base_costs, index=option_ids_np, dtype=float)
    dc_to_option_ids = options_matrix.dc_groups()

    num_candidate_scenarios = cfg.SAA_Q * cfg.SAA_N1
    num_evaluation_scenarios = cfg.SAA_N2
    total_scenarios = num_candidate_scenarios + num_evaluation_scenarios
    timings: dict[str, float | int] = {}

    split_start = time.time()
    if demand_scenarios is None or shipping_costs is None:
        raise ValueError("run_saa_procedure requires pre-generated demand_scenarios and shipping_costs.")

    if verbose:
        print(f"1. Using pre-generated scenarios ({total_scenarios} total).")
        print(
            "   Second-stage future shipping costs: "
            + ("base+sampled penalties" if use_future_penalties else "base deterministic only")
        )

    all_demand_scenarios = demand_scenarios
    all_shipping_costs = shipping_costs

    if all_demand_scenarios and all_shipping_costs:
        # Split scenarios into candidate and evaluation sets
        all_demand_scenarios_N1 = {
            sku: df.iloc[:num_candidate_scenarios] for sku, df in all_demand_scenarios.items()
        }
        all_shipping_costs_N1 = {
            sku: df.iloc[:num_candidate_scenarios] for sku, df in all_shipping_costs.items()
        }

        future_penalties_N1_all = None
        future_penalties_N2 = None
        if use_future_penalties:
            # Optional sampled penalty augmentation for second-stage shipping costs.
            penalty_rng = np.random.default_rng(master_seed + 1)
            future_penalties_all = _sample_penalty_matrix(
                option_ids,
                option_to_carrier,
                order_set,
                promise_days,
                total_scenarios,
                penalty_rng,
            )
            future_penalties_N1_all = future_penalties_all[:num_candidate_scenarios]

        if num_evaluation_scenarios > 0:
            demand_scenarios_N2 = {
                sku: df.iloc[num_candidate_scenarios:].reset_index(drop=True).rename(lambda i: f"scenario_{i}", axis='index') 
                for sku, df in all_demand_scenarios.items()
            }
            shipping_costs_N2 = {
                sku: df.iloc[num_candidate_scenarios:].reset_index(drop=True).rename(lambda i: f"scenario_{i}", axis='index')
                for sku, df in all_shipping_costs.items()
            }
            if use_future_penalties:
                future_penalties_N2 = future_penalties_all[num_candidate_scenarios:]
        else:
            demand_scenarios_N2, shipping_costs_N2, future_penalties_N2 = {}, {}, None
    else:
        all_demand_scenarios_N1, all_shipping_costs_N1 = {}, {}
        future_penalties_N1_all = None
        demand_scenarios_N2, shipping_costs_N2, future_penalties_N2 = {}, {}, None
    timings['scenario_split'] = time.time() - split_start

    # --- Candidate Generation ---
    if verbose:
        print(f"\n2. Generating {cfg.SAA_Q} candidate solutions using {cfg.SAA_N1} scenarios each...")
    candidates, objective_values = [], []
    candidate_current_stage_values: list[float] = []
    candidate_future_recourse_values: list[float] = []
    candidate_indices: list[int] = []
    candidate_stage_times: list[float] = []
    for q in range(cfg.SAA_Q):
        if verbose:
            print(f"  - Solving for candidate {q+1}/{cfg.SAA_Q}...")
        
        # Time model solving, scenario generation is already done
        start_solve = time.time()
        
        # Slice the scenarios for the current candidate
        start_idx = q * cfg.SAA_N1
        end_idx = (q + 1) * cfg.SAA_N1
        
        demand_scenarios_N1 = {
            sku: df.iloc[start_idx:end_idx].reset_index(drop=True).rename(lambda i: f"scenario_{i}", axis='index')
            for sku, df in all_demand_scenarios_N1.items()
        }
        shipping_costs_N1 = {
            sku: df.iloc[start_idx:end_idx].reset_index(drop=True).rename(lambda i: f"scenario_{i}", axis='index')
            for sku, df in all_shipping_costs_N1.items()
        }
        future_penalties_N1 = None
        if future_penalties_N1_all is not None:
            future_penalties_N1 = future_penalties_N1_all[start_idx:end_idx]

        solution, obj_val, _, objective_parts = solve_stochastic_lookahead_model(
            order_items=order_items,
            options_df=options_matrix,
            on_hand_inventory=on_hand_inventory_pivot,
            demand_scenarios=demand_scenarios_N1,
            shipping_costs=shipping_costs_N1,
            stockout_penalty=stockout_penalty,
            base_costs_by_option=base_costs_by_option,
            future_penalties=future_penalties_N1,
            return_objective_components=True,
        )
        solve_time = time.time() - start_solve
        total_model_solve_time += solve_time
        candidate_stage_times.append(solve_time)
        if verbose:
            print(f"    - Model Solve Time: {solve_time:.2f}s")

        if solution:
            candidates.append(solution)
            objective_values.append(obj_val)
            candidate_current_stage_values.append(float(objective_parts.get("current_stage", float("nan"))))
            candidate_future_recourse_values.append(float(objective_parts.get("future_recourse", float("nan"))))
            candidate_indices.append(int(q))

    if not candidates:
        if verbose:
            print("SAA Warning: No candidate solutions were generated.")
        total_saa_time = time.time() - start_time_saa
        timings['candidate_total'] = sum(candidate_stage_times)
        timings['candidate_avg'] = (
            sum(candidate_stage_times) / len(candidate_stage_times)
            if candidate_stage_times else 0.0
        )
        timings['num_candidates'] = len(candidate_stage_times)
        timings['evaluation_total'] = 0.0
        timings['evaluation_avg'] = 0.0
        timings['num_evaluations'] = 0
        timings['ubar_f'] = float('nan')
        timings['bar_f'] = float('nan')
        timings['gap'] = float('nan')
        timings['ubar_f_std'] = float('nan')
        timings['ubar_f_ci95'] = float('nan')
        timings['ubar_f_n'] = 0
        timings['ubar_f_current_stage'] = float('nan')
        timings['ubar_f_current_stage_std'] = float('nan')
        timings['ubar_f_current_stage_ci95'] = float('nan')
        timings['ubar_f_future_recourse'] = float('nan')
        timings['ubar_f_future_recourse_std'] = float('nan')
        timings['ubar_f_future_recourse_ci95'] = float('nan')
        timings['bar_f_std'] = float('nan')
        timings['bar_f_ci95'] = float('nan')
        timings['bar_f_n'] = 0
        timings['bar_f_current_stage'] = float('nan')
        timings['bar_f_current_stage_std'] = float('nan')
        timings['bar_f_current_stage_ci95'] = float('nan')
        timings['bar_f_future_recourse'] = float('nan')
        timings['bar_f_future_recourse_std'] = float('nan')
        timings['bar_f_future_recourse_ci95'] = float('nan')
        timings['total'] = total_saa_time
        if not return_candidates:
            return None, total_saa_time, timings

        cand_payload = {
            "master_seed": int(master_seed),
            "future_penalty_seed": int(master_seed + 1) if use_future_penalties else None,
            "use_future_penalties": bool(use_future_penalties),
            "candidate_indices": [],
            "candidate_obj_n1": [],
            "candidate_obj_n1_current_stage": [],
            "candidate_obj_n1_future_recourse": [],
            "candidate_eval_mean": [],
            "candidate_eval_std": [],
            "candidate_eval_current_stage_mean": [],
            "candidate_eval_current_stage_std": [],
            "candidate_eval_future_recourse_mean": [],
            "candidate_eval_future_recourse_std": [],
            "ubar_f_std": 0.0,
            "best_pos": None,
            "best_candidate_idx": None,
        }
        return None, total_saa_time, timings, {"candidates": [], "stats": cand_payload}
    timings['candidate_total'] = sum(candidate_stage_times)
    timings['candidate_avg'] = (
        sum(candidate_stage_times) / len(candidate_stage_times)
        if candidate_stage_times else 0.0
    )
    timings['num_candidates'] = len(candidate_stage_times)

    lower_bound = np.mean(objective_values) if objective_values else 0.0
    lower_bound_current_stage = np.mean(candidate_current_stage_values) if candidate_current_stage_values else float('nan')
    lower_bound_future_recourse = np.mean(candidate_future_recourse_values) if candidate_future_recourse_values else float('nan')
    timings['ubar_f'] = float(lower_bound)
    timings['ubar_f_current_stage'] = float(lower_bound_current_stage)
    timings['ubar_f_future_recourse'] = float(lower_bound_future_recourse)
    if verbose:
        print(f"\nStatistical Lower Bound (ubar_f): {lower_bound:.4f}")

    # --- Candidate Evaluation ---
    if verbose:
        print(f"\n3. Evaluating {len(candidates)} candidates using {cfg.SAA_N2} scenarios each...")
    best_candidate, best_objective, best_gap = None, float('inf'), float('inf')
    best_pos: int | None = None
    candidate_eval_mean: list[float] = []
    candidate_eval_std: list[float] = []
    candidate_eval_current_stage_mean: list[float] = []
    candidate_eval_current_stage_std: list[float] = []
    candidate_eval_future_recourse_mean: list[float] = []
    candidate_eval_future_recourse_std: list[float] = []
    evaluation_times: list[float] = []
    for i, z_candidate in enumerate(candidates):
        if verbose:
            print(f"  - Evaluating candidate {i+1}/{len(candidates)}...")
        
        start_eval = time.time()
        eval_stats = evaluate_candidate_solution(
            z_candidate=z_candidate,
            on_hand_inventory=on_hand_inventory_pivot,
            demand_scenarios_N2=demand_scenarios_N2,
            shipping_costs_N2=shipping_costs_N2,
            stockout_penalty=stockout_penalty,
            base_costs_by_option=base_costs_by_option,
            dc_to_option_ids=dc_to_option_ids,
            option_to_dc=option_to_dc,
            option_to_carrier=option_to_carrier,
            order_set=order_set,
            promise_days=promise_days,
            future_penalties=future_penalties_N2,
            order_items=order_items,
            return_components=True,
        )
        estimated_obj = float(eval_stats["objective_mean"])
        est_std = float(eval_stats["objective_std"])
        candidate_eval_mean.append(float(estimated_obj))
        candidate_eval_std.append(float(est_std))
        candidate_eval_current_stage_mean.append(float(eval_stats["current_stage_mean"]))
        candidate_eval_current_stage_std.append(float(eval_stats["current_stage_std"]))
        candidate_eval_future_recourse_mean.append(float(eval_stats["future_recourse_mean"]))
        candidate_eval_future_recourse_std.append(float(eval_stats["future_recourse_std"]))

        # Ensure scalar for logging/comparisons in both branches.
        estimated_obj = float(estimated_obj)
        eval_time = time.time() - start_eval
        total_model_solve_time += eval_time
        evaluation_times.append(eval_time)

        gap = estimated_obj - lower_bound
        if verbose:
            print(f"    - Estimated Objective (bar_f): {estimated_obj:.4f}, Gap: {gap:.4f}, Eval Time: {eval_time:.2f}s")

        if estimated_obj < best_objective:
            best_gap, best_objective, best_candidate = gap, estimated_obj, z_candidate
            best_pos = i

    timings['bar_f'] = float(best_objective) if best_objective != float('inf') else float('inf')
    timings['gap'] = float(best_gap) if best_gap != float('inf') else float('inf')
    ubar_std = float(np.std(np.asarray(objective_values), ddof=1)) if len(objective_values) > 1 else 0.0
    ubar_n = int(len(objective_values))
    ubar_ci95 = 1.96 * ubar_std / np.sqrt(ubar_n) if ubar_n > 0 else float('nan')
    valid_ubar_current = np.asarray(
        [float(v) for v in candidate_current_stage_values if np.isfinite(float(v))],
        dtype=float,
    )
    valid_ubar_future = np.asarray(
        [float(v) for v in candidate_future_recourse_values if np.isfinite(float(v))],
        dtype=float,
    )
    ubar_current_std = float(np.std(valid_ubar_current, ddof=1)) if valid_ubar_current.size > 1 else 0.0
    ubar_future_std = float(np.std(valid_ubar_future, ddof=1)) if valid_ubar_future.size > 1 else 0.0
    ubar_current_ci95 = (
        1.96 * ubar_current_std / np.sqrt(valid_ubar_current.size)
        if valid_ubar_current.size > 0
        else float('nan')
    )
    ubar_future_ci95 = (
        1.96 * ubar_future_std / np.sqrt(valid_ubar_future.size)
        if valid_ubar_future.size > 0
        else float('nan')
    )
    bar_n = int(len(next(iter(demand_scenarios_N2.values())))) if demand_scenarios_N2 else 0
    if best_pos is not None and best_pos < len(candidate_eval_std):
        bar_std = float(candidate_eval_std[best_pos])
    else:
        bar_std = float('nan')
    bar_ci95 = 1.96 * bar_std / np.sqrt(bar_n) if bar_n > 0 and np.isfinite(bar_std) else float('nan')
    if best_pos is not None and best_pos < len(candidate_eval_current_stage_mean):
        bar_current_stage = float(candidate_eval_current_stage_mean[best_pos])
        bar_current_stage_std = float(candidate_eval_current_stage_std[best_pos])
        bar_future_recourse = float(candidate_eval_future_recourse_mean[best_pos])
        bar_future_recourse_std = float(candidate_eval_future_recourse_std[best_pos])
    else:
        bar_current_stage = float('nan')
        bar_current_stage_std = float('nan')
        bar_future_recourse = float('nan')
        bar_future_recourse_std = float('nan')
    bar_current_stage_ci95 = (
        1.96 * bar_current_stage_std / np.sqrt(bar_n)
        if bar_n > 0 and np.isfinite(bar_current_stage_std)
        else float('nan')
    )
    bar_future_recourse_ci95 = (
        1.96 * bar_future_recourse_std / np.sqrt(bar_n)
        if bar_n > 0 and np.isfinite(bar_future_recourse_std)
        else float('nan')
    )
    timings['ubar_f_std'] = float(ubar_std)
    timings['ubar_f_ci95'] = float(ubar_ci95)
    timings['ubar_f_n'] = int(ubar_n)
    timings['ubar_f_current_stage_std'] = float(ubar_current_std)
    timings['ubar_f_current_stage_ci95'] = float(ubar_current_ci95)
    timings['ubar_f_future_recourse_std'] = float(ubar_future_std)
    timings['ubar_f_future_recourse_ci95'] = float(ubar_future_ci95)
    timings['bar_f_std'] = float(bar_std)
    timings['bar_f_ci95'] = float(bar_ci95)
    timings['bar_f_n'] = int(bar_n)
    timings['bar_f_current_stage'] = float(bar_current_stage)
    timings['bar_f_current_stage_std'] = float(bar_current_stage_std)
    timings['bar_f_current_stage_ci95'] = float(bar_current_stage_ci95)
    timings['bar_f_future_recourse'] = float(bar_future_recourse)
    timings['bar_f_future_recourse_std'] = float(bar_future_recourse_std)
    timings['bar_f_future_recourse_ci95'] = float(bar_future_recourse_ci95)
    
    total_saa_time = time.time() - start_time_saa
    timings['evaluation_total'] = sum(evaluation_times)
    timings['evaluation_avg'] = (
        sum(evaluation_times) / len(evaluation_times)
        if evaluation_times else 0.0
    )
    timings['num_evaluations'] = len(evaluation_times)
    timings['total'] = total_saa_time
    if verbose:
        print(f"\n--- SAA Procedure Complete ---")
        if total_saa_time > 0:
            print(f"Total time: {total_saa_time:.2f}s")
            print(f"  - Model Solving:       {total_model_solve_time:.2f}s ({total_model_solve_time/total_saa_time:.1%})")
        print(f"\nBest solution found with estimated objective: {best_objective:.4f} and gap: {best_gap:.4f}")
    
    fulfillment_plan = pd.DataFrame(columns=['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity'])
    if best_candidate:
        option_metadata = options_matrix.to_dataframe().set_index('option_id')[['dc_id', 'carrier_service_id']]
        for (sku, opt_id), qty in best_candidate.items():
            if qty > 1e-6:
                dc_val = int(option_metadata.loc[opt_id, 'dc_id'])
                carrier_val = int(option_metadata.loc[opt_id, 'carrier_service_id'])
                fulfillment_plan.loc[len(fulfillment_plan)] = [sku, dc_val, carrier_val, int(round(qty))]

    if not return_candidates:
        return fulfillment_plan, total_saa_time, timings

    # Return candidate solutions and their evaluation stats for external logging/storage.
    cand_payload = {
        "master_seed": int(master_seed),
        "future_penalty_seed": int(master_seed + 1) if use_future_penalties else None,
        "use_future_penalties": bool(use_future_penalties),
        "candidate_indices": candidate_indices,
        "candidate_obj_n1": [float(v) for v in objective_values],
        "candidate_obj_n1_current_stage": [float(v) for v in candidate_current_stage_values],
        "candidate_obj_n1_future_recourse": [float(v) for v in candidate_future_recourse_values],
        "candidate_eval_mean": candidate_eval_mean,
        "candidate_eval_std": candidate_eval_std,
        "candidate_eval_current_stage_mean": candidate_eval_current_stage_mean,
        "candidate_eval_current_stage_std": candidate_eval_current_stage_std,
        "candidate_eval_future_recourse_mean": candidate_eval_future_recourse_mean,
        "candidate_eval_future_recourse_std": candidate_eval_future_recourse_std,
        "ubar_f_std": ubar_std,
        "best_pos": int(best_pos) if best_pos is not None else None,
        "best_candidate_idx": int(candidate_indices[best_pos]) if best_pos is not None and best_pos < len(candidate_indices) else None,
    }

    return fulfillment_plan, total_saa_time, timings, {"candidates": candidates, "stats": cand_payload}
