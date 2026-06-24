import time
from collections import defaultdict
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd

from src.data_structures.options import OptionMatrix, ensure_option_matrix


class PrimalDualState:
    """
    Persistent state for the online primal-dual algorithm.

    Tracks and persists dual variables λ_{sku,dc} across orders, and the initial
    inventory s0 per (sku,dc) pair which determines the scaling B for updates.

    Required:
      - s0_by_key[(sku_ID, dc_ID)] = initial on-hand inventory s^0_{i,j} (> 0)

    Params must be one of:
      (A) Auto mode: {'auto_params': True, 'kappa': >1, 'allow_null': bool}
          - alpha1, alpha2, beta are derived once from κ and persisted.
      (B) Manual mode: {'alpha1': >0, 'alpha2': any, 'beta': >0, 'allow_null': bool}
          - use provided parameters as-is, no derivation.
    """

    def __init__(self, *, s0_by_key: Dict[Tuple[Any, Any], float], params: Dict[str, Any]):
        if not s0_by_key:
            raise ValueError("s0_by_key is required and cannot be empty.")
        # Keep only positive s0
        self.s0_by_key: Dict[Tuple[Any, Any], float] = {
            k: float(v) for k, v in s0_by_key.items() if float(v) > 0.0
        }
        if not self.s0_by_key:
            raise ValueError("All s0 entries are non-positive; nothing to optimize.")

        self.lambda_by_key: Dict[Tuple[Any, Any], float] = defaultdict(float)
        self.params: Dict[str, Any] = dict(params)

    def B(self, key: Tuple[Any, Any], alpha1: float, alpha2: float) -> float:
        if key not in self.s0_by_key:
            raise KeyError(f"s0 missing for key={key}")
        return float(alpha1 * self.s0_by_key[key] + alpha2)


def _derive_params_from_kappa_strict(kappa: float, s0_min: float) -> tuple[float, float, float]:
    """
    Derive (alpha1, alpha2, beta) from competitive-ratio parameter κ (>1) as in the paper.

    Let B_min = max(alpha1 * s0_min, 1). Then
      alpha1 = 1 / (1 + log κ), alpha2 = 0,
      beta   = κ / ((1 + 1/B_min)^(B_min/alpha1) - 1).

    Args:
        kappa: competitive-ratio parameter (> 1.0)
        s0_min: minimum initial inventory across all keys (must be > 0)

    Returns:
        (alpha1, alpha2, beta) all positive and finite.
    """
    if not (kappa is not None and kappa > 1.0 and np.isfinite(kappa)):
        raise ValueError("kappa must be provided and > 1.0 in auto-params mode.")
    alpha1 = 1.0 / (1.0 + np.log(kappa))
    alpha2 = 0.0
    s0m = float(s0_min)
    if s0m <= 0.0 or not np.isfinite(s0m):
        raise ValueError("s0_min must be positive and finite.")
    B_min = max(alpha1 * s0m, 1.0)
    beta = float(kappa / (np.power(1.0 + 1.0 / B_min, B_min / alpha1) - 1.0))
    if not (beta > 0 and np.isfinite(beta)):
        raise ValueError("Derived beta is non-positive or non-finite; check kappa and s0.")
    return alpha1, alpha2, beta


def _paper_reward_from_costs(eligible_dcs: List[Any], base_costs: pd.Series) -> Dict[Any, float]:
    """
    Map base shipping costs to per-DC rewards per the paper's transformation.

    Rewards increase as cost decreases; exponentiated, centered by the mean cost
    to keep values in a numerically stable range.

    Args:
        eligible_dcs: DCs with at least 1 unit of inventory for the SKU.
        base_costs: Series of costs by DC (finite for all eligible).

    Returns:
        Dict[dc] -> reward(dc)
    """
    costs = base_costs.loc[eligible_dcs].astype(float)
    if costs.isna().any() or (~np.isfinite(costs)).any():
        raise ValueError("base_costs contains NaN/Inf for one or more eligible DCs.")
    c_bar = float(costs.mean()) if len(costs) else 0.0
    denom = max(c_bar, 1.0)
    return {dc: float(np.exp((c_bar - float(base_costs.loc[dc])) / denom)) for dc in eligible_dcs}


def primal_dual_fulfillment(
    order_info: pd.Series,
    order_items: pd.DataFrame,
    on_hand_inventory_pivot: pd.DataFrame,
    base_costs: pd.Series,
    compatible_dcs: List[Any],
    dc_to_region: Dict[Any, Any],     # unused; kept for API
    order_set: str,
    simulation_date: str,
    order_idx: int,
    verbose: bool = False,
    state: PrimalDualState | None = None,
    order_options: OptionMatrix | pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Online primal–dual assignment for a single order, with persistent duals.

    For each unit in each SKU of the order:
      1) Build eligible DCs with available inventory.
      2) Compute rewards from costs, and apply null-node gating (if enabled) to
         ensure only profitable assignments proceed.
      3) Choose DC maximizing (reward - lambda), with a deterministic tie-breaker
         on lower base cost.
      4) Update dual variable λ for the chosen (sku, dc) using the paper's update
         with B(s0) scaling and parameter β.

    Args:
        order_info, order_items, on_hand_inventory_pivot, base_costs: order context.
        unit_costs_df, dc_to_region: unused here; included for API consistency.
        compatible_dcs: list of DCs allowed for this order.
        order_set, simulation_date, order_idx: metadata for logging.
        verbose: toggles verbose internals (not used here).
        state: persistent PrimalDualState with s0 and parameters.

    Returns:
        plan_df: DataFrame of assignments with ['sku_ID','dc_ori','quantity'].
        meta: runtime and parameter diagnostics.
    """
    if state is None:
        raise ValueError("PrimalDualState required.")
    start = time.perf_counter()

    # Parameters (paper-exact): derive once if auto mode
    params = state.params
    if params.get('auto_params', False):
        if 'alpha1' not in params:
            s0_min = min(state.s0_by_key.values())
            a1, a2, b = _derive_params_from_kappa_strict(float(params['kappa']), float(s0_min))
            params.update({'alpha1': float(a1), 'alpha2': float(a2), 'beta': float(b)})
    for req in ('alpha1', 'alpha2', 'beta'):
        if req not in params:
            raise KeyError(f"Missing required parameter '{req}'. Provide manual params or enable auto_params with kappa.")
    alpha1 = float(params['alpha1'])
    alpha2 = float(params['alpha2'])
    beta   = float(params['beta'])
    if not (alpha1 > 0 and beta > 0 and np.isfinite(alpha1) and np.isfinite(beta)):
        raise ValueError("alpha1 and beta must be positive and finite.")
    allow_null = bool(params.get('allow_null', True))  # default True

    # Working inventory used only within this order's allocation loop
    working_inv = on_hand_inventory_pivot.copy()

    assigned_qty: Dict[Tuple[Any, Any], int] = defaultdict(int)
    null_skipped_units = 0

    # Carrier annotation (choose cheapest carrier per DC for this order, if provided)
    best_carrier_by_dc: Dict[Any, Any] = {}
    if order_options is not None:
        try:
            om = ensure_option_matrix(order_options)
            options_df = om.to_dataframe()
        except Exception:
            options_df = None
        if options_df is not None and not options_df.empty and {'dc_id', 'carrier_service_id', 'base_cost'} <= set(options_df.columns):
            tmp = options_df[['dc_id', 'carrier_service_id', 'base_cost']].copy()
            tmp['base_cost'] = tmp['base_cost'].astype(float)
            idx = tmp.groupby('dc_id')['base_cost'].idxmin()
            best = tmp.loc[idx]
            best_carrier_by_dc = dict(zip(best['dc_id'], best['carrier_service_id']))

    # Respect declared compatible DC ordering
    ordered_dcs = [dc for dc in compatible_dcs if dc in base_costs.index]

    for sku, qty in order_items[['sku_ID', 'quantity']].itertuples(index=False):
        remaining = int(qty)
        if remaining <= 0:
            continue

        inv_row = working_inv.loc[sku] if sku in working_inv.index else pd.Series(0, index=working_inv.columns)

        while remaining > 0:
            # Eligible DCs: positive inventory and in the provided order list
            eligible = [dc for dc in ordered_dcs if dc in inv_row.index and inv_row.loc[dc] >= 1]
            if not eligible:
                break

            rewards = _paper_reward_from_costs(eligible, base_costs)

            # Null-node (φ) gating per paper: stop if all (reward - lambda) ≤ 0
            if allow_null:
                max_score = max((rewards[dc] - state.lambda_by_key[(sku, dc)] for dc in eligible), default=-np.inf)
                if max_score <= 0.0:
                    null_skipped_units += remaining
                    break

            # Choose argmax_dc (reward - lambda) with cost tie-breaker
            best_dc = None
            best_score = -np.inf
            for dc in eligible:
                lam = state.lambda_by_key[(sku, dc)]
                score = rewards[dc] - lam
                if (score > best_score) or (score == best_score and (best_dc is None or base_costs.loc[dc] < base_costs.loc[best_dc])):
                    best_score = score
                    best_dc = dc

            if best_dc is None:
                break

            # Assign one unit and consume inventory
            assigned_qty[(sku, best_dc)] += 1
            inv_row.loc[best_dc] = float(inv_row.loc[best_dc]) - 1.0
            remaining -= 1

            # Dual update for the chosen edge only: λ ← λ*(1+1/B) + β*(R/B)
            key = (sku, best_dc)
            lam_old = state.lambda_by_key[key]
            B_val = state.B(key, alpha1, alpha2)
            if not (B_val > 0 and np.isfinite(B_val)):
                raise ValueError(f"Non-positive/invalid B for key={key}.")
            lam_new = lam_old * (1.0 + 1.0 / B_val) + beta * (rewards[best_dc] / B_val)
            state.lambda_by_key[key] = lam_new

    rows = []
    for (sku, dc), q in assigned_qty.items():
        if q <= 0:
            continue
        rows.append({
            'sku_ID': sku,
            'dc_ori': dc,
            'carrier_service_id': best_carrier_by_dc.get(dc, None) if best_carrier_by_dc else None,
            'quantity': int(q),
        })
    plan_df = (
        pd.DataFrame(rows, columns=['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity'])
        if rows
        else pd.DataFrame(columns=['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity'])
    )

    meta = {
        'algo': 'primal_dual_paper_exact',
        'runtime_seconds': time.perf_counter() - start,
        'alpha1': alpha1,
        'alpha2': alpha2,
        'beta': beta,
        'kappa': float(params['kappa']) if params.get('auto_params', False) else None,
        'allow_null': allow_null,
        'null_skipped_units': int(null_skipped_units),
    }
    return plan_df, meta 


def choose_fulfillment_option_level(
    order,
    items,
    option_ids,
    eligible_mask,
    features_df,
    costs_series,
    inventory_snapshot,
    *,
    order_set: str = 'test',
    simulation_date: str | None = None,
    order_idx: int = 0,
    verbose: bool = False,
    state: PrimalDualState | None = None,
):
    """
    Option-level wrapper for primal-dual (simulator-compatible).

    Returns:
        (OrderDecision, runtime_seconds)
    """
    from src.simulator.entities import OrderDecision, ItemAllocation
    if simulation_date is None:
        raise ValueError("simulation_date is required for primal_dual")
    if state is None:
        raise ValueError("PrimalDualState required for primal_dual")

    # Reuse the proven adapters from contextual_saa to keep schemas consistent.
    from src.algo.contextual_saa import _prepare_order_items, _build_inventory_pivot, _build_option_matrix

    fallback_order_time = order.order_time or (pd.Timestamp(simulation_date) if simulation_date else None)
    order_items_df, unique_skus, original_qty = _prepare_order_items(order, items, fallback_order_time)
    inventory_pivot = _build_inventory_pivot(unique_skus, option_ids, inventory_snapshot)
    order_options = _build_option_matrix(option_ids, costs_series)

    # Base costs by DC: min carrier option for this order
    options_df = order_options.to_dataframe()
    base_costs_by_dc = (
        options_df.groupby('dc_id')['base_cost'].min()
        if not options_df.empty and {'dc_id', 'base_cost'} <= set(options_df.columns)
        else pd.Series(dtype=float)
    )
    compatible_dcs = sorted({int(opt[0]) for opt in option_ids})

    order_info = pd.Series({
        'order_time': order.order_time or pd.Timestamp.now(),
        'dc_des': str(getattr(order, 'customer_dc', '') or order.dest_state),
        'promise_delivery_days': order.promise_delivery_days,
    })

    plan_df, meta = primal_dual_fulfillment(
        order_info=order_info,
        order_items=order_items_df,
        on_hand_inventory_pivot=inventory_pivot,
        base_costs=base_costs_by_dc,
        compatible_dcs=compatible_dcs,
        dc_to_region={},  # unused
        order_set=order_set,
        simulation_date=simulation_date,
        order_idx=order_idx,
        verbose=verbose,
        state=state,
        order_options=order_options,
    )

    runtime = float(meta.get('runtime_seconds', 0.0) or 0.0)

    allocations: list[ItemAllocation] = []
    allocated_qty: dict[str, int] = defaultdict(int)
    unfilled: dict[str, int] = {}

    if plan_df is not None and not plan_df.empty:
        plan_values = plan_df[['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity']].to_numpy()
    else:
        plan_values = np.empty((0, 4))

    for sku_id, dc_id, carrier_id, qty in plan_values:
        sku_id = str(sku_id)
        qty_int = int(float(qty))
        if qty_int <= 0 or pd.isna(dc_id) or pd.isna(carrier_id):
            continue
        allocations.append(ItemAllocation(
            sku_id=sku_id,
            option_id=(int(dc_id), int(carrier_id)),
            quantity=qty_int,
        ))
        allocated_qty[sku_id] += qty_int

    for sku_id, orig in original_qty.items():
        rem = int(orig) - int(allocated_qty.get(sku_id, 0))
        if rem > 0:
            unfilled[sku_id] = rem

    decision = OrderDecision(allocations=allocations, unfilled=unfilled if unfilled else None)
    return decision, runtime


def create_policy_for_simulation(
    catalog,
    precompute,
    state,
    order_set: str,
    simulation_date: str,
    *,
    pd_auto_params: bool = True,
    pd_kappa: float = 2.0,
    pd_alpha1: float | None = None,
    pd_alpha2: float | None = None,
    pd_beta: float | None = None,
    pd_allow_null: bool = True,
    **kwargs,
):
    """Create primal-dual policy closure for the unified simulator."""
    from src.simulator.features import build_costs
    from src.simulator.entities import OrderDecision

    # Build s0_by_key from the *initial* inventory snapshot.
    s0_by_key: dict[tuple[Any, Any], float] = {}
    for dc_id, sku_map in getattr(state, 'inventory', {}).items():
        for sku_id, qty in sku_map.items():
            qf = float(qty)
            if qf > 0:
                s0_by_key[(str(sku_id), int(dc_id))] = qf
    if not s0_by_key:
        raise ValueError("No positive (sku,dc) initial inventory for primal_dual.")

    if pd_auto_params:
        state_params = {'auto_params': True, 'kappa': float(pd_kappa), 'allow_null': bool(pd_allow_null)}
    else:
        if pd_alpha1 is None or pd_alpha2 is None or pd_beta is None:
            raise ValueError("Provide pd_alpha1, pd_alpha2, pd_beta when pd_auto_params is False")
        state_params = {'alpha1': float(pd_alpha1), 'alpha2': float(pd_alpha2), 'beta': float(pd_beta), 'allow_null': bool(pd_allow_null)}

    pd_state = PrimalDualState(s0_by_key=s0_by_key, params=state_params)

    def policy(order):
        option_ids = catalog.eligible_for_order(order)
        if not option_ids:
            return OrderDecision(
                allocations=[],
                unfilled={item.sku_id: item.quantity for item in order.items}
            ), 0.0

        costs_series = build_costs(order, option_ids, catalog, precompute)
        local_idx = hash(order.order_id) % 10000
        return choose_fulfillment_option_level(
            order=order,
            items=order.items,
            option_ids=option_ids,
            eligible_mask=None,
            features_df=None,
            costs_series=costs_series,
            inventory_snapshot=state.inventory,
            order_set=order_set,
            simulation_date=simulation_date,
            order_idx=local_idx,
            verbose=False,
            state=pd_state,
        )

    return policy