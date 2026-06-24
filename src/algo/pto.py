"""
Point-Predict-Then-Optimize (PTO) baseline algorithm.

This module implements the PTO approach that solves deterministic optimization
using the median (50% quantile) of ML model outputs for both demand and delivery time.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

import src.config as cfg
from src.scenario_generator import generate_single_scenario
from src.optimization_models import solve_deterministic_model
from src.utils import calculate_lookahead_periods
from src.data_structures.options import OptionMatrix, ensure_option_matrix

logger = logging.getLogger(__name__)


def pto_fulfillment(
    order_info: pd.Series,
    order_items: pd.DataFrame,
    on_hand_inventory_pivot: pd.DataFrame,
    order_options: OptionMatrix | pd.DataFrame,
    order_set: str,
    simulation_date: str,
    verbose: bool = False,
    dynamic_features: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate fulfillment plan using Point-Predict-Then-Optimize (PTO).
    
    Solves deterministic optimization using the median (50% quantile) of ML model
    outputs for both demand and delivery time predictions.
    
    Args:
        order_info: Order information Series
        order_items: Order items DataFrame
        on_hand_inventory_pivot: Pivoted inventory DataFrame
        order_options: Option matrix / DataFrame of eligible (dc, carrier) options
        order_set: Order set ('test' or 'proxy_train')
        simulation_date: Simulation date string
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple of (plan_df, metadata_dict)
    """
    current_order_time = order_info['order_time']
    lookahead_periods = calculate_lookahead_periods(current_order_time, simulation_date)
    period = 'proxy' if order_set == 'proxy_train' else 'test'

    options_matrix = ensure_option_matrix(order_options)
    order_options_df = options_matrix.to_dataframe()
    
    # --- Single scenario via helper ---
    demand_median, median_costs = generate_single_scenario(
        order_info=order_info,
        order_items=order_items,
        options_df=order_options_df,
        period=period,
        lookahead_periods=lookahead_periods,
        verbose=verbose,
        dynamic_features=dynamic_features,
    )
    
    # Create shipping costs scenario in DataFrame format (same costs for all SKUs)
    # Convert median_costs dict {option_id: cost} to DataFrame with option_id columns
    # This matches the format expected by solve_deterministic_model (as used by empirical SAA)
    option_ids = order_options_df['option_id'].tolist()
    base_costs_series = order_options_df.set_index('option_id')['base_cost']
    costs_row = [float(median_costs.get(opt, base_costs_series.loc[opt])) for opt in option_ids]
    costs_df = pd.DataFrame(
        [costs_row],
        index=['scenario_0'],
        columns=option_ids
    )
    shipping_costs_scenario = {sku: costs_df.copy() for sku in demand_median.keys()}
    
    # Solve deterministic optimization
    plan_df, runtime = solve_deterministic_model(
        order_items=order_items,
        options_df=order_options_df,
        on_hand_inventory=on_hand_inventory_pivot.copy(),
        demand_scenario=demand_median,
        shipping_costs_scenario=shipping_costs_scenario,
        stockout_penalty=cfg.STOCKOUT_PENALTY_PER_UNIT,
        base_costs_by_option=None,
    )
    
    metadata = {
        'runtime_seconds': runtime,
        'deterministic': 'median',
        'lookahead_periods': lookahead_periods
    }
    
    return plan_df, metadata


def choose_fulfillment_option_level(
    order,
    items,
    option_ids,
    eligible_mask,
    features_df,
    costs_series,
    inventory_snapshot,
    order_set: str = 'test',
    simulation_date: str | None = None,
    order_idx: int = 0,
    verbose: bool = False,
    dynamic_features: pd.DataFrame | None = None,
):
    """
    Option-level wrapper for PTO fulfillment.

    Mirrors `contextual_saa.choose_fulfillment_option_level` so policies can be swapped
    without re-learning data-shape expectations.
    """
    from src.simulator.entities import OrderDecision, ItemAllocation

    if simulation_date is None:
        raise ValueError("simulation_date is required for PTO")

    # Use shared algorithm utilities for consistent data preparation
    from src.algo.algo_utils import prepare_order_items, build_inventory_pivot, build_option_matrix

    fallback_order_time = order.order_time or (pd.Timestamp(simulation_date) if simulation_date else None)
    order_items_df, unique_skus, original_qty = prepare_order_items(order, items, fallback_order_time)
    inventory_pivot = build_inventory_pivot(unique_skus, option_ids, inventory_snapshot)
    order_options = build_option_matrix(option_ids, costs_series)

    dest_dc_value = order.customer_dc
    order_info = pd.Series({
        'order_time': order.order_time or pd.Timestamp.now(),
        'dc_des': dest_dc_value if dest_dc_value is not None else order.dest_state,
        'promise_delivery_days': order.promise_delivery_days,
    })

    plan_df, metadata = pto_fulfillment(
        order_info=order_info,
        order_items=order_items_df,
        on_hand_inventory_pivot=inventory_pivot,
        order_options=order_options,
        order_set=order_set,
        simulation_date=simulation_date,
        verbose=verbose,
        dynamic_features=dynamic_features,
    )
    if plan_df is None:
        plan_df = pd.DataFrame(columns=['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity'])

    runtime = metadata.get('runtime_seconds', 0.0)

    allocations = []
    allocated_qty: dict[str, int] = {}

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
        allocated_qty[sku_id] = allocated_qty.get(sku_id, 0) + qty_int

    unfilled = {}
    for sku_id, orig in original_qty.items():
        rem = int(orig) - int(allocated_qty.get(sku_id, 0))
        if rem > 0:
            unfilled[sku_id] = rem

    decision = OrderDecision(allocations=allocations, unfilled=unfilled or None)
    return decision, runtime


def create_policy_for_simulation(catalog, precompute, state, order_set, simulation_date, **kwargs):
    """Create PTO policy closure for simulation (mirrors CSAA/proxy style)."""
    from src.simulator.features import build_costs
    from src.simulator.entities import OrderDecision

    def policy(order):
        option_ids = catalog.eligible_for_order(order)
        if not option_ids:
            return OrderDecision(
                allocations=[],
                unfilled={item.sku_id: item.quantity for item in order.items}
            ), 0.0

        costs_series = build_costs(order, option_ids, catalog, precompute)
        candidate_dc_ids = sorted(set(int(opt[0]) for opt in option_ids))
        dynamic_snapshot = state.build_dc_event_snapshot(dc_ids=candidate_dc_ids)

        order_idx = hash(order.order_id) % 10000
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
            order_idx=order_idx,
            verbose=False,
            dynamic_features=dynamic_snapshot,
        )

    return policy
