"""
Empirical SAA (Stochastic Average Approximation) baseline algorithm.

This module implements the empirical SAA approach that uses the SAA procedure
but generates scenarios from empirical distributions of the training data
instead of ML model predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

import src.config as cfg
from src.empirical_scenarios import generate_empirical_scenarios
from src.saa_procedure import run_saa_procedure
from src.utils import calculate_lookahead_periods
from src.algo.algo_utils import prepare_order_items, build_inventory_pivot, build_option_matrix
from src.data_structures.options import OptionMatrix, ensure_option_matrix

logger = logging.getLogger(__name__)


def empirical_saa_fulfillment(
    order_info: pd.Series,
    order_items: pd.DataFrame,
    on_hand_inventory_pivot: pd.DataFrame,
    order_options: OptionMatrix | pd.DataFrame,
    order_set: str,
    simulation_date: str,
    order_idx: int,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate fulfillment plan using empirical SAA.
    
    Uses the SAA procedure but generates scenarios from empirical distributions
    of the training data instead of ML model predictions.
    
    Args:
        order_info: Order information Series
        order_items: Order items DataFrame
        on_hand_inventory_pivot: Pivoted inventory DataFrame
        base_costs: Base shipping costs Series
        unit_costs_df: Unit costs DataFrame
        compatible_dcs: List of compatible DCs
        dc_to_region: DC to region mapping
        order_set: Order set ('test' or 'proxy_train')
        simulation_date: Simulation date string
        order_idx: Order index for seeding
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple of (plan_df, metadata_dict)
    """
    current_order_time = order_info['order_time']
    customer_dc = order_info['dc_des']
    promise_days = order_info['promise_delivery_days']
    lookahead_periods = calculate_lookahead_periods(current_order_time, simulation_date)
    total_scenarios_needed = cfg.SAA_Q * cfg.SAA_N1 + cfg.SAA_N2
    
    # Convert OptionMatrix to DataFrame for generate_empirical_scenarios
    options_matrix = ensure_option_matrix(order_options)
    order_options_df = options_matrix.to_dataframe()
    
    # Generate empirical scenarios
    demand_scenarios, shipping_costs = generate_empirical_scenarios(
        order_items=order_items,
        order_options=order_options_df,
        promise_days=promise_days,
        num_scenarios=total_scenarios_needed,
        order_set=order_set,
        lookahead_periods=lookahead_periods,
        seed=cfg.RANDOM_SEED + order_idx,
        verbose=verbose,
    )
    
    # Run SAA procedure with empirical scenarios
    plan_df, runtime, saa_timings = run_saa_procedure(
        order_items=order_items,
        options_df=order_options,
        on_hand_inventory_pivot=on_hand_inventory_pivot.copy(),
        customer_dc=customer_dc,
        promise_days=promise_days,
        stockout_penalty=cfg.STOCKOUT_PENALTY_PER_UNIT,
        order_set=order_set,
        master_seed=cfg.RANDOM_SEED + order_idx,
        demand_scenarios=demand_scenarios,
        shipping_costs=shipping_costs,
        verbose=verbose or logger.isEnabledFor(logging.DEBUG),  # Enable verbose if DEBUG logging
        use_future_penalties=bool(getattr(cfg, "SAA_USE_FUTURE_PENALTIES", False)),
    )
    
    metadata = {
        'runtime_seconds': runtime,
        'saa_n1': cfg.SAA_N1,
        'saa_n2': cfg.SAA_N2,
        'saa_q': cfg.SAA_Q,
        'source': 'empirical',
        'lookahead_periods': lookahead_periods,
        'use_future_penalties': bool(getattr(cfg, "SAA_USE_FUTURE_PENALTIES", False)),
        'timings': {'saa_details': saa_timings},
    }
    
    return plan_df, metadata


def create_policy_for_simulation(catalog, precompute, state, order_set, 
                                  simulation_date, **kwargs):
    """
    Create Empirical SAA policy closure for simulation.
    
    Args:
        catalog: OptionsCatalog instance
        precompute: PrecomputeStore instance
        state: SimulationState instance
        order_set: Order set ('test' or 'proxy_train')
        simulation_date: Simulation date string (YYYY-MM-DD)
        **kwargs: Additional unused arguments for API consistency
    
    Returns:
        Policy function that takes an Order and returns (OrderDecision, runtime_seconds)
    """
    import time
    import pandas as pd
    from src.simulator.entities import OrderDecision, ItemAllocation
    from src.simulator.features import build_costs
    
    order_counter = [0]  # Mutable counter for order_idx
    
    def policy(order):
        """Empirical SAA policy using empirical scenario distributions."""
        start_time = time.perf_counter()
        
        option_ids = catalog.eligible_for_order(order)
        if not option_ids:
            return OrderDecision(
                allocations=[],
                unfilled={item.sku_id: item.quantity for item in order.items}
            ), 0.0
        
        # Build costs
        costs_series = build_costs(order, option_ids, catalog, precompute)
        
        # Prepare order items using shared utility
        fallback_time = order.order_time or pd.Timestamp(simulation_date)
        order_items, unique_skus, original_qty = prepare_order_items(order, order.items, fallback_time)
        
        # Build inventory pivot using shared utility
        on_hand_inventory_pivot = build_inventory_pivot(unique_skus, option_ids, state.inventory)
        
        # Build order_options using shared utility
        order_options = build_option_matrix(option_ids, costs_series)
        
        # Build order_info Series
        order_info = pd.Series({
            'order_ID': order.order_id,
            'order_time': fallback_time,
            'customer_zip5': order.dest_zip5,
            'customer_state': order.dest_state,
            'customer_lat': order.dest_lat,
            'customer_lon': order.dest_lng,
            'promise_delivery_days': order.promise_delivery_days,
            'dc_des': order.customer_dc,
        })
        
        # Get order index for seeding
        order_idx = order_counter[0]
        order_counter[0] += 1
        
        # Call Empirical SAA fulfillment
        plan_df, metadata = empirical_saa_fulfillment(
            order_info=order_info,
            order_items=order_items,
            on_hand_inventory_pivot=on_hand_inventory_pivot,
            order_options=order_options,
            order_set=order_set,
            simulation_date=simulation_date,
            order_idx=order_idx,
            verbose=False,
        )

        if logger.isEnabledFor(logging.DEBUG):
            # Essential diagnostics: inventory availability and cheapest feasible options
            inv_max = 0.0
            cheap_opts_with_inv = []
            try:
                if not on_hand_inventory_pivot.empty:
                    inv_max = float(on_hand_inventory_pivot.to_numpy().max())
                    # Find cheapest options that actually have inventory
                    series = costs_series.dropna()
                    for (dc, cs), c in series.nsmallest(5).items():
                        dc_id = int(dc)
                        if not on_hand_inventory_pivot.empty:
                            sku_list = on_hand_inventory_pivot.index.tolist()
                            if sku_list and dc_id in on_hand_inventory_pivot.columns:
                                inv_qty = float(on_hand_inventory_pivot.loc[sku_list[0], dc_id])
                                if inv_qty > 0:
                                    cheap_opts_with_inv.append((dc_id, int(cs), float(c), inv_qty))
            except Exception:
                pass
            
            logger.debug(
                "[empirical_saa] order=%s idx=%d options=%d inv_max=%.1f cheapest_with_inv=%s plan_rows=%d",
                order.order_id,
                order_idx,
                len(option_ids),
                inv_max,
                cheap_opts_with_inv[:3] if cheap_opts_with_inv else [],  # Top 3 cheapest with inventory
                0 if plan_df.empty else len(plan_df),
            )
        
        # Convert plan to OrderDecision
        allocations = []
        unfilled = {}
        
        if not plan_df.empty:
            for _, row in plan_df.iterrows():
                sku_id = str(row['sku_ID'])
                dc_id = row.get('dc_ori')
                carrier_id = row.get('carrier_service_id')
                qty = row.get('quantity', 0)
                
                if pd.notna(dc_id) and pd.notna(carrier_id) and qty > 0:
                    allocations.append(ItemAllocation(
                        sku_id=sku_id,
                        option_id=(int(dc_id), int(carrier_id)),
                        quantity=int(qty),
                    ))
        
        # Calculate unfilled quantities
        allocated_qty = {}
        for alloc in allocations:
            allocated_qty[alloc.sku_id] = allocated_qty.get(alloc.sku_id, 0) + alloc.quantity
        
        for sku_id, orig_qty in original_qty.items():
            alloc = allocated_qty.get(sku_id, 0)
            if alloc < orig_qty:
                unfilled[sku_id] = orig_qty - alloc
        
        runtime = time.perf_counter() - start_time
        return OrderDecision(
            allocations=allocations,
            unfilled=unfilled if unfilled else None
        ), runtime
    
    return policy
