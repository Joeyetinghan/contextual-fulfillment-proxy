import pandas as pd
import time


def _fallback_base_costs(order_items, on_hand_inventory_pivot, base_costs):
    """Legacy greedy allocation using DC-level base costs only."""
    sorted_dcs = base_costs.sort_values().index
    inventory = on_hand_inventory_pivot.copy()
    plan = []

    for sku, qty in order_items[['sku_ID', 'quantity']].values:
        assigned = False
        if sku in inventory.index:
            valid_dcs = sorted_dcs[inventory.loc[sku, sorted_dcs] >= qty]
            if len(valid_dcs):
                dc = valid_dcs[0]
                inventory.loc[sku, dc] -= qty
                plan.append({
                    'sku_ID': sku,
                    'dc_ori': dc,
                    'carrier_service_id': None,
                    'quantity': qty,
                })
                assigned = True
        if not assigned:
            plan.append({
                'sku_ID': sku,
                'dc_ori': None,
                'carrier_service_id': None,
                'quantity': 0,
            })

    return pd.DataFrame(plan)


def greedy_fulfillment(order_items, on_hand_inventory_pivot, order_options=None, base_costs=None):
    """Greedy baseline using option-level costs when available."""
    if order_options is None or order_options.empty:
        if base_costs is None:
            raise ValueError("base_costs is required when order_options is None or empty.")
        return _fallback_base_costs(order_items, on_hand_inventory_pivot, base_costs)

    inventory = on_hand_inventory_pivot.copy()
    options_sorted = order_options.sort_values('base_cost').reset_index(drop=True)
    plan = []

    for _, row in order_items.iterrows():
        sku = str(row['sku_ID'])  # Ensure SKU is a string
        qty = float(row['quantity'])
        assigned = False
        if sku in inventory.index:
            for _, opt in options_sorted.iterrows():
                dc = int(opt['dc_id'])
                carrier = int(opt['carrier_service_id'])
                if dc in inventory.columns and inventory.loc[sku, dc] >= qty:
                    inventory.loc[sku, dc] -= qty
                    plan.append({
                        'sku_ID': sku,
                        'dc_ori': dc,
                        'carrier_service_id': carrier,
                        'quantity': qty,
                    })
                    assigned = True
                    break
        if not assigned:
            plan.append({
                'sku_ID': sku,
                'dc_ori': None,
                'carrier_service_id': None,
                'quantity': 0,
            })

    return pd.DataFrame(plan)


def greedy_unit_fulfillment(order_items, on_hand_inventory_pivot, order_options=None, base_costs=None):
    """Per-unit greedy assignment using option-level costs when available."""
    if order_options is None or order_options.empty:
        if base_costs is None:
            raise ValueError("base_costs is required when order_options is None or empty.")
        # Reuse fallback but expand quantities unit by unit
        base_plan = _fallback_base_costs(order_items, on_hand_inventory_pivot, base_costs)
        return base_plan

    inventory = on_hand_inventory_pivot.copy()
    options_sorted = order_options.sort_values('base_cost').reset_index(drop=True)
    assignments: dict[tuple, int] = {}

    for sku, qty in order_items[['sku_ID', 'quantity']].values:
        remaining = int(qty)
        if sku not in inventory.index:
            continue
        while remaining > 0:
            allocated = False
            for _, opt in options_sorted.iterrows():
                dc = int(opt['dc_id'])
                carrier = int(opt['carrier_service_id'])
                if dc not in inventory.columns:
                    continue
                if inventory.loc[sku, dc] >= 1:
                    inventory.loc[sku, dc] -= 1
                    assignments[(sku, dc, carrier)] = assignments.get((sku, dc, carrier), 0) + 1
                    remaining -= 1
                    allocated = True
                    break
            if not allocated:
                break

    rows = [
        {
            'sku_ID': sku,
            'dc_ori': dc,
            'carrier_service_id': carrier,
            'quantity': qty,
        }
        for (sku, dc, carrier), qty in assignments.items()
    ]
    return pd.DataFrame(rows)


def choose_fulfillment_option_level(
    order,
    items,
    option_ids,
    eligible_mask,
    features_df,
    costs_series,
    inventory_snapshot,
):
    """
    Greedy fulfillment with order splitting across multiple DCs.
    
    Iterates through options sorted by cost and allocates as much inventory as possible
    from each option until the order is fully satisfied or inventory is exhausted.
    
    Args:
        order: Order object (unused, kept for API consistency)
        items: List of OrderItem objects
        option_ids: List of OptionId tuples (dc_id, carrier_service_id)
        eligible_mask: Boolean mask (unused, kept for API consistency)
        features_df: Option-indexed features DataFrame (unused, kept for API consistency)
        costs_series: Dict or Series mapping OptionId to cost
        inventory_snapshot: Dict mapping dc_id -> {sku_id: quantity}
        
    Returns:
        Tuple of (OrderDecision, runtime_seconds)
    """
    from src.simulator.entities import OrderDecision, ItemAllocation
    
    start_time = time.perf_counter()
    
    # Precompute useful structures to avoid repeated pandas allocations per order
    relevant_skus = {str(item.sku_id) for item in items}
    option_cost_pairs = sorted(
        ((opt_id, float(costs_series.get(opt_id, 0.0))) for opt_id in option_ids),
        key=lambda pair: pair[1],
    )
    sorted_options = [pair[0] for pair in option_cost_pairs]
    
    # Copy only the inventory entries we need for this order
    per_dc_inventory = {}
    for dc_id, sku_map in inventory_snapshot.items():
        filtered = {
            str(sku_id): float(qty)
            for sku_id, qty in sku_map.items()
            if str(sku_id) in relevant_skus and qty > 0
        }
        if filtered:
            try:
                dc_key = int(dc_id)
            except (TypeError, ValueError):
                dc_key = dc_id
            per_dc_inventory[dc_key] = filtered
    
    # Build allocations greedily without going through pandas DataFrames
    allocations = []
    unfilled = {}
    
    for item in items:
        sku_id = str(item.sku_id)
        remaining_qty = int(item.quantity)
        
        for option_dc, option_carrier in sorted_options:
            if remaining_qty <= 0:
                break

            dc_inventory = per_dc_inventory.get(option_dc)
            if not dc_inventory:
                continue
            available = dc_inventory.get(sku_id, 0.0)
            
            if available > 0:
                take_qty = min(remaining_qty, int(available))
                dc_inventory[sku_id] = available - take_qty
                
                allocations.append(ItemAllocation(
                    sku_id=sku_id,
                    option_id=(int(option_dc), int(option_carrier)),
                    quantity=take_qty,
                ))
                remaining_qty -= take_qty
        
        if remaining_qty > 0:
            unfilled[sku_id] = unfilled.get(sku_id, 0) + remaining_qty
    
    runtime = time.perf_counter() - start_time
    decision = OrderDecision(allocations=allocations, unfilled=unfilled if unfilled else None)
    return decision, runtime


def create_policy_for_simulation(catalog, precompute, state, **kwargs):
    """
    Create greedy policy closure for simulation.
    
    Args:
        catalog: OptionsCatalog instance
        precompute: PrecomputeStore instance
        state: SimulationState instance
        **kwargs: Additional unused arguments for API consistency
    
    Returns:
        Policy function that takes an Order and returns (OrderDecision, runtime_seconds)
    """
    from src.simulator.features import build_costs
    from src.simulator.entities import OrderDecision
    
    def policy(order):
        """Greedy policy that selects cheapest available option."""
        option_ids = catalog.eligible_for_order(order)
        if not option_ids:
            return OrderDecision(
                allocations=[],
                unfilled={item.sku_id: item.quantity for item in order.items}
            ), 0.0
        
        costs_series = build_costs(order, option_ids, catalog, precompute)
        return choose_fulfillment_option_level(
            order=order,
            items=order.items,
            option_ids=option_ids,
            eligible_mask=None,
            features_df=None,
            costs_series=costs_series,
            inventory_snapshot=state.inventory,
        )
    
    return policy
