"""Shared utilities for optimization algorithms (contextual_saa, empirical_saa, pto, etc.)."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from src.data_structures.options import OptionMatrix


def prepare_order_items(order, items, fallback_order_time=None):
    """
    Convert order items to DataFrame format required by SAA procedures.
    
    Args:
        order: Order object
        items: List of OrderItem objects
        fallback_order_time: Optional fallback timestamp if order.order_time is None
        
    Returns:
        Tuple of (order_items_df, unique_skus_array, original_qty_map)
    """
    if not items:
        empty_df = pd.DataFrame(columns=['sku_ID', 'quantity', 'order_ID', 'order_date', 'order_time'])
        return empty_df, np.array([], dtype=str), {}

    sku_ids = np.array([str(item.sku_id) for item in items], dtype=str)
    quantities = np.array([int(item.quantity) for item in items], dtype=int)

    order_id = getattr(order, 'order_id', None)
    order_time_value = getattr(order, 'order_time', None) or fallback_order_time
    order_ts = pd.Timestamp(order_time_value) if order_time_value is not None else pd.Timestamp.now()
    order_date = order_ts.normalize()

    order_items_df = pd.DataFrame({
        'sku_ID': sku_ids,
        'quantity': quantities,
    })
    order_items_df = order_items_df.assign(
        order_ID=str(order_id) if order_id is not None else pd.NA,
        order_time=order_ts,
        order_date=order_date,
    )

    unique_skus = np.unique(sku_ids)
    original_qty = {sku: int(qty) for sku, qty in zip(sku_ids, quantities)}
    return order_items_df, unique_skus, original_qty


def build_inventory_pivot(unique_skus, option_ids, inventory_snapshot):
    """
    Construct dense inventory matrix using NumPy, then wrap in DataFrame.
    
    The resulting DataFrame is indexed by SKU (rows) with DC IDs as columns,
    which is the format expected by the SAA optimization model.
    
    Args:
        unique_skus: Array of unique SKU IDs (strings)
        option_ids: List of (dc_id, carrier_id) tuples
        inventory_snapshot: Dict mapping dc_id -> {sku_id: quantity}
        
    Returns:
        DataFrame with index=sku_IDs, columns=dc_ids, values=inventory quantities
    """
    if unique_skus.size == 0 or not option_ids:
        return pd.DataFrame()
    
    dc_ids = np.array(sorted({int(opt[0]) for opt in option_ids}), dtype=int)
    num_skus = unique_skus.size
    num_dcs = dc_ids.size
    inv_matrix = np.zeros((num_skus, num_dcs), dtype=float)
    sku_index = {sku: idx for idx, sku in enumerate(unique_skus)}
    dc_index = {dc: idx for idx, dc in enumerate(dc_ids)}
    
    for dc_key, sku_map in inventory_snapshot.items():
        try:
            dc_id = int(dc_key)
        except (TypeError, ValueError):
            continue
        dc_idx = dc_index.get(dc_id)
        if dc_idx is None:
            continue
        for sku_id, qty in sku_map.items():
            sku = str(sku_id)
            row_idx = sku_index.get(sku)
            if row_idx is None:
                continue
            inv_matrix[row_idx, dc_idx] = float(qty)
    
    return pd.DataFrame(inv_matrix, index=unique_skus.tolist(), columns=dc_ids.tolist())


def build_option_matrix(option_ids: List[Tuple[int, int]], costs_series: pd.Series) -> OptionMatrix:
    """
    Build NumPy-backed OptionMatrix from option IDs and costs.
    
    Args:
        option_ids: List of (dc_id, carrier_service_id) tuples
        costs_series: Series indexed by option_id with base costs
        
    Returns:
        OptionMatrix instance
    """
    num_opts = len(option_ids)
    option_idx = np.arange(num_opts, dtype=int)
    dc_ids = np.fromiter((int(opt[0]) for opt in option_ids), dtype=int, count=num_opts)
    carrier_ids = np.fromiter((int(opt[1]) for opt in option_ids), dtype=int, count=num_opts)
    base_costs = np.array([float(costs_series.get(opt, 0.0)) for opt in option_ids], dtype=float)
    return OptionMatrix(
        option_ids=option_idx,
        dc_ids=dc_ids,
        carrier_service_ids=carrier_ids,
        base_costs=base_costs,
    )

