"""Feature builder for fulfillment options."""

from typing import List, Optional
from functools import lru_cache
import pandas as pd
import numpy as np

import src.config as cfg
from src.simulator.entities import Order, OptionId
from src.simulator.precompute import PrecomputeStore
from src.simulator.catalog import OptionsCatalog
from src.simulator.state import SimulationState
from src.data_utils import compute_haversine_km, get_per_unit_base_cost


def build_features(
    order: Order,
    option_ids: List[OptionId],
    state: SimulationState,
    catalog: OptionsCatalog,
    precompute: PrecomputeStore,
) -> pd.DataFrame:
    """
    Build features DataFrame for fulfillment options.
    
    Args:
        order: The order
        option_ids: List of option IDs to build features for
        state: Current simulation state
        catalog: Options catalog
        precompute: Precompute store
        
    Returns:
        DataFrame indexed by option_id with features
    """
    # Start with option-level data
    rows = []
    for opt_id in option_ids:
        opt = catalog.index[opt_id]
        row = {
            'option_id': opt_id,
            'dc_id': opt.dc_id,
            'carrier_service_id': opt.carrier_service_id,
        }
        rows.append(row)

    features_df = pd.DataFrame(rows)

    # Add static features (order-level, same for all options)
    static_features = precompute.get_static_features(order.order_id) or {}

    # Add option-specific features first (dc_ori differs by option)
    features_df['dc_ori'] = features_df['dc_id']  # Each option has its own dc_id
    dest_dc = (
        order.customer_dc
        if order.customer_dc is not None
        else static_features.get('dc_des', cfg.DELIVERY_DL_UNKNOWN_TOKEN)
    )
    features_df['dc_des'] = dest_dc

    for feat in cfg.STATIC_FEATURES:
        if feat == 'dc_des':
            continue
        if feat in static_features:
            features_df[feat] = static_features[feat]
        else:
            features_df[feat] = 0.0

    # Add option-static features (distance, service flags)
    distances = []
    for opt_id in option_ids:
        opt = catalog.index[opt_id]
        dist = precompute.get_distance_km(opt.dc_id, order.dest_zip5)
        if dist is None:
            # Compute on the fly
            dist = _compute_distance(opt.dc_lat, opt.dc_lng, order.dest_lat, order.dest_lng)
        distances.append(dist)

    features_df['distance_km'] = distances
    
    # Add option static flags if available
    for opt_id in option_ids:
        opt_static = precompute.get_option_static(opt_id)
        if opt_static:
            for key, val in opt_static.items():
                if key not in features_df.columns:
                    features_df.loc[features_df['option_id'] == opt_id, key] = val
    
    # Add dynamic features from DC snapshot
    dc_snapshot = state.build_dc_event_snapshot(state.now)
    
    # Join DC-level dynamic features
    for feat in cfg.DYNAMIC_FEATURES:
        if feat in dc_snapshot.columns:
            # Map DC-level features to options
            dc_feat_map = dc_snapshot.set_index('dc_id')[feat].to_dict()
            features_df[feat] = features_df['dc_id'].map(dc_feat_map).fillna(0.0)
        else:
            features_df[feat] = 0.0
    
    # Ensure dc_ori is set correctly per option (override any static feature that might have set it)
    features_df['dc_ori'] = features_df['dc_id']
    
    # Set option_id as index
    features_df.set_index('option_id', inplace=True)
    
    return features_df


@lru_cache(maxsize=1)
def _load_carrier_cost_coefficients() -> dict[int, float]:
    try:
        cost_models_df = pd.read_csv(cfg.REAL_COST_MODELS_CS_PATH)
    except FileNotFoundError:
        return {}
    if {'carrier_service_id', 'coef_distance_km'} - set(cost_models_df.columns):
        return {}
    return (
        cost_models_df[['carrier_service_id', 'coef_distance_km']]
        .dropna()
        .drop_duplicates('carrier_service_id')
        .set_index('carrier_service_id')['coef_distance_km']
        .astype(float)
        .to_dict()
    )


def build_costs(
    order: Order,
    option_ids: List[OptionId],
    catalog: OptionsCatalog,
    precompute: PrecomputeStore,
) -> pd.Series:
    """
    Build cost Series for fulfillment options.
    
    Args:
        order: The order
        option_ids: List of option IDs
        catalog: Options catalog
        precompute: Precompute store
        
    Returns:
        Series indexed by option_id with base costs
    """
    costs = []
    cost_coef_map = _load_carrier_cost_coefficients()
    
    for opt_id in option_ids:
        opt = catalog.index[opt_id]
        
        # Try precomputed cost first
        variable_cost = precompute.get_cost(opt.dc_id, opt.carrier_service_id, order.dest_zip5)
        
        if variable_cost is None:
            # Compute from distance
            dist = precompute.get_distance_km(opt.dc_id, order.dest_zip5)
            if dist is None:
                dist = _compute_distance(opt.dc_lat, opt.dc_lng, order.dest_lat, order.dest_lng)
            
            coef = cost_coef_map.get(opt.carrier_service_id, 0.0)
            variable_cost = dist * coef if coef > 0 else 0.0

        fixed_cost = get_per_unit_base_cost(opt.dc_id)
        total_cost = float(variable_cost or 0.0) + fixed_cost
        costs.append(total_cost)
    
    return pd.Series(costs, index=option_ids, name='base_cost')


def build_dc_region_cost_matrix(
    catalog: OptionsCatalog,
    precompute: PrecomputeStore,
    region_zip_weights: dict,
    zip_coords: Optional[dict] = None,
    dc_ids: Optional[List[int]] = None,
    max_zips_per_region: int = 20,
) -> pd.DataFrame:
    """Build a (DC x destination-region) per-unit shipping cost matrix from the
    live cost model.

    This uses the same primitives as :func:`build_costs` (precomputed
    cost/distance lookups, carrier distance coefficients, per-DC base cost), so
    a policy that formulates a transportation LP over this matrix is on the same
    cost basis as every policy that prices options through ``build_costs``.

    Args:
        catalog: Options catalog — source of the DC universe and the
            (dc_id, carrier_service_id) options.
        precompute: Precompute store — live cost/distance lookups.
        region_zip_weights: ``{region_key: {zip5: weight}}`` giving representative
            destination ZIPs per region with order-count weights. Keys must
            already be normalized the way the order data presents them
            (region and zip cast to ``str``), so lookups match ``build_costs``.
        zip_coords: Optional ``{zip5: (lat, lon)}`` used for the haversine
            distance fallback (mirrors build_costs) when a (DC, ZIP) pair is
            absent from the precomputed distance table. ZIP keys cast to ``str``.
        dc_ids: Explicit DC universe; defaults to all DCs in the catalog.
        max_zips_per_region: Cap on representative ZIPs per region (highest
            weight first) to bound setup cost.

    Returns:
        DataFrame indexed by ``dc_id`` (index name ``dc_ori``), columns = region
        keys (str), values = order-weighted mean cheapest-option per-unit cost
        from the DC to the region. Unknown (DC, region) pairs are ``np.inf``.
    """
    cost_coef_map = _load_carrier_cost_coefficients()
    # Whether a precomputed unit-cost table is loaded. When it is not (the
    # current sim configuration), build_costs reduces to distance * coef, so we
    # can skip per-carrier get_cost lookups entirely.
    has_cost_table = getattr(precompute, '_costs', None) is not None
    zip_coords = zip_coords or {}

    carriers_by_dc: dict[int, set] = {}
    dc_coords: dict[int, tuple] = {}
    for opt in catalog.all_options:
        carriers_by_dc.setdefault(opt.dc_id, set()).add(opt.carrier_service_id)
        dc_coords.setdefault(opt.dc_id, (opt.dc_lat, opt.dc_lng))
    if dc_ids is None:
        dc_ids = sorted(carriers_by_dc)
    regions = sorted({str(r) for r in region_zip_weights})

    def _distance_km(dc_id: int, zip5: str) -> Optional[float]:
        d = precompute.get_distance_km(dc_id, zip5)
        if d is not None:
            return float(d)
        # Haversine fallback (mirrors build_costs) from DC and ZIP coordinates.
        dcc = dc_coords.get(dc_id)
        zc = zip_coords.get(zip5)
        if dcc and zc and None not in dcc and None not in zc:
            return _compute_distance(dcc[0], dcc[1], zc[0], zc[1])
        return None

    def _cheapest_cost(dc_id: int, zip5: str) -> Optional[float]:
        # Cheapest-carrier variable cost to (dc_id, zip5), mirroring the per-option
        # logic in build_costs (precomputed cost, else distance * coef; coef<=0
        # contributes 0.0 when a distance is available).
        carriers = carriers_by_dc.get(dc_id) or ()
        dist = _distance_km(dc_id, zip5)
        best = None
        for carrier in carriers:
            vc = precompute.get_cost(dc_id, carrier, zip5) if has_cost_table else None
            if vc is None:
                if dist is None:
                    continue
                coef = cost_coef_map.get(carrier, 0.0)
                vc = dist * coef if coef > 0 else 0.0
            vc = float(vc)
            if best is None or vc < best:
                best = vc
        return best

    matrix = pd.DataFrame(
        np.inf,
        index=pd.Index(dc_ids, name='dc_ori'),
        columns=regions,
        dtype=float,
    )
    for region in regions:
        zw = region_zip_weights.get(region, {})
        top_zips = sorted(zw.items(), key=lambda kv: kv[1], reverse=True)[:max_zips_per_region]
        if not top_zips:
            continue
        for dc_id in dc_ids:
            base = get_per_unit_base_cost(dc_id)
            num = den = 0.0
            for zip5, weight in top_zips:
                best = _cheapest_cost(dc_id, str(zip5))
                if best is not None:
                    num += weight * (best + base)
                    den += weight
            if den > 0:
                matrix.at[dc_id, region] = num / den
    return matrix


def _compute_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in km."""
    from src.data_utils import compute_haversine_km
    return float(compute_haversine_km(
        np.array([lat1]), np.array([lon1]), lat2, lon2
    )[0])
