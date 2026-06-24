import time
from collections import defaultdict
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise ImportError("gurobipy is required for dtlp_bidprice; install and license Gurobi.") from e

# Optional PTO import for forecast-driven d_daily
try:
    from src.algo.pto import generate_single_scenario
    from src.utils import calculate_lookahead_periods
    PTO_AVAILABLE = True
except Exception:
    PTO_AVAILABLE = False

from src.data_structures.options import OptionMatrix, ensure_option_matrix


class DtlpBidPriceState:
    """
    Persistent cache for DTLP bid-price duals (per SKU) with configurable refresh policy.

    Params:
      - solve_cadence: 'per_order' | 'inventory_buckets' | 'every_n_orders'
      - q: positive int bucket size for 'inventory_buckets' (paper-like CEIL(||X||/q))
      - n_orders: positive int for 'every_n_orders' cadence
      - eps: float tolerance for numeric change detection

    Per-SKU cache entry:
      - mu_by_dc: pd.Series of duals indexed by DC
      - bucket: int inventory bucket at last solve
      - tau_days: float tau at last solve
      - phi_sig: tuple signature of phi_jm (index, rounded values)
      - costs_sig: tuple signature of costs_ijm (rows, cols)
      - dc_set: tuple of DC IDs observed in X_tau_by_fc at last solve
      - orders_since_solve: int counter
    """

    def __init__(self, *, solve_cadence: str = 'per_order', q: int = 100, n_orders: int = 1, eps: float = 1e-6):
        if solve_cadence not in ('per_order', 'inventory_buckets', 'every_n_orders'):
            raise ValueError("solve_cadence must be one of {'per_order','inventory_buckets','every_n_orders'}")
        self.solve_cadence: str = solve_cadence
        self.q: int = int(q) if int(q) > 0 else 1
        self.n_orders: int = int(n_orders) if int(n_orders) > 0 else 1
        self.eps: float = float(eps)
        self._per_sku: Dict[Any, Dict[str, Any]] = {}

    @staticmethod
    def _phi_signature(phi_jm: pd.Series, round_decimals: int = 6) -> Tuple[Tuple[Any, ...], Tuple[float, ...]]:
        idx = tuple(list(phi_jm.index))
        vals = tuple([float(v) if np.isfinite(v) else np.inf for v in np.round(np.asarray(phi_jm.values, dtype=float), round_decimals)])
        return idx, vals

    @staticmethod
    def _costs_signature(costs_ijm: pd.DataFrame) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        return tuple(list(costs_ijm.index)), tuple(list(costs_ijm.columns))

    def _should_solve(self, sku: Any, X_tau_by_fc: pd.Series, tau_days: float, phi_jm: pd.Series, costs_ijm: pd.DataFrame) -> bool:
        # No cache means that must solve
        if sku not in self._per_sku:
            return True
        entry = self._per_sku[sku]
        # Core-change checks
        cur_phi_sig = self._phi_signature(phi_jm)
        cur_costs_sig = self._costs_signature(costs_ijm)
        cur_dc_set = tuple(list(X_tau_by_fc.index))
        if abs(float(entry['tau_days']) - float(tau_days)) > self.eps:
            return True
        if entry['phi_sig'] != cur_phi_sig:
            return True
        if entry['costs_sig'] != cur_costs_sig:
            return True
        if entry['dc_set'] != cur_dc_set:
            return True
        # Cadence policy
        if self.solve_cadence == 'per_order':
            return True
        if self.solve_cadence == 'every_n_orders':
            return int(entry['orders_since_solve']) >= (self.n_orders)
        if self.solve_cadence == 'inventory_buckets':
            total_supply = float(np.nansum(np.asarray(X_tau_by_fc.values, dtype=float)))
            bucket = int(np.ceil(max(0.0, total_supply) / float(self.q)))
            return int(entry.get('bucket', -1)) != bucket
        # Fallback
        return True

    def update_entry(self, sku: Any, mu_by_dc: pd.Series, X_tau_by_fc: pd.Series, tau_days: float, phi_jm: pd.Series, costs_ijm: pd.DataFrame):
        total_supply = float(np.nansum(np.asarray(X_tau_by_fc.values, dtype=float)))
        bucket = int(np.ceil(max(0.0, total_supply) / float(self.q)))
        self._per_sku[sku] = {
            'mu_by_dc': mu_by_dc.astype(float),
            'bucket': bucket,
            'tau_days': float(tau_days),
            'phi_sig': self._phi_signature(phi_jm),
            'costs_sig': self._costs_signature(costs_ijm),
            'dc_set': tuple(list(X_tau_by_fc.index)),
            'orders_since_solve': 0,
        }

    def get_mu(self, sku: Any) -> pd.Series | None:
        if sku not in self._per_sku:
            return None
        return self._per_sku[sku]['mu_by_dc']

    def tick(self, sku: Any):
        if sku in self._per_sku:
            self._per_sku[sku]['orders_since_solve'] = int(self._per_sku[sku].get('orders_since_solve', 0)) + 1


def _solve_dtlp_duals(
    X_tau_by_fc: pd.Series,
    d_daily: float,
    tau_days: float,
    phi_jm: pd.Series,
    costs_ijm: pd.DataFrame,
) -> Tuple[List[Any], np.ndarray]:
    """
    Solve the deterministic transportation LP (DTLP) for a given SKU to obtain
    supply dual prices (bid prices) by DC.

    Formulation (per SKU):
      - Variables: x[(dc, (region, speed))] ≥ 0 are the units shipped from dc to
        demand class (region, speed).
      - Objective: minimize sum_{dc, jm} c_ijm * x_ijm where c_ijm is shipping cost.
      - Supply constraints: for each dc, sum_jm x_ijm ≤ X^tau_i (on-hand over horizon τ).
      - Demand constraints: for each jm, sum_i x_ijm = phi_jm * d_daily * tau_days.

    If total supply < required demand, τ is reduced so the LP remains feasible.

    Args:
        X_tau_by_fc: Series indexed by dc (supply over τ for the SKU).
        d_daily: Effective daily demand rate for the SKU.
        tau_days: Planning horizon in days (may be reduced if infeasible).
        phi_jm: Series over MultiIndex (dc_des, speed) that sums to 1.
        costs_ijm: DataFrame with index=dc and columns=(dc_des, speed) giving c_ijm.

    Returns:
        (dcs_with_positive_supply, pis)
        dcs_with_positive_supply: list of dc IDs included in the LP.
        pis: numpy array of dual prices for supply constraints in the same order as dcs.
    """
    # Sanity
    if d_daily < 0 or not np.isfinite(d_daily):
        raise ValueError(f"d_daily must be >= 0 (got {d_daily})")
    if tau_days <= 0 or not np.isfinite(tau_days):
        raise ValueError(f"tau_days must be > 0 (got {tau_days})")
    s = float(phi_jm.sum())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"phi_jm must sum to 1 (got {s})")

    total_supply = float(X_tau_by_fc.sum())
    rhs_total = float(d_daily * tau_days)
    # Shrink τ if necessary to keep LP feasible when supply < total demand over τ
    if total_supply < rhs_total and d_daily > 0:
        tau_days = total_supply / max(d_daily, 1e-12)
        rhs_total = float(d_daily * tau_days)

    dcs = [dc for dc, s in X_tau_by_fc.items() if float(s) > 0]
    if not dcs:
        return dcs, np.zeros((0,), dtype=float)

    jm_list = [jm for jm, v in phi_jm.items() if float(v) > 0]
    missing_cols = [jm for jm in jm_list if jm not in costs_ijm.columns]
    if missing_cols:
        raise KeyError(f"costs_ijm missing columns for (region,speed): {missing_cols}")

    m = gp.Model(); m.Params.OutputFlag = 0

    # Variables: x[(dc,jm)]
    x: Dict[Tuple[Any, Any], gp.Var] = {}
    for dc in dcs:
        for jm in jm_list:
            cijm = costs_ijm.loc[dc, jm]
            if not np.isfinite(cijm):
                continue
            x[(dc, jm)] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x[{dc},{jm}]")
    m.update()

    # Objective: minimize shipping cost
    obj = gp.LinExpr()
    for (dc, jm), var in x.items():
        obj.add(var, float(costs_ijm.loc[dc, jm]))
    m.setObjective(obj, GRB.MINIMIZE)

    # Supply constraints: sum_jm x_ijm <= X^tau_i
    supply_cons = []
    for dc in dcs:
        expr = gp.LinExpr()
        for (dc2, jm), var in x.items():
            if dc2 == dc:
                expr.add(var, 1.0)
        supply_cons.append(m.addConstr(expr <= float(X_tau_by_fc.loc[dc]), name=f"sup[{dc}]"))

    # Demand constraints: sum_i x_ijm == phi_jm * d * tau
    for jm in jm_list:
        rhs = float(phi_jm.loc[jm]) * float(d_daily * tau_days)
        expr = gp.LinExpr()
        for (dc, jm2), var in x.items():
            if jm2 == jm:
                expr.add(var, 1.0)
        if rhs > 0 and expr.size() == 0:
            raise ValueError(f"No arcs for demand at jm={jm} with RHS {rhs}")
        if expr.size() > 0:
            m.addConstr(expr == rhs, name=f"dem[{jm}]")

    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi DTLP not optimal: status={m.Status}")

    pis = np.array([con.Pi for con in supply_cons], dtype=float)
    return dcs, pis


def _forecast_d_daily_from_pto(
    order_info: pd.Series,
    order_items: pd.DataFrame,
    options_df: pd.DataFrame | None,
    order_set: str,
    simulation_date: str,
    verbose: bool,
    sku: Any,
    dynamic_features: pd.DataFrame | None = None,
) -> float:
    """
    Estimate per-SKU daily demand rate using PTO's single-scenario forecast.

    Steps:
      1) Compute lookahead_periods given order time and simulation date.
      2) Call generate_single_scenario to obtain total forecasted units over that horizon.
      3) Convert to a daily rate: d_daily_eff = total_units / max(1, lookahead_periods).

    Returns:
      - 0.0 if the forecast indicates zero demand for the SKU.
      - None if PTO is unavailable or there is an exception (to trigger fallback).
    """
    if not PTO_AVAILABLE:
        return None
    try:
        if options_df is None or options_df.empty:
            # New PTO path requires option-level df; fall back to global d_daily.
            return None
        lookahead_periods = calculate_lookahead_periods(order_info['order_time'], simulation_date)
        period = 'proxy' if order_set == 'proxy_train' else 'test'
        demand_median, median_costs = generate_single_scenario(
            order_info=order_info,
            order_items=order_items,
            options_df=options_df,
            period=period,
            lookahead_periods=lookahead_periods,
            verbose=verbose,
            dynamic_features=dynamic_features,
        )
        # Pull per-SKU total units from PTO output
        total_units = 0.0
        if isinstance(demand_median, dict):
            val = demand_median.get(sku, 0)
        else:
            val = demand_median
        try:
            total_units = float(np.nansum(np.asarray(val)))
        except Exception:
            try:
                total_units = float(np.nansum(pd.DataFrame(val).values))
            except Exception:
                try:
                    total_units = float(val)
                except Exception:
                    total_units = 0.0
        try:
            eff_tau_days = max(1.0, float(lookahead_periods))
        except Exception:
            eff_tau_days = 1.0
        if total_units <= 0:
            return 0.0
        return float(total_units) / eff_tau_days
    except Exception:
        return None


def dtlp_bidprice_fulfillment(
    order_info: pd.Series,
    order_items: pd.DataFrame,
    on_hand_inventory_pivot: pd.DataFrame,
    base_costs: pd.Series,
    unit_costs_df: pd.DataFrame,
    compatible_dcs: list,
    dc_to_region: dict,
    order_set: str,
    simulation_date: str,
    order_idx: int,
    verbose: bool = False,
    lp_inputs: Dict[str, Any] | None = None,
    state: DtlpBidPriceState | None = None,
    order_options: OptionMatrix | pd.DataFrame | None = None,
    dynamic_features: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    DTLP bid-price heuristic for per-SKU fulfillment (Acimovic–Farias style) with optional dual caching.

    For each SKU line in the order:
      1) Build supply vector X^tau across DCs from on-hand inventory.
      2) Estimate SKU-specific daily demand d_daily via PTO if available; fallback to lp_inputs['d_daily'].
      3) Solve one transportation LP with demand split by phi_jm and horizon τ to obtain supply duals (mu_i), or reuse cached duals based on cadence policy.
      4) Assign units greedily to argmin_{dc in eligible} (c_ijm - mu_i) while respecting remaining supply and DC compatibility.

    Args:
        order_info: Series with order metadata (must include 'dc_des'; should include 'promise_delivery_days').
        order_items: DataFrame with columns ['sku_ID','quantity'] for this order.
        on_hand_inventory_pivot: DataFrame index=sku, columns=dc, values=on-hand units.
        base_costs: Series of base shipping costs by dc (not directly used here; costs_ijm is used).
        unit_costs_df: DataFrame index=dc, columns=str(dc_des) of deterministic shipping costs.
        compatible_dcs: list of DCs eligible for this order (or empty/None for all).
        dc_to_region: dict mapping dc -> region for fallback when costs columns keyed by region.
        order_set: 'test' or 'proxy_train' (affects PTO 'period').
        simulation_date: str simulation date (passed to PTO helper).
        order_idx: integer order index for logging/trace.
        verbose: verbosity flag for PTO helpers.
        lp_inputs: dict with keys {'d_daily','tau_days','phi_jm','costs_ijm'} prepared upstream.
        state: optional DtlpBidPriceState to control LP refresh cadence and cache duals across orders.

    Returns:
        plan_df: DataFrame with columns ['sku_ID','dc_ori','quantity'] summarizing assignments.
        meta: dict with diagnostics (algo name, runtime_seconds, tau_days, pto_used flag, cadence info).
    """
    if lp_inputs is None:
        raise ValueError("lp_inputs is required for dtlp_bidprice")

    d_daily_global = float(lp_inputs['d_daily'])
    tau_days = float(lp_inputs['tau_days'])
    phi_jm = lp_inputs['phi_jm']
    costs_ijm = lp_inputs['costs_ijm']

    # Stash simulation_date into order_info for PTO helper
    order_info = order_info.copy()
    order_info['simulation_date'] = simulation_date
    options_df_for_pto: pd.DataFrame | None = None
    best_carrier_by_dc: Dict[Any, Any] = {}
    if order_options is not None:
        try:
            om = ensure_option_matrix(order_options)
            options_df_for_pto = om.to_dataframe()
        except Exception:
            options_df_for_pto = None
        if options_df_for_pto is not None and not options_df_for_pto.empty:
            # Pick the cheapest carrier per DC for this order (used to annotate plan_df).
            if {'dc_id', 'carrier_service_id', 'base_cost'} <= set(options_df_for_pto.columns):
                tmp = options_df_for_pto[['dc_id', 'carrier_service_id', 'base_cost']].copy()
                tmp['base_cost'] = tmp['base_cost'].astype(float)
                idx = tmp.groupby('dc_id')['base_cost'].idxmin()
                best = tmp.loc[idx]
                best_carrier_by_dc = dict(zip(best['dc_id'], best['carrier_service_id']))

    t0 = time.perf_counter()

    assigned_qty: Dict[Tuple[Any, Any], int] = defaultdict(int)
    pto_d_daily_cache: Dict[Any, float] = {}
    solves_this_order = 0
    # Time spent on the amortizable opportunity-cost work (PTO demand forecast +
    # LP dual solve). Charged only on orders that trigger a re-solve; the
    # per-order online decision (the bid-price argmin) is the rest of t0..now.
    solve_seconds = 0.0

    def _d_daily_for(sku: Any) -> float:
        # PTO forecast feeds the LP solve only (not the bid-price argmin), so it
        # is computed lazily — only when we actually re-solve — and cached per SKU.
        if sku in pto_d_daily_cache:
            return pto_d_daily_cache[sku]
        d = _forecast_d_daily_from_pto(
            order_info=order_info,
            order_items=order_items[order_items['sku_ID'] == sku],
            options_df=options_df_for_pto,
            order_set=order_set,
            simulation_date=simulation_date,
            verbose=verbose,
            sku=sku,
            dynamic_features=dynamic_features,
        )
        if d is None:
            d = d_daily_global
        pto_d_daily_cache[sku] = d
        return d

    for sku, qty in order_items[['sku_ID', 'quantity']].itertuples(index=False):
        remaining = int(qty)
        if remaining <= 0:
            continue

        # Build per-SKU supply across DCs from on-hand inventory
        if sku in on_hand_inventory_pivot.index:
            X_tau_by_fc = on_hand_inventory_pivot.loc[sku].copy()
        else:
            X_tau_by_fc = pd.Series(0, index=unit_costs_df.index)
        X_tau_by_fc = X_tau_by_fc.reindex(unit_costs_df.index).fillna(0).astype(float)

        # Solve DTLP for this SKU to obtain mu_i, or reuse cached duals. The
        # forecast + solve run on the cadence (not every order) and are timed
        # separately so the per-order solve cost can be amortized in reporting.
        mu_by_dc: pd.Series | None = None
        if state is not None:
            if state._should_solve(sku, X_tau_by_fc, tau_days, phi_jm, costs_ijm):
                _t_solve = time.perf_counter()
                d_daily_eff = _d_daily_for(sku)
                dcs_used, pis = _solve_dtlp_duals(X_tau_by_fc, d_daily_eff, tau_days, phi_jm, costs_ijm)
                solve_seconds += time.perf_counter() - _t_solve
                mu_by_dc = pd.Series(pis, index=dcs_used, dtype=float)
                state.update_entry(sku, mu_by_dc, X_tau_by_fc, tau_days, phi_jm, costs_ijm)
                solves_this_order += 1
            else:
                mu_by_dc = state.get_mu(sku)
                state.tick(sku)
        else:
            _t_solve = time.perf_counter()
            d_daily_eff = _d_daily_for(sku)
            dcs_used, pis = _solve_dtlp_duals(X_tau_by_fc, d_daily_eff, tau_days, phi_jm, costs_ijm)
            solve_seconds += time.perf_counter() - _t_solve
            mu_by_dc = pd.Series(pis, index=dcs_used, dtype=float)
            solves_this_order += 1

        while remaining > 0:
            # Eligible DCs must have positive remaining supply and be compatible (if provided)
            if compatible_dcs:
                eligible = [dc for dc in mu_by_dc.index if float(X_tau_by_fc.loc[dc]) > 0 and dc in compatible_dcs]
            else:
                eligible = [dc for dc in mu_by_dc.index if float(X_tau_by_fc.loc[dc]) > 0]
            if not eligible:
                break

            # Determine (region,speed) demand class; fallback map via dc_to_region if costs_ijm lacks jm
            if 'dc_des' not in order_info:
                raise KeyError("order_info must include 'dc_des'")
            region = order_info['dc_des']
            speed = order_info.get('promise_delivery_days', None)
            jm = (region, speed)
            if jm not in costs_ijm.columns:
                region_alt = dc_to_region.get(region, region)
                jm_alt = (region_alt, speed)
                if jm_alt in costs_ijm.columns:
                    jm = jm_alt
                else:
                    raise KeyError(f"Missing c_ijm for jm={jm} (also tried {jm_alt})")

            # Bid-price rule: choose argmin_dc { c_ijm - mu_i }
            costs = costs_ijm.loc[eligible, jm].astype(float)
            mus = mu_by_dc.reindex(eligible).fillna(0.0)
            scores = costs - mus
            best_dc = scores.idxmin()

            assigned_qty[(sku, best_dc)] += 1
            X_tau_by_fc.loc[best_dc] = float(X_tau_by_fc.loc[best_dc]) - 1.0
            remaining -= 1

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
    plan_df = pd.DataFrame(rows, columns=['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity']) if rows else pd.DataFrame(columns=['sku_ID','dc_ori','carrier_service_id','quantity'])

    total_seconds = time.perf_counter() - t0
    meta = {
        'algo': 'dtlp_bidprice',
        # Full per-order wall-clock (online decision + any solve), consistent with
        # how every other policy self-reports runtime_seconds.
        'runtime_seconds': total_seconds,
        # Split for transparency / amortization: the online bid-price decision vs.
        # the amortizable forecast+LP-solve cost (nonzero only on re-solve orders).
        'decision_seconds': max(0.0, total_seconds - solve_seconds),
        'solve_seconds': solve_seconds,
        'tau_days': tau_days,
        'pto_used': PTO_AVAILABLE,
    }
    if state is not None:
        meta.update({
            'solve_cadence': state.solve_cadence,
            'q': state.q,
            'n_orders': state.n_orders,
            'lp_solves_this_order': solves_this_order,
        })
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
    dynamic_features: pd.DataFrame | None = None,
    lp_inputs: Dict[str, Any] | None = None,
    dtlp_state: DtlpBidPriceState | None = None,
    unit_costs_df: pd.DataFrame | None = None,
    dc_to_region: Dict[Any, Any] | None = None,
):
    """
    Option-level wrapper for DTLP bid-price heuristic (simulator-compatible).

    Returns:
        (OrderDecision, runtime_seconds)
    """
    from src.simulator.entities import OrderDecision, ItemAllocation

    if simulation_date is None:
        raise ValueError("simulation_date is required for dtlp_bidprice")
    if lp_inputs is None:
        raise ValueError("lp_inputs is required for dtlp_bidprice")
    if unit_costs_df is None:
        raise ValueError("unit_costs_df is required for dtlp_bidprice")

    # Reuse the proven adapters from contextual_saa to keep schemas consistent.
    from src.algo.contextual_saa import _prepare_order_items, _build_inventory_pivot, _build_option_matrix

    fallback_order_time = order.order_time or (pd.Timestamp(simulation_date) if simulation_date else None)
    order_items_df, unique_skus, original_qty = _prepare_order_items(order, items, fallback_order_time)
    inventory_pivot = _build_inventory_pivot(unique_skus, option_ids, inventory_snapshot)
    order_options = _build_option_matrix(option_ids, costs_series)

    dest_dc_value = getattr(order, 'customer_dc', None)
    order_info = pd.Series({
        'order_time': order.order_time or pd.Timestamp.now(),
        # Keep dc_des consistent with cost-matrix keys (costs_ijm region columns are strings)
        'dc_des': str(dest_dc_value) if dest_dc_value is not None else order.dest_state,
        'promise_delivery_days': order.promise_delivery_days,
    })

    # Base costs by DC for primal-dual reward mapping / tie breaks (DTLP core doesn't use it)
    options_df = order_options.to_dataframe()
    base_costs_by_dc = (
        options_df.groupby('dc_id')['base_cost'].min()
        if not options_df.empty and 'dc_id' in options_df.columns and 'base_cost' in options_df.columns
        else pd.Series(dtype=float)
    )
    compatible_dcs = sorted({int(opt[0]) for opt in option_ids})

    plan_df, meta = dtlp_bidprice_fulfillment(
        order_info=order_info,
        order_items=order_items_df,
        on_hand_inventory_pivot=inventory_pivot,
        base_costs=base_costs_by_dc,
        unit_costs_df=unit_costs_df,
        compatible_dcs=compatible_dcs,
        dc_to_region=dc_to_region or {},
        order_set=order_set,
        simulation_date=simulation_date,
        order_idx=order_idx,
        verbose=verbose,
        lp_inputs=lp_inputs,
        state=dtlp_state,
        order_options=order_options,
        dynamic_features=dynamic_features,
    )

    runtime = float(meta.get('runtime_seconds', 0.0) or 0.0)
    # Surface the decision/solve split to the engine, which merges policy stats
    # into the per-order runtimes table (see SimulationEngine policy handling).
    stats = {
        'decision_seconds': float(meta.get('decision_seconds', runtime) or 0.0),
        'solve_seconds': float(meta.get('solve_seconds', 0.0) or 0.0),
        'lp_solves_this_order': int(meta.get('lp_solves_this_order', 0) or 0),
    }

    allocations: list[ItemAllocation] = []
    allocated_qty: dict[str, int] = defaultdict(int)
    unfilled: dict[str, int] = {}

    if not plan_df.empty:
        plan_values = plan_df[['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity']].to_numpy()
    else:
        plan_values = np.empty((0, 4))

    for sku_id, dc_id, carrier_id, qty in plan_values:
        sku_id = str(sku_id)
        qty_int = int(float(qty))
        if qty_int <= 0 or pd.isna(dc_id):
            continue
        dc_id_int = int(dc_id)
        if pd.isna(carrier_id):
            # Fallback: pick cheapest carrier for that DC among eligible options
            if not options_df.empty:
                sub = options_df[options_df['dc_id'] == dc_id_int]
                if not sub.empty:
                    carrier_id = int(sub.loc[sub['base_cost'].astype(float).idxmin(), 'carrier_service_id'])
        if pd.isna(carrier_id):
            continue
        allocations.append(ItemAllocation(
            sku_id=sku_id,
            option_id=(dc_id_int, int(carrier_id)),
            quantity=qty_int,
        ))
        allocated_qty[sku_id] += qty_int

    for sku_id, orig in original_qty.items():
        rem = int(orig) - int(allocated_qty.get(sku_id, 0))
        if rem > 0:
            unfilled[sku_id] = rem

    decision = OrderDecision(allocations=allocations, unfilled=unfilled if unfilled else None)
    return decision, runtime, stats


def create_policy_for_simulation(
    catalog,
    precompute,
    state,
    order_set: str,
    simulation_date: str,
    *,
    dtlp_cadence: str = 'per_order',
    dtlp_q: int = 100,
    dtlp_every_n_orders: int = 1,
    tau_days: float = 1.0,
    **kwargs,
):
    """
    Create DTLP bid-price policy closure for the unified simulator.

    Notes:
      - Uses a date-level demand mix φ_{j,m} from preprocessed_data_cs.csv.
      - Builds the cost matrix c_{i,j,m} from the live cost model (same basis as
        build_costs), via features.build_dc_region_cost_matrix.
      - Chooses the cheapest carrier per chosen DC for each order (carrier affects outcomes, not DTLP LP).
    """
    import src.config as cfg
    from src.simulator.features import build_costs, build_dc_region_cost_matrix
    from src.simulator.entities import OrderDecision

    # Region fallback mapping (optional)
    dc_to_region: dict[Any, Any] = {}
    try:
        network_df = pd.read_csv(cfg.NETWORK_PATH)
        if {'dc_ID', 'region_ID'} <= set(network_df.columns):
            dc_to_region = network_df.set_index('dc_ID')['region_ID'].to_dict()
    except Exception:
        dc_to_region = {}

    # Build date-level φ_{j,m} and a region→ZIP map for the live cost matrix.
    region_zip_weights: dict[str, dict[str, int]] = {}
    zip_coords: dict[str, tuple] = {}
    try:
        orders_path = cfg.PROCESSED_DATA_DIR / 'preprocessed_data_cs.csv'
        cols = ['order_time', 'dc_des', 'promise_delivery_days',
                'customer_zip5', 'customer_lat', 'customer_lon']
        all_orders = pd.read_csv(orders_path, usecols=cols, parse_dates=['order_time'])
        # Normalize destination/zip key types to string (matches cost-model lookups).
        all_orders['dc_des'] = all_orders['dc_des'].astype(str)
        all_orders['customer_zip5'] = all_orders['customer_zip5'].astype(str)
        # Region→ZIP weights over ALL orders (geography is date-independent), so
        # every region present on any date has representative destinations.
        for region, grp in all_orders.groupby('dc_des'):
            region_zip_weights[region] = grp['customer_zip5'].value_counts().to_dict()
        # ZIP→(lat, lon) for the cost matrix's haversine fallback.
        zip_coords = (
            all_orders.dropna(subset=['customer_lat', 'customer_lon'])
            .drop_duplicates('customer_zip5')
            .set_index('customer_zip5')[['customer_lat', 'customer_lon']]
            .apply(lambda r: (float(r['customer_lat']), float(r['customer_lon'])), axis=1)
            .to_dict()
        )
        date_orders = all_orders[all_orders['order_time'].dt.date == pd.to_datetime(simulation_date).date()]
        jm_counts = date_orders.groupby(['dc_des', 'promise_delivery_days']).size()
        if jm_counts.empty:
            # Fallback single cell; will likely be overridden by dc_to_region fallback at runtime
            jm_index = pd.MultiIndex.from_tuples([('0', 0)], names=['dc_des', 'speed'])
            phi_jm = pd.Series([1.0], index=jm_index, dtype=float)
            regions = ['0']
            speeds = [0]
        else:
            phi_jm = (jm_counts / float(jm_counts.sum())).astype(float)
            phi_jm.index = phi_jm.index.set_names(['dc_des', 'speed'])
            regions = sorted(phi_jm.index.get_level_values('dc_des').unique().tolist())
            speeds = sorted(phi_jm.index.get_level_values('speed').unique().tolist())
    except Exception:
        jm_index = pd.MultiIndex.from_tuples([('0', 0)], names=['dc_des', 'speed'])
        phi_jm = pd.Series([1.0], index=jm_index, dtype=float)
        regions = ['0']
        speeds = [0]

    # Cost matrix c_{i,j} from the live cost model (same basis as build_costs).
    # Columns are destination regions (dc_des); index is the full DC universe.
    unit_costs_df = build_dc_region_cost_matrix(
        catalog, precompute, region_zip_weights, zip_coords=zip_coords)

    jm_cols = pd.MultiIndex.from_product([regions, speeds], names=['dc_des', 'speed'])
    costs_ijm = pd.DataFrame(index=unit_costs_df.index, columns=jm_cols, dtype=float)
    for j in regions:
        col = str(j)
        for m_key in speeds:
            if col in unit_costs_df.columns:
                costs_ijm[(j, m_key)] = unit_costs_df[col].astype(float)
            else:
                costs_ijm[(j, m_key)] = np.inf

    lp_inputs = {
        'd_daily': 1.0,  # fallback only; PTO per-SKU demand is used when order_options are provided
        'tau_days': float(tau_days),
        'phi_jm': phi_jm,
        'costs_ijm': costs_ijm,
    }

    dtlp_state = DtlpBidPriceState(solve_cadence=dtlp_cadence, q=int(dtlp_q), n_orders=int(dtlp_every_n_orders))

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
            dynamic_features=dynamic_snapshot,
            lp_inputs=lp_inputs,
            dtlp_state=dtlp_state,
            unit_costs_df=unit_costs_df,
            dc_to_region=dc_to_region,
        )

    return policy