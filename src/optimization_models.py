import time
from pathlib import Path
from typing import Optional, MutableMapping, Dict
import pandas as pd
import src.config as cfg
import numpy as np
from src.data_structures.options import ensure_option_matrix

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

def solve_stochastic_lookahead_model(order_items, options_df, on_hand_inventory,
                                     demand_scenarios, shipping_costs, stockout_penalty, base_costs_by_option,
                                     future_penalties=None,
                                     write_model_path: Optional[str] = None, model_format: str = "lp",
                                     skip_optimize: bool = False,
                                     model_stats: Optional[MutableMapping[str, int]] = None,
                                     return_objective_components: bool = False):
    """Construct and solve the stochastic lookahead model using option indices.
    
    Args:
        order_items: DataFrame with columns ['sku_ID', 'quantity']
        options_df: DataFrame with columns ['option_id', 'dc_id', 'carrier_service_id', 'base_cost']
        on_hand_inventory: DataFrame indexed by SKU, columns are DC IDs
        demand_scenarios: Dict mapping SKU (str) to DataFrame of demand scenarios
        shipping_costs: Dict mapping SKU (str) to DataFrame of shipping costs per scenario
        stockout_penalty: Penalty per unit for stockouts
        base_costs_by_option: Base costs (legacy parameter, may be unused)
        write_model_path: Optional path to write model file
        model_format: Format for model file ('lp' or 'mps')
        skip_optimize: If True, build model but don't solve
        model_stats: Optional dict to populate with model statistics
    """
    start_time = time.time()
    if not GUROBI_AVAILABLE:
        print(f"    - Warning: Gurobipy not found. Skipping solution for stochastic model.")
        if return_objective_components:
            return None, 0.0, 0.0, {}
        return None, 0.0, 0.0

    # Extract order information from order_items DataFrame
    # Normalize SKU identifiers to strings (scenario dictionaries are keyed by string)
    raw_skus = order_items['sku_ID'].unique()
    order_skus = [str(sku) for sku in raw_skus]
    sku_id_map = {str(sku): sku for sku in raw_skus}
    qty_map = {str(k): float(v) for k, v in order_items.set_index('sku_ID')['quantity'].to_dict().items()}
    
    options_matrix = ensure_option_matrix(options_df)
    option_ids_np = options_matrix.option_ids.astype(int)
    option_ids = option_ids_np.tolist()
    option_to_dc = pd.Series(options_matrix.dc_ids, index=option_ids_np)
    option_to_carrier = pd.Series(options_matrix.carrier_service_ids, index=option_ids_np)
    base_cost_series = pd.Series(options_matrix.base_costs, index=option_ids_np, dtype=float)
    dc_to_option_ids = options_matrix.dc_groups()
    
    # Align inventory matrix on DC columns and string SKU index
    inventory_matrix = on_hand_inventory.reindex(
        columns=sorted(option_to_dc.unique()), fill_value=0
    ).copy()
    inventory_matrix.index = inventory_matrix.index.astype(str)

    num_scenarios = len(list(demand_scenarios.values())[0])
    prob = 1.0 / num_scenarios
    model_time_limit = cfg.GUROBI_TIME_LIMIT

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            model.setParam('MIPFocus', cfg.GUROBI_MIP_FOCUS)
            model.setParam('Threads', cfg.GUROBI_THREADS)
            model.setParam('TimeLimit', model_time_limit)

            # ===================================================================
            # DECISION VARIABLES
            # ===================================================================
            
            # z[i, j]: Quantity of SKU i shipped via option j in the current period
            #   - i: SKU ID (string)
            #   - j: Option ID (integer, maps to (dc_id, carrier_service_id))
            #   This is the primary fulfillment decision variable
            z = model.addVars(order_skus, option_ids, lb=0, vtype=GRB.INTEGER, name="z")
            
            # s[i]: Shortage (unfulfilled quantity) for SKU i in current period
            s = model.addVars(order_skus, lb=0, vtype=GRB.CONTINUOUS, name="s")
            
            # S[i, ω]: Shortage for SKU i in future scenario ω
            S = model.addVars(order_skus, range(num_scenarios), lb=0, vtype=GRB.CONTINUOUS, name="S")
            
            # v[i, j, ω]: Quantity of SKU i reserved for future demand via option j in scenario ω
            v = model.addVars(order_skus, option_ids, range(num_scenarios), lb=0, vtype=GRB.INTEGER, name="v")
            
            # y[j]: Binary indicator if option j is used (for discount logic)
            y = model.addVars(option_ids, vtype=GRB.BINARY, name="y")
            
            # w[j]: Discounted cost for option j (auxiliary variable for discount calculation)
            w = model.addVars(option_ids, lb=0, vtype=GRB.CONTINUOUS, name="w")

            # ===================================================================
            # OBJECTIVE FUNCTION
            # ===================================================================
            # Minimize: Expected shipping costs + stockout penalties - discounts
            
            rho = stockout_penalty
            
            # Expected shipping cost for current period (averaged across scenarios)
            exp_ship_now = gp.quicksum(
                prob * shipping_costs[str_i].loc[f'scenario_{s}', opt] * z[str_i, opt]
                for str_i in order_skus
                for opt in option_ids
                for s in range(num_scenarios)
            ) if shipping_costs else 0.0
            
            # Stockout penalty for current period
            curr_short_cost = rho * gp.quicksum(s[str_i] for str_i in order_skus)
            
            # Discount for using options (volume-based discount)
            discount = cfg.BETA_DISCOUNT * gp.quicksum(w[opt] for opt in option_ids)
            
            # Expected stockout penalty for future periods
            future_short_cost = rho * gp.quicksum(
                prob * S[str_i, ω] for str_i in order_skus for ω in range(num_scenarios)
            )
            
            # Expected shipping cost for future periods
            base_cost_vector = base_cost_series.reindex(option_ids).to_numpy(dtype=float)

            # By default, future shipments use deterministic base costs. When `future_penalties`
            # is provided, add a scenario-dependent delivery penalty per option to better align
            # with evaluation-stage SAA and simulator cost accounting.
            if future_penalties is not None:
                pen = future_penalties
                if isinstance(pen, pd.DataFrame):
                    pen = pen.to_numpy(dtype=float)
                else:
                    pen = np.asarray(pen, dtype=float)
                if pen.shape != (num_scenarios, len(option_ids)):
                    raise ValueError(
                        f"future_penalties must have shape (num_scenarios, num_options)=({num_scenarios},{len(option_ids)}); "
                        f"got {getattr(pen, 'shape', None)}"
                    )
                exp_ship_future = gp.quicksum(
                    prob * (base_cost_vector[opt_idx] + float(pen[ω, opt_idx])) * v[str_i, opt, ω]
                    for str_i in order_skus
                    for opt_idx, opt in enumerate(option_ids)
                    for ω in range(num_scenarios)
                )
            else:
                exp_ship_future = gp.quicksum(
                    prob * base_cost_vector[opt_idx] * v[str_i, opt, ω]
                    for str_i in order_skus
                    for opt_idx, opt in enumerate(option_ids)
                    for ω in range(num_scenarios)
                )

            model.setObjective(
                exp_ship_now + curr_short_cost - discount + future_short_cost + exp_ship_future,
                GRB.MINIMIZE,
            )

            # ===================================================================
            # CONSTRAINTS
            # ===================================================================
            
            # Current period fulfillment: For each SKU i, sum of z[i, j] + shortage = order quantity
            # This ensures all demand is either fulfilled or recorded as shortage
            for str_i in order_skus:
                qty_i = qty_map.get(str_i, 0.0)
                model.addConstr(
                    gp.quicksum(z[str_i, opt] for opt in option_ids) + s[str_i] == qty_i,
                    name=f"fulfill_q_{str_i}",
                )

            # Future period constraints: For each scenario ω and SKU i
            for ω in range(num_scenarios):
                for str_i in order_skus:
                    # Future demand fulfillment: sum of v[i, j, ω] + shortage = future demand
                    demand_series = demand_scenarios.get(str_i)
                    future_demand = float(demand_series.loc[f'scenario_{ω}']) if demand_series is not None else 0.0
                    model.addConstr(
                        gp.quicksum(v[str_i, opt, ω] for opt in option_ids) + S[str_i, ω] == future_demand,
                        name=f"future_demand_{str_i}_{ω}",
                    )
                    
                    # Inventory constraints: For each DC, current shipments + future reservations <= inventory
                    # Note: We iterate over DCs and sum over all options for that DC
                    for dc, opt_list in dc_to_option_ids.items():
                        if str_i in inventory_matrix.index and dc in inventory_matrix.columns:
                            inv_val = float(inventory_matrix.loc[str_i, dc])
                        else:
                            inv_val = 0.0
                        if inv_val >= 0:
                            # Sum of z[i, j] for all options j at this DC
                            shipped_now = gp.quicksum(z[str_i, opt] for opt in opt_list)
                            # Sum of v[i, j, ω] for all options j at this DC in scenario ω
                            reserved_future = gp.quicksum(v[str_i, opt, ω] for opt in opt_list)
                            model.addConstr(
                                shipped_now + reserved_future <= inv_val,
                                name=f"inv_{str_i}_{dc}_{ω}",
                            )

            # Discount logic: Big-M constraints to model volume-based discounts
            # y[j] = 1 if option j is used (sum of z[i, j] >= 2), 0 otherwise
            # w[j] = base_cost * sum(z[i, j]) if y[j] = 1, else 0
            q_total = float(order_items['quantity'].sum())
            N = q_total + 2.0  # Big-M constant
            for opt in option_ids:
                base_cost = float(base_cost_series.loc[opt]) if opt in base_cost_series.index else 0.0
                M_opt = base_cost * q_total if q_total > 0 else 0.0  # Option-specific Big-M
                sum_z_opt = gp.quicksum(z[str_i, opt] for str_i in order_skus)
                
                # y[j] = 1 if sum(z[i, j]) >= 2, else 0
                model.addConstr(sum_z_opt >= 2 - N * (1 - y[opt]), name=f"y_logic_lower_{opt}")
                model.addConstr(sum_z_opt <= 1 + N * y[opt], name=f"y_logic_upper_{opt}")
                
                # w[j] = base_cost * sum(z[i, j]) if y[j] = 1, else 0
                base_cost_z_opt = base_cost * sum_z_opt
                model.addConstr(w[opt] <= M_opt * y[opt], name=f"w_logic_upper1_{opt}")
                model.addConstr(w[opt] <= base_cost_z_opt + (1 - y[opt]) * M_opt, name=f"w_logic_upper2_{opt}")
                model.addConstr(w[opt] >= base_cost_z_opt - (1 - y[opt]) * M_opt, name=f"w_logic_lower_{opt}")

            model.update()
            if model_stats is not None:
                total_vars = int(model.NumVars)
                bin_vars = int(model.NumBinVars)
                gen_int_vars = max(int(model.NumIntVars) - bin_vars, 0)
                cont_vars = max(total_vars - bin_vars - gen_int_vars, 0)
                model_stats.clear()
                model_stats.update({
                    'total_vars': total_vars,
                    'binary_vars': bin_vars,
                    'integer_vars': gen_int_vars,
                    'continuous_vars': cont_vars,
                })

            if write_model_path is not None:
                model_path = Path(write_model_path)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = model_format.lower()
                if fmt not in {"lp", "mps"}:
                    raise ValueError(f"Unsupported model format '{model_format}'. Use 'lp' or 'mps'.")
                if fmt == "lp" and model_path.suffix.lower() != ".lp":
                    model_path = model_path.with_suffix(".lp")
                if fmt == "mps" and model_path.suffix.lower() not in {".mps", ".gz"}:
                    model_path = model_path.with_suffix(".mps")
                model.write(str(model_path))
                if skip_optimize:
                    runtime = time.time() - start_time
                    if return_objective_components:
                        return None, 0.0, runtime, {}
                    return None, 0.0, runtime

            model.optimize()

            runtime = time.time() - start_time
            if model.status != GRB.OPTIMAL:
                if return_objective_components:
                    return None, 0.0, runtime, {}
                return None, 0.0, runtime

            solution = model.getAttr('X', z)
            if return_objective_components:
                current_stage_val = float(exp_ship_now.getValue() + curr_short_cost.getValue() - discount.getValue())
                future_recourse_val = float(future_short_cost.getValue() + exp_ship_future.getValue())
                objective_parts = {
                    "current_stage": current_stage_val,
                    "future_recourse": future_recourse_val,
                    "total": float(model.ObjVal),
                }
                return solution, model.ObjVal, runtime, objective_parts
            return solution, model.ObjVal, runtime

def solve_deterministic_model(order_items, options_df, on_hand_inventory, demand_scenario,
                              shipping_costs_scenario, stockout_penalty, base_costs_by_option):
    """Solve the deterministic single-scenario model using option indices.

    When ``options_df`` is ``None`` this function falls back to DC-only inputs
    and synthesises a single carrier per DC (carrier_service_id = 0).
    
    Args:
        order_items: DataFrame with columns ['sku_ID', 'quantity']
        options_df: DataFrame with columns ['option_id', 'dc_id', 'carrier_service_id', 'base_cost']
                   If None, creates options from base_costs_by_option
        on_hand_inventory: DataFrame indexed by SKU, columns are DC IDs
        demand_scenario: Dict mapping SKU to future demand quantity
        shipping_costs_scenario: Dict mapping SKU to shipping costs (various formats supported)
        stockout_penalty: Penalty per unit for stockouts
        base_costs_by_option: Base costs Series (required if options_df is None)
    """
    start_time = time.time()
    if not GUROBI_AVAILABLE:
        print(f"    - Warning: Gurobipy not found. Skipping solution for deterministic model.")
        return None, 0.0

    # Handle legacy mode: if options_df is None, create it from base_costs_by_option
    legacy_mode = options_df is None or (isinstance(options_df, pd.DataFrame) and options_df.empty)
    if legacy_mode:
        if base_costs_by_option is None:
            raise ValueError("base_costs_by_option is required when options_df is None.")
        legacy_dcs = list(base_costs_by_option.index)
        options_df = pd.DataFrame({
            'option_id': np.arange(len(legacy_dcs), dtype=int),
            'dc_id': legacy_dcs,
            'carrier_service_id': [0] * len(legacy_dcs),
            'base_cost': [float(base_costs_by_option.loc[dc]) for dc in legacy_dcs],
        })

    options_matrix = ensure_option_matrix(options_df)

    # Extract order information from order_items DataFrame
    raw_skus = order_items['sku_ID'].unique()
    order_skus = [str(sku) for sku in raw_skus]
    sku_id_map = {str(sku): sku for sku in raw_skus}
    qty_map = {str(k): float(v) for k, v in order_items.set_index('sku_ID')['quantity'].to_dict().items()}
    
    option_ids_np = options_matrix.option_ids.astype(int)
    option_ids = option_ids_np.tolist()
    option_to_dc = pd.Series(options_matrix.dc_ids, index=option_ids_np)
    option_to_carrier = pd.Series(options_matrix.carrier_service_ids, index=option_ids_np)
    base_cost_series = pd.Series(options_matrix.base_costs, index=option_ids_np, dtype=float)
    dc_to_option_ids = options_matrix.dc_groups()
    
    # Align inventory matrix on DC columns and string SKU index
    inventory_matrix = on_hand_inventory.reindex(
        columns=sorted(option_to_dc.unique()), fill_value=0
    ).copy()
    inventory_matrix.index = inventory_matrix.index.astype(str)

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            model.setParam('MIPFocus', cfg.GUROBI_MIP_FOCUS)
            model.setParam('Threads', cfg.GUROBI_THREADS)
            model.setParam('TimeLimit', cfg.GUROBI_TIME_LIMIT)

            # ===================================================================
            # DECISION VARIABLES
            # ===================================================================
            
            # z[i, j]: Quantity of SKU i shipped via option j in the current period
            #   - i: SKU ID (string)
            #   - j: Option ID (integer, maps to (dc_id, carrier_service_id))
            #   This is the primary fulfillment decision variable
            z = model.addVars(order_skus, option_ids, lb=0, vtype=GRB.INTEGER, name="z")
            
            # s[i]: Shortage (unfulfilled quantity) for SKU i in current period
            s = model.addVars(order_skus, lb=0, vtype=GRB.CONTINUOUS, name="s")
            
            # S[i]: Shortage for SKU i in future period
            S = model.addVars(order_skus, lb=0, vtype=GRB.CONTINUOUS, name="S")
            
            # v[i, j]: Quantity of SKU i reserved for future demand via option j
            v = model.addVars(order_skus, option_ids, lb=0, vtype=GRB.INTEGER, name="v")
            
            # y[j]: Binary indicator if option j is used (for discount logic)
            y = model.addVars(option_ids, vtype=GRB.BINARY, name="y")
            
            # w[j]: Discounted cost for option j (auxiliary variable for discount calculation)
            w = model.addVars(option_ids, lb=0, vtype=GRB.CONTINUOUS, name="w")

            def _normalize_shipping(entry):
                if isinstance(entry, pd.DataFrame):
                    df = entry.copy()
                    df.columns = [int(c) for c in df.columns]
                    return df.reindex(columns=option_ids).fillna(base_cost_series.loc[option_ids])
                if isinstance(entry, dict):
                    row = [float(entry.get(option_to_dc.loc[opt], base_cost_series.loc[opt])) for opt in option_ids]
                    return pd.DataFrame([row], index=['scenario_0'], columns=option_ids)
                if isinstance(entry, (pd.Series, list, tuple)):
                    row_dict = dict(entry)
                    row = [float(row_dict.get(option_to_dc.loc[opt], base_cost_series.loc[opt])) for opt in option_ids]
                    return pd.DataFrame([row], index=['scenario_0'], columns=option_ids)
                raise ValueError("Unrecognised legacy shipping cost format.")

            # Normalize shipping costs to consistent format (DataFrame with option_id columns)
            option_id_set = set(option_ids)
            shipping_costs_norm: Dict[str, pd.DataFrame] = {}
            for sku_key, entry in shipping_costs_scenario.items():
                str_key = str(sku_key)
                if isinstance(entry, pd.DataFrame) and set(entry.columns) == option_id_set:
                    df = entry
                    if list(entry.columns) != option_ids:
                        df = entry.reindex(columns=option_ids).fillna(base_cost_series.loc[option_ids])
                    shipping_costs_norm[str_key] = df
                else:
                    shipping_costs_norm[str_key] = _normalize_shipping(entry)

            # ===================================================================
            # OBJECTIVE FUNCTION
            # ===================================================================
            # Minimize: Shipping costs + stockout penalties - discounts
            
            # Current period shipping costs
            #
            # NOTE: `shipping_costs_norm[str_i]` is normalized to a DataFrame with a
            # single row (scenario_0) and option_id columns. Using `.loc[:, opt]`
            # returns a Series, which cannot be multiplied into a Gurobi expression.
            current_shipping_cost = gp.quicksum(
                float(shipping_costs_norm[str_i].loc[:, opt].iloc[0]) * z[str_i, opt]
                for str_i in order_skus for opt in option_ids
            )
            
            # Discount for using options (volume-based discount)
            discount = gp.quicksum(cfg.BETA_DISCOUNT * w[opt] for opt in option_ids)
            
            # Stockout penalties
            current_stockout_penalty = gp.quicksum(stockout_penalty * s[str_i] for str_i in order_skus)
            future_stockout_penalty = gp.quicksum(stockout_penalty * S[str_i] for str_i in order_skus)
            
            # Future period shipping costs
            future_shipping_cost = gp.quicksum(
                float(base_cost_series.loc[opt]) * v[str_i, opt]
                for str_i in order_skus for opt in option_ids
            )
            
            model.setObjective(
                current_shipping_cost - discount + current_stockout_penalty + 
                future_stockout_penalty + future_shipping_cost,
                GRB.MINIMIZE
            )

            # ===================================================================
            # CONSTRAINTS
            # ===================================================================
            
            # Current period fulfillment: For each SKU i, sum of z[i, j] + shortage = order quantity
            for str_i in order_skus:
                qty_i = qty_map.get(str_i, 0.0)
                model.addConstr(
                    gp.quicksum(z[str_i, opt] for opt in option_ids) + s[str_i] == qty_i,
                    name=f"fulfill_q_{str_i}",
                )

            # Inventory constraints: For each DC and SKU, current shipments + future reservations <= inventory
            for str_i in order_skus:
                for dc, opt_list in dc_to_option_ids.items():
                    if str_i in inventory_matrix.index and dc in inventory_matrix.columns:
                        inv_val = float(inventory_matrix.loc[str_i, dc])
                    else:
                        inv_val = 0.0
                    if inv_val >= 0:
                        # Sum of z[i, j] for all options j at this DC
                        shipped_now = gp.quicksum(z[str_i, opt] for opt in opt_list)
                        # Sum of v[i, j] for all options j at this DC
                        reserved_future = gp.quicksum(v[str_i, opt] for opt in opt_list)
                        model.addConstr(shipped_now + reserved_future <= inv_val,
                                        name=f"inv_{str_i}_{dc}")

            # Future period fulfillment: For each SKU i, sum of v[i, j] + shortage = future demand
            for str_i in order_skus:
                future_demand = float(demand_scenario.get(sku_id_map[str_i], demand_scenario.get(str_i, 0)))
                model.addConstr(
                    gp.quicksum(v[str_i, opt] for opt in option_ids) + S[str_i] == future_demand,
                    name=f"future_demand_{str_i}",
                )

            # Discount logic: Big-M constraints to model volume-based discounts
            # y[j] = 1 if option j is used (sum of z[i, j] >= 2), 0 otherwise
            # w[j] = base_cost * sum(z[i, j]) if y[j] = 1, else 0
            q_total = float(order_items['quantity'].sum())
            N = q_total + 2.0  # Big-M constant
            for opt in option_ids:
                base_cost = float(base_cost_series.loc[opt]) if opt in base_cost_series.index else 0.0
                M_opt = base_cost * q_total if q_total > 0 else 0.0  # Option-specific Big-M
                sum_z_opt = gp.quicksum(z[str_i, opt] for str_i in order_skus)
                
                # y[j] = 1 if sum(z[i, j]) >= 2, else 0
                model.addConstr(sum_z_opt >= 2 - N * (1 - y[opt]), name=f"y_logic_lower_{opt}")
                model.addConstr(sum_z_opt <= 1 + N * y[opt], name=f"y_logic_upper_{opt}")
                
                # w[j] = base_cost * sum(z[i, j]) if y[j] = 1, else 0
                base_cost_z_opt = base_cost * sum_z_opt
                model.addConstr(w[opt] <= M_opt * y[opt], name=f"w_logic_upper1_{opt}")
                model.addConstr(w[opt] <= base_cost_z_opt + (1 - y[opt]) * M_opt, name=f"w_logic_upper2_{opt}")
                model.addConstr(w[opt] >= base_cost_z_opt - (1 - y[opt]) * M_opt, name=f"w_logic_lower_{opt}")

            model.optimize()
            runtime = time.time() - start_time
            if model.status != GRB.OPTIMAL:
                return None, runtime

            plan_rows = []
            for str_i in order_skus:
                sku_original = sku_id_map[str_i]
                for opt in option_ids:
                    qty_val = z[str_i, opt].X
                    if qty_val > 1e-6:
                        plan_rows.append({
                            'sku_ID': sku_original,
                            'dc_ori': int(option_to_dc.loc[opt]),
                            'carrier_service_id': int(option_to_carrier.loc[opt]),
                            'quantity': int(round(qty_val)),
                        })

            fulfillment_plan = pd.DataFrame(plan_rows, columns=['sku_ID', 'dc_ori', 'carrier_service_id', 'quantity'])
            return fulfillment_plan, runtime 
