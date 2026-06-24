"""Discrete-event simulation engine."""

import logging
import time
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

import src.config as cfg
from src.simulator.entities import Order, OrderDecision, ItemAllocation, OrderItem
from src.simulator.catalog import OptionsCatalog
from src.simulator.precompute import PrecomputeStore
from src.simulator.state import SimulationState
from src.simulator.features import build_features, build_costs
from src.simulator.delivery_sampler import OutcomeSampler
from src.simulator.processing_time import ProcessingTimeModel
from src.data_utils import compute_haversine_km, get_per_unit_base_cost

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Discrete-event simulation engine for order fulfillment."""
    
    def __init__(
        self,
        catalog: OptionsCatalog,
        precompute: PrecomputeStore,
        state: SimulationState,
        sampler: OutcomeSampler,
        policy: Callable,
        processing_model: Optional[ProcessingTimeModel] = None,
    ):
        """
        Initialize simulation engine.
        
        Args:
            catalog: Options catalog
            precompute: Precompute store
            state: Simulation state
            sampler: Outcome sampler
            policy: Policy function that takes only (order) -> OrderDecision
        """
        self.catalog = catalog
        self.precompute = precompute
        self.state = state
        self.sampler = sampler
        self.policy = policy
        rates_df = None
        try:
            rates_df = self.precompute.get_processing_rates()
        except Exception:
            rates_df = None
        self.processing_model = processing_model or ProcessingTimeModel(rates_df)
        
        # Results storage
        self.results: List[Dict] = []
        self.order_runtimes: List[Dict] = []
        self.rng: Optional[np.random.Generator] = None
        self._timing_stats: Optional[DefaultDict[str, Dict[str, Optional[float]]]] = None
        self._slow_order_seconds: float = getattr(cfg, "SIM_LOG_SLOW_ORDER_SECONDS", 1.0)
        self._carrier_cost_coef = self._load_cost_model_coeffs()
        # Lightweight run counters for logging/auditing (especially resume + collect-only runs).
        self.orders_total_in_run: int = 0
        self.orders_start_index: int = 0
        self.orders_processed_this_run: int = 0
        self.orders_no_eligible_this_run: int = 0

    @staticmethod
    def _compute_implied_unfilled(order: Order, decision: OrderDecision) -> Dict[str, int]:
        """
        Compute lost-sales quantities as (order demand - allocated) per SKU.

        Policies may supply `decision.unfilled`, but the simulator treats this implied residual
        as the source of truth for metrics/cost to avoid policy-specific accounting bugs.
        """
        demand: Dict[str, int] = defaultdict(int)
        for item in order.items:
            demand[str(item.sku_id)] += int(item.quantity)

        allocated: Dict[str, int] = defaultdict(int)
        for alloc in decision.allocations:
            allocated[str(alloc.sku_id)] += int(alloc.quantity)

        unfilled: Dict[str, int] = {}
        for sku_id, qty in demand.items():
            rem = int(qty) - int(allocated.get(sku_id, 0))
            if rem > 0:
                unfilled[sku_id] = rem
        return unfilled
    
    def run(
        self,
        orders: Iterable[Order],
        num_replications: int = 1,
        rng_seed: Optional[int] = None,
        show_progress: bool = False,
        start_index: int = 0,
        resume_results: Optional[List[Dict]] = None,
        resume_order_runtimes: Optional[List[Dict]] = None,
        rng_state: Optional[Dict[str, object]] = None,
        checkpoint_callback: Optional[Callable[[int, 'SimulationEngine'], None]] = None,
        collect_only: bool = False,
    ) -> List[Dict]:
        """
        Run simulation over orders.
        
        Args:
            orders: Iterable of Order objects
            num_replications: Number of replications for outcome sampling
            rng_seed: Random seed for reproducibility
            show_progress: Whether to print progress updates while processing orders
            collect_only: If True, skip outcome simulation (for data collection)
            
        Returns:
            List of result dictionaries
        """
        if rng_state is not None:
            self.rng = np.random.default_rng()
            try:
                self.rng.bit_generator.state = rng_state  # type: ignore[assignment]
            except Exception:
                logger.warning("Failed to restore RNG state; reseeding with provided seed.")
                self.rng = np.random.default_rng(rng_seed or cfg.RANDOM_SEED)
        else:
            self.rng = np.random.default_rng(rng_seed or cfg.RANDOM_SEED)

        self.results = list(resume_results or [])
        self.order_runtimes = list(resume_order_runtimes or [])
        self._init_timing_stats()
        self.orders_processed_this_run = 0
        self.orders_no_eligible_this_run = 0
        
        orders_list = list(orders)
        total_orders = len(orders_list)
        start_index = max(0, int(start_index))
        self.orders_total_in_run = total_orders
        self.orders_start_index = start_index
        if start_index >= total_orders:
            if logger.isEnabledFor(logging.INFO):
                logger.info("No orders left to process (start_index=%d, total=%d).", start_index, total_orders)
            return self.results
        progress_step = max(1, total_orders // 20) if total_orders else 1
        run_start = time.perf_counter()
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Simulation run starting for %d orders (replications=%d)",
                total_orders,
                num_replications,
            )
        
        # Process orders chronologically
        for idx, order in enumerate(orders_list[start_index:], start=start_index + 1):
            self._process_order(order, num_replications, collect_only=collect_only)
            self.orders_processed_this_run += 1
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(idx, self)
                except Exception:
                    logger.exception("Checkpoint callback failed at order index %d", idx)
            if show_progress and total_orders:
                if idx == 1 or idx == total_orders or idx % progress_step == 0:
                    pct = idx / total_orders * 100
                    print(f"  Processed {idx}/{total_orders} orders ({pct:5.1f}%)", flush=True)
        
        total_run_time = time.perf_counter() - run_start
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Simulation run finished in %.2fs for %d orders",
                total_run_time,
                total_orders,
            )
        self._log_timing_summary()
        
        return self.results
    
    def _process_order(self, order: Order, num_replications: int, collect_only: bool = False):
        """Process a single order.
        
        Args:
            collect_only: If True, skip outcome simulation (only apply state changes)
        """
        order_start = time.perf_counter()
        stage_durations: Dict[str, float] = {}
        item_count = len(order.items)
        total_qty = sum(item.quantity for item in order.items)
        if logger.isEnabledFor(logging.DEBUG):
            customer_dc_val = getattr(order, "customer_dc", None)
            sku_summary = self._summarize_items(order.items)
            logger.debug(
                "Processing order %s (order_time=%s, items=%d, total_qty=%d, dest=%s, dc_des=%s, promise_days=%s, skus=%s)",
                order.order_id,
                order.order_time,
                item_count,
                total_qty,
                getattr(order, "dest_zip5", None),
                customer_dc_val,
                getattr(order, "promise_delivery_days", None),
                sku_summary,
            )
        # Set current time
        if order.order_time:
            if isinstance(order.order_time, datetime):
                self.state.set_time(order.order_time)
            else:
                self.state.set_time(pd.to_datetime(order.order_time))
        
        # Get eligible options
        eligible_start = time.perf_counter()
        eligible_option_ids = self.catalog.eligible_for_order(order)
        stage_durations['eligible_lookup'] = time.perf_counter() - eligible_start
        self._record_stage_time('eligible_lookup', stage_durations['eligible_lookup'], order.order_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Order %s has %d eligible options (lookup %.4fs)",
                order.order_id,
                len(eligible_option_ids),
                stage_durations['eligible_lookup'],
            )
        if not eligible_option_ids:
            self.orders_no_eligible_this_run += 1
            # No eligible options, all lost sales
            decision = OrderDecision(
                allocations=[],
                unfilled={item.sku_id: item.quantity for item in order.items}
            )
            if logger.isEnabledFor(logging.INFO):
                logger.info("Order %s has no eligible options; recording lost sale.", order.order_id)
            if not collect_only:
                record_start = time.perf_counter()
                self._record_results(order, decision, num_replications)
                stage_durations['record_results'] = time.perf_counter() - record_start
                self._record_stage_time('record_results', stage_durations['record_results'], order.order_id)
            stage_durations['order_total'] = time.perf_counter() - order_start
            self._record_stage_time('order_total', stage_durations['order_total'], order.order_id)
            self._log_order_stage_durations(order.order_id, stage_durations)
            self._log_slow_order(order.order_id, stage_durations)
            return
        
        # Call policy (expects only the Order object).
        # Supported returns:
        #   OrderDecision
        #   (OrderDecision, runtime_seconds)
        #   (OrderDecision, runtime_seconds, stats_dict)
        policy_start = time.perf_counter()
        policy_result = self.policy(order)
        stage_durations['policy'] = time.perf_counter() - policy_start
        self._record_stage_time('policy', stage_durations['policy'], order.order_id)
        policy_stats: Dict[str, Any] = {}
        if isinstance(policy_result, tuple):
            if len(policy_result) >= 3:
                decision, policy_runtime, maybe_stats = policy_result[0], policy_result[1], policy_result[2]
                if isinstance(maybe_stats, dict):
                    policy_stats = maybe_stats
            elif len(policy_result) == 2:
                decision, policy_runtime = policy_result
            elif len(policy_result) == 1:
                decision = policy_result[0]
                policy_runtime = 0.0
            else:
                decision = OrderDecision(allocations=[], unfilled={item.sku_id: item.quantity for item in order.items})
                policy_runtime = 0.0
        else:
            # Backward compatibility: if policy returns just OrderDecision
            decision = policy_result
            policy_runtime = 0.0

        implied_unfilled = self._compute_implied_unfilled(order, decision)
        if (decision.unfilled or {}) != implied_unfilled:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Order %s overriding policy unfilled (policy=%s) with implied_unfilled=%s",
                    order.order_id,
                    decision.unfilled,
                    implied_unfilled,
                )
            decision.unfilled = implied_unfilled or None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Order %s policy completed in %.4fs (reported %.4fs) -> allocations=%d, unfilled_qty=%d",
                order.order_id,
                stage_durations['policy'],
                policy_runtime,
                len(decision.allocations),
                sum(decision.unfilled.values()) if decision.unfilled else 0,
            )
            if decision.allocations:
                self._log_allocation_detail(order.order_id, decision.allocations)
        
        # Store runtime for this order
        runtime_row: Dict[str, Any] = {
            'order_id': order.order_id,
            'runtime_seconds': float(policy_runtime) if policy_runtime is not None else 0.0,
            'eligible_option_count': int(len(eligible_option_ids)),
            'order_item_count': int(item_count),
            'order_total_qty': int(total_qty),
        }
        if policy_stats:
            for key, value in policy_stats.items():
                if isinstance(value, np.generic):
                    runtime_row[key] = value.item()
                else:
                    runtime_row[key] = value
        self.order_runtimes.append(runtime_row)
        
        # Estimate per-DC processing delays
        process_time_map: Dict[int, float] = {}
        if decision.allocations:
            qty_by_dc: Dict[int, int] = defaultdict(int)
            for alloc in decision.allocations:
                dc_id = alloc.option_id[0]
                qty_by_dc[dc_id] += alloc.quantity
            for dc_id, total_qty in qty_by_dc.items():
                waiting_units = self.state.get_total_waiting_quantity(dc_id)
                order_time = None if self.state.now == datetime.min else self.state.now
                process_time_map[dc_id] = self.processing_model.estimate_minutes(
                    dc_id=dc_id,
                    order_time=order_time,
                    quantity=total_qty,
                    waiting_units=waiting_units,
                )
        
        # Apply decision to state
        apply_start = time.perf_counter()
        if process_time_map:
            self.state.apply_decision(order.order_id, decision, process_time_minutes=process_time_map)
        else:
            self.state.apply_decision(order.order_id, decision)
        stage_durations['apply_decision'] = time.perf_counter() - apply_start
        self._record_stage_time('apply_decision', stage_durations['apply_decision'], order.order_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Order %s state updated (apply_decision=%.4fs, process_time_map=%s)",
                order.order_id,
                stage_durations['apply_decision'],
                process_time_map if process_time_map else "{}",
            )
            self._log_state_snapshot(order.order_id, decision)
        
        # Record results (skip in collect_only mode)
        if not collect_only:
            record_start = time.perf_counter()
            self._record_results(order, decision, num_replications)
            stage_durations['record_results'] = time.perf_counter() - record_start
            self._record_stage_time('record_results', stage_durations['record_results'], order.order_id)
        
        stage_durations['order_total'] = time.perf_counter() - order_start
        self._record_stage_time('order_total', stage_durations['order_total'], order.order_id)
        self._log_order_stage_durations(order.order_id, stage_durations)
        self._log_slow_order(order.order_id, stage_durations)
    
    def _record_results(self, order: Order, decision: OrderDecision, num_replications: int):
        """Record results for an order across replications."""
        # Sample outcomes for allocated options
        allocation_metadata: List[Dict] = []
        option_totals: Dict[Tuple[int, int], int] = {}
        option_base_costs: Dict[Tuple[int, int], float] = {}
        delivery_sample_map: Dict[Tuple[int, int], Optional[np.ndarray]] = {}

        if decision.allocations:
            # Build features for allocated options only
            allocated_option_ids = [alloc.option_id for alloc in decision.allocations]
            allocation_metadata, option_totals, option_base_costs = self._build_allocation_metadata(
                order,
                decision.allocations,
            )
            features_df = build_features(
                order, allocated_option_ids, self.state, self.catalog, self.precompute
            )
            if features_df.index.has_duplicates:
                duplicates = features_df.index[features_df.index.duplicated()].unique().tolist()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Order %s has %d duplicate option rows for delivery sampling; keeping first occurrence. Duplicates: %s",
                        order.order_id,
                        len(duplicates),
                        duplicates,
                    )
                features_df = features_df[~features_df.index.duplicated(keep='first')]
            
            # Sample delivery times
            delivery_samples = self.sampler.sample(features_df, num_replications, self.rng)
            delivery_sample_map = self._build_delivery_sample_map(
                delivery_samples,
                allocated_option_ids,
                num_replications,
                order,
            )
        else:
            delivery_samples = pd.DataFrame()
        
        # Compute costs for each replication
        for rep in range(num_replications):
            result = {
                'order_id': order.order_id,
                'replication': rep,
                'allocations': [],
                'unfilled': decision.unfilled or {},
                'realized_cost': 0.0,
                'lost_sales_quantity': sum(decision.unfilled.values()) if decision.unfilled else 0.0,
                'late_delivery_pct': 0.0,
                'cumulative_lateness': 0.0,
            }
            
            # Process allocations
            total_shipping_cost = 0.0
            total_units = 0
            late_units = 0
            cumulative_lateness = 0.0
            
            for alloc_meta in allocation_metadata:
                opt_id = alloc_meta['option_id']
                sku_id = alloc_meta['sku_id']
                qty = alloc_meta['quantity']
                base_cost = alloc_meta['base_cost']
                
                # Get delivery time sample
                sample_arr = delivery_sample_map.get(opt_id)
                if sample_arr is not None and rep < len(sample_arr):
                    delivered_days = int(sample_arr[rep])
                else:
                    delivered_days = order.promise_delivery_days  # Fallback
                
                # Compute service penalty
                deviation = delivered_days - order.promise_delivery_days
                penalty = (
                    cfg.GAMMA_PLUS_LATE_PENALTY * max(0, deviation) +
                    cfg.GAMMA_MINUS_EARLY_PENALTY * max(0, -deviation)
                )
                unit_cost = base_cost + penalty
                shipping_cost = unit_cost * qty
                total_shipping_cost += shipping_cost
                
                # Track delivery metrics
                total_units += qty
                if deviation > 0:
                    late_units += qty
                    cumulative_lateness += qty * deviation
                
                result['allocations'].append({
                    'sku_id': sku_id,
                    'dc_id': opt_id[0],
                    'carrier_service_id': opt_id[1],
                    'quantity': qty,
                    'base_cost': base_cost,
                    'unit_cost': unit_cost,
                    'delivered_days': delivered_days,
                })
            
            # Compute consolidation discount
            # Group by (dc_id, carrier_service_id) and check if >= 2 units
            consolidation_discount = 0.0
            for opt_id, total_qty in option_totals.items():
                if total_qty >= 2:
                    base_cost = option_base_costs.get(opt_id)
                    if base_cost is None:
                        base_cost = self._get_base_cost(opt_id, order)
                        option_base_costs[opt_id] = base_cost
                    consolidation_discount += base_cost * total_qty * cfg.BETA_DISCOUNT
            
            # Lost sales penalty
            lost_sales_penalty = (
                sum(decision.unfilled.values()) * cfg.STOCKOUT_PENALTY_PER_UNIT
                if decision.unfilled else 0.0
            )
            
            # Total cost
            result['realized_cost'] = (
                total_shipping_cost - consolidation_discount + lost_sales_penalty
            )
            result['late_delivery_pct'] = (
                (late_units / total_units * 100.0) if total_units > 0 else 0.0
            )
            result['cumulative_lateness'] = cumulative_lateness
            
            self.results.append(result)
    
    def _log_state_snapshot(self, order_id: str, decision: OrderDecision):
        """Emit a concise snapshot of key state metrics for touched DCs."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if not decision.allocations:
            logger.debug("Order %s state snapshot: no allocations applied.", order_id)
            return
        dc_sku_map: DefaultDict[int, set[str]] = defaultdict(set)
        for alloc in decision.allocations:
            dc_id = int(alloc.option_id[0])
            dc_sku_map[dc_id].add(str(alloc.sku_id))
        snapshots = []
        for dc_id in sorted(dc_sku_map.keys()):
            sku_parts = []
            for sku_id in sorted(dc_sku_map[dc_id]):
                inv = self.state.inventory.get(dc_id, {}).get(sku_id, 0)
                waiting = self.state.waiting_quantity.get((dc_id, sku_id), 0)
                sku_parts.append(f"{sku_id}:inv={inv},wait={waiting}")
            pending = len(self.state.pending_shipments.get(dc_id, []))
            waiting_total = self.state.waiting_total_by_dc.get(dc_id, 0)
            waiting_orders = len(self.state.waiting_order_ids.get(dc_id, set()))
            ship_orders_2h, ship_skus_2h = self._recent_shipments_for_dc(dc_id)
            baseline_row = None
            if getattr(self.state, "_baseline_dynamic_features", None) is not None:
                baseline_df = self.state._baseline_dynamic_features
                if dc_id in baseline_df.index:
                    baseline_row = baseline_df.loc[dc_id]
            parts = [
                f"pending={pending}",
                f"wait_orders={self._format_with_baseline(waiting_orders, baseline_row, 'waiting_orders')}",
                f"wait_units={self._format_with_baseline(waiting_total, baseline_row, 'waiting_skus')}",
                f"shipped_orders_2h={self._format_with_baseline(ship_orders_2h, baseline_row, 'shipped_orders_last_2h')}",
                f"shipped_skus_2h={self._format_with_baseline(ship_skus_2h, baseline_row, 'shipped_skus_last_2h')}",
            ]
            snapshots.append(
                f"dc={dc_id} [{', '.join(parts)}] {{{'; '.join(sku_parts)}}}"
            )
        unfilled_str = ""
        if decision.unfilled:
            unfilled_pairs = ", ".join(
                f"{sku}:{qty}" for sku, qty in sorted(decision.unfilled.items())
            )
            unfilled_str = f" | unfilled={{{unfilled_pairs}}}"
        logger.debug(
            "Order %s state snapshot after apply: %s%s",
            order_id,
            " | ".join(snapshots),
            unfilled_str,
        )

    def _recent_shipments_for_dc(self, dc_id: int) -> Tuple[int, int]:
        """Return (orders_shipped, skus_shipped) in the last 2 hours for a DC."""
        order_ids = set()
        sku_ids = set()
        for (ship_dc, sku_id), shipments in self.state.shipped_quantity_last_2h.items():
            if ship_dc != dc_id:
                continue
            sku_ids.add(sku_id)
            for _, _, order_id in shipments:
                order_ids.add(order_id)
        return len(order_ids), len(sku_ids)

    def _format_with_baseline(self, actual: float, baseline_row: Optional[pd.Series], feature_name: str) -> str:
        """Format metric showing historical baseline when available."""
        baseline_val = None
        if baseline_row is not None and feature_name in baseline_row.index:
            try:
                baseline_val = float(baseline_row[feature_name])
            except (TypeError, ValueError):
                baseline_val = None
        baseline_repr = "NA" if baseline_val is None else self._format_scalar(baseline_val)
        return f"{self._format_scalar(actual)} (baseline={baseline_repr})"

    @staticmethod
    def _format_scalar(value: float) -> str:
        """Format numeric metric without trailing decimals for integers."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.2f}"
    
    def _summarize_items(self, items: List[OrderItem], max_items: int = 8) -> str:
        """Return a compact sku:qty summary for logging."""
        parts = [f"{item.sku_id}:{item.quantity}" for item in items[:max_items]]
        remaining = len(items) - max_items
        if remaining > 0:
            parts.append(f"...(+{remaining} more)")
        return "[" + ", ".join(parts) + "]"
    
    def _log_allocation_detail(self, order_id: str, allocations: List[ItemAllocation]):
        """Log detailed allocation mapping sku -> dc-carrier."""
        if not allocations:
            logger.debug("Order %s allocations: <none>", order_id)
            return
        parts = []
        for alloc in allocations:
            dc_id, carrier_id = alloc.option_id
            parts.append(f"{alloc.sku_id}->{dc_id}-{carrier_id} x{alloc.quantity}")
        logger.debug("Order %s allocations: %s", order_id, "; ".join(parts))

    def _init_timing_stats(self):
        self._timing_stats = defaultdict(lambda: {
            'total': 0.0,
            'count': 0,
            'max': 0.0,
            'max_order': None,
        })

    def _record_stage_time(self, stage: str, duration: float, order_id):
        if self._timing_stats is None or duration is None:
            return
        stats = self._timing_stats[stage]
        stats['total'] += duration
        stats['count'] += 1
        if duration > stats['max']:
            stats['max'] = duration
            stats['max_order'] = order_id

    def _log_order_stage_durations(self, order_id, stage_durations: Dict[str, float]):
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Order %s timings: eligible=%.4fs policy=%.4fs apply=%.4fs record=%.4fs total=%.4fs",
            order_id,
            stage_durations.get('eligible_lookup', 0.0),
            stage_durations.get('policy', 0.0),
            stage_durations.get('apply_decision', 0.0),
            stage_durations.get('record_results', 0.0),
            stage_durations.get('order_total', 0.0),
        )

    def _log_slow_order(self, order_id, stage_durations: Dict[str, float]):
        threshold = getattr(self, "_slow_order_seconds", None)
        total_time = stage_durations.get('order_total')
        if (
            threshold is None or threshold <= 0 or
            total_time is None or total_time < threshold
        ):
            return
        logger.warning(
            "Order %s slow path: total %.3fs (eligible=%.3fs, policy=%.3fs, apply_decision=%.3fs, record_results=%.3fs)",
            order_id,
            total_time,
            stage_durations.get('eligible_lookup', 0.0),
            stage_durations.get('policy', 0.0),
            stage_durations.get('apply_decision', 0.0),
            stage_durations.get('record_results', 0.0),
        )

    def _log_timing_summary(self):
        if not self._timing_stats or not logger.isEnabledFor(logging.INFO):
            return
        for stage, stats in sorted(self._timing_stats.items()):
            if stats['count'] == 0:
                continue
            avg_duration = stats['total'] / stats['count']
            logger.info(
                "Timing summary [%s]: avg=%.4fs max=%.4fs (order=%s) samples=%d",
                stage,
                avg_duration,
                stats['max'],
                stats['max_order'],
                stats['count'],
            )

    def _load_cost_model_coeffs(self) -> Dict[int, float]:
        try:
            df = pd.read_csv(cfg.REAL_COST_MODELS_CS_PATH)
        except FileNotFoundError:
            return {}
        coeffs = {}
        if 'carrier_service_id' not in df.columns or 'coef_distance_km' not in df.columns:
            return coeffs
        for _, row in df.iterrows():
            try:
                carrier = int(row['carrier_service_id'])
                coef = float(row['coef_distance_km'])
            except (ValueError, TypeError):
                continue
            coeffs[carrier] = coef
        return coeffs

    def _get_base_cost(self, opt_id: Tuple[int, int], order: Order) -> float:
        dc_id, carrier_id = opt_id
        fixed_cost = get_per_unit_base_cost(dc_id)
        cached = self.precompute.get_cost(dc_id, carrier_id, order.dest_zip5)
        if cached is not None:
            return float(cached) + fixed_cost
        opt = self.catalog.index[opt_id]
        dist = self.precompute.get_distance_km(opt.dc_id, order.dest_zip5)
        if dist is None:
            dist = float(compute_haversine_km(
                np.array([opt.dc_lat]), np.array([opt.dc_lng]),
                order.dest_lat, order.dest_lng
            )[0])
        carrier_coef = self._carrier_cost_coef.get(int(opt.carrier_service_id))
        if carrier_coef is None:
            variable_cost = 0.0
        else:
            variable_cost = dist * carrier_coef
        return variable_cost + fixed_cost

    def _build_allocation_metadata(
        self,
        order: Order,
        allocations: List[ItemAllocation],
    ) -> Tuple[List[Dict], Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]]:
        metadata: List[Dict] = []
        option_totals: Dict[Tuple[int, int], int] = {}
        option_base_costs: Dict[Tuple[int, int], float] = {}
        for alloc in allocations:
            opt_id = alloc.option_id
            base_cost = self._get_base_cost(opt_id, order)
            metadata.append({
                'option_id': opt_id,
                'sku_id': alloc.sku_id,
                'quantity': alloc.quantity,
                'base_cost': base_cost,
            })
            option_totals[opt_id] = option_totals.get(opt_id, 0) + alloc.quantity
            if opt_id not in option_base_costs:
                option_base_costs[opt_id] = base_cost
        return metadata, option_totals, option_base_costs

    def _build_delivery_sample_map(
        self,
        delivery_samples: pd.DataFrame,
        option_ids: List[Tuple[int, int]],
        num_replications: int,
        order: Order,
    ) -> Dict[Tuple[int, int], Optional[np.ndarray]]:
        if delivery_samples.empty:
            return {}
        # Avoid expensive fallback path when MultiIndex is unsorted
        if isinstance(delivery_samples.index, pd.MultiIndex):
            # pandas warns about lexsort depth when the index is unsorted
            if not delivery_samples.index.is_monotonic_increasing:
                delivery_samples = delivery_samples.sort_index()
        sample_map: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        for opt_id in dict.fromkeys(option_ids):
            sample_arr = self._extract_delivery_array(delivery_samples, opt_id)
            if sample_arr is None or len(sample_arr) < num_replications:
                if sample_arr is None:
                    logger.warning(
                        "Missing delivery samples for option %s; using promise days for order %s.",
                        opt_id,
                        order.order_id,
                    )
                else:
                    logger.warning(
                        "Insufficient delivery samples for option %s (have %d, need %d); using promise days for remaining reps.",
                        opt_id,
                        len(sample_arr),
                        num_replications,
                    )
                    sample_arr = None
            sample_map[opt_id] = sample_arr
        return sample_map

    def _extract_delivery_array(
        self,
        delivery_samples: pd.DataFrame,
        opt_id: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        try:
            if opt_id in delivery_samples.index:
                sample_value = delivery_samples.loc[opt_id]
            else:
                sample_value = delivery_samples.xs(opt_id)
        except KeyError:
            return None
        if isinstance(sample_value, pd.DataFrame):
            sample_value = sample_value.iloc[0]
        if isinstance(sample_value, pd.Series):
            return sample_value.to_numpy(copy=True)
        if isinstance(sample_value, np.ndarray):
            return sample_value.astype(float, copy=True)
        try:
            return np.array([float(sample_value)], dtype=float)
        except (TypeError, ValueError):
            return None
