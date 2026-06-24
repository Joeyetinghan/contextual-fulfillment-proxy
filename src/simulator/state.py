"""Simulation state management."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Iterable
import pandas as pd
import logging

import src.config as cfg
from src.simulator.entities import OrderDecision, ItemAllocation

logger = logging.getLogger(__name__)


def _inventory_default_factory():
    """Pickle-friendly default factory for the inventory map."""
    return defaultdict(int)

def initialize_dynamic_features_from_history(
    simulation_date: str,
    snapshot_hour: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    """
    Initialize dynamic features from the carrier-aware `preprocessed_data_cs.csv`.
    
    This snapshot captures backlog and recent shipments immediately before `simulation_date`.
    When `snapshot_hour` is provided the snapshot time is shifted to that hour on
    the simulation date instead of midnight.
    """
    data_path = cfg.PROCESSED_DATA_DIR / 'preprocessed_data_cs.csv'
    usecols = [
        'order_ID', 'order_time', 'ship_out_time', 'dc_ori', 'sku_ID', 'quantity'
    ]
    
    try:
        orders_df = pd.read_csv(
            data_path,
            usecols=usecols,
            parse_dates=['order_time', 'ship_out_time'],
        )
    except FileNotFoundError:
        logger.warning("preprocessed data not found at %s; no historical baseline.", data_path)
        return None
    except Exception as e:
        logger.warning("Could not load preprocessed data: %s", e)
        return None
    
    snapshot_time = pd.to_datetime(simulation_date).normalize()
    if snapshot_hour is not None:
        try:
            hour = int(snapshot_hour)
        except (TypeError, ValueError):
            hour = None
        else:
            hour = max(0, min(23, hour))
        if hour is not None:
            snapshot_time = snapshot_time + pd.Timedelta(hours=hour)
    orders_df.dropna(subset=['order_time', 'dc_ori'], inplace=True)
    orders_df['dc_ori'] = pd.to_numeric(orders_df['dc_ori'], errors='coerce')
    orders_df.dropna(subset=['dc_ori'], inplace=True)
    if orders_df.empty:
        logger.warning(
            "No historical orders prior to %s; baseline disabled.",
            snapshot_time.date(),
        )
        return None
    orders_df['dc_ori'] = orders_df['dc_ori'].astype(int)
    
    historical_orders = orders_df[orders_df['order_time'] < snapshot_time].copy()
    if historical_orders.empty:
        return None
    
    pending_mask = historical_orders['ship_out_time'].isna() | (historical_orders['ship_out_time'] >= snapshot_time)
    pending_orders = historical_orders.loc[pending_mask].copy()
    
    waiting_orders = pending_orders.groupby('dc_ori')['order_ID'].nunique()
    if 'quantity' in pending_orders.columns:
        pending_orders['quantity'] = pd.to_numeric(pending_orders['quantity'], errors='coerce').fillna(0)
        waiting_units = pending_orders.groupby('dc_ori')['quantity'].sum()
    else:
        waiting_units = pending_orders.groupby('dc_ori')['sku_ID'].count()
    
    window_start = snapshot_time - timedelta(hours=2)
    shipped_mask = (
        historical_orders['ship_out_time'].notna()
        & (historical_orders['ship_out_time'] < snapshot_time)
        & (historical_orders['ship_out_time'] >= window_start)
    )
    shipped_window = historical_orders.loc[shipped_mask]
    shipped_orders = shipped_window.groupby('dc_ori')['order_ID'].nunique()
    shipped_skus = shipped_window.groupby('dc_ori')['sku_ID'].nunique()
    
    dc_ids = sorted(historical_orders['dc_ori'].unique())
    snapshot = pd.DataFrame({'dc_id': dc_ids})
    snapshot['waiting_orders'] = snapshot['dc_id'].map(waiting_orders).fillna(0).astype(float)
    snapshot['waiting_skus'] = snapshot['dc_id'].map(waiting_units).fillna(0).astype(float)
    snapshot['shipped_orders_last_2h'] = snapshot['dc_id'].map(shipped_orders).fillna(0).astype(float)
    snapshot['shipped_skus_last_2h'] = snapshot['dc_id'].map(shipped_skus).fillna(0).astype(float)
    
    feature_cols = ['dc_id'] + cfg.DYNAMIC_FEATURES
    
    waiting_quantity: Dict[Tuple[int, str], float] = defaultdict(float)
    waiting_total_by_dc: Dict[int, float] = defaultdict(float)
    waiting_order_ids: Dict[int, set[str]] = defaultdict(set)
    pending_shipments: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    shipped_history: Dict[Tuple[int, str], List[Tuple[datetime, float, str]]] = defaultdict(list)
    
    grouped = historical_orders.groupby('order_ID', sort=False)
    for order_id, group in grouped:
        if group.empty:
            continue
        try:
            dc_id = int(group['dc_ori'].iloc[0])
        except (ValueError, TypeError):
            continue
        ship_out = group['ship_out_time'].iloc[0]
        allocations: List[Tuple[str, float]] = []
        total_qty = 0.0
        for _, row in group.iterrows():
            sku = str(row.get('sku_ID', '')).strip()
            try:
                qty = float(row.get('quantity', 0) or 0)
            except (ValueError, TypeError):
                qty = 0.0
            allocations.append((sku, qty))
            total_qty += qty
        
        if pd.isna(ship_out) or ship_out >= snapshot_time:
            waiting_order_ids[dc_id].add(order_id)
            waiting_total_by_dc[dc_id] += total_qty
            for sku, qty in allocations:
                waiting_quantity[(dc_id, sku)] += qty
            completion_time = ship_out if pd.notna(ship_out) else snapshot_time + timedelta(minutes=5)
            pending_shipments[dc_id].append({
                'order_id': order_id,
                'completion_time': completion_time.to_pydatetime() if isinstance(completion_time, pd.Timestamp) else completion_time,
                'allocations': allocations,
            })
        elif ship_out >= window_start:
            ship_time = ship_out.to_pydatetime() if isinstance(ship_out, pd.Timestamp) else ship_out
            for sku, qty in allocations:
                shipped_history[(dc_id, sku)].append((ship_time, qty, order_id))
    
    logger.debug(
        "Historical baseline snapshot built for %d DCs using %d orders prior to %s.",
        len(snapshot),
        len(historical_orders),
        snapshot_time.date(),
    )
    
    return {
        'baseline_features': snapshot[feature_cols],
        'waiting_quantity': dict(waiting_quantity),
        'waiting_total_by_dc': dict(waiting_total_by_dc),
        'waiting_order_ids': {dc: set(orders) for dc, orders in waiting_order_ids.items()},
        'pending_shipments': dict(pending_shipments),
        'shipped_history': dict(shipped_history),
    }


class SimulationState:
    """Maintains simulation state: inventory, queues, and time."""
    
    def __init__(
        self,
        initial_inventory: pd.DataFrame,
        initial_dynamic_features: Optional[Union[pd.DataFrame, Dict[str, object]]] = None,
    ):
        """
        Initialize simulation state.
        
        Args:
            initial_inventory: DataFrame with MultiIndex (sku_id, dc_id) and column 'onhand_inventory'
            initial_dynamic_features: Optional DataFrame with initial dynamic feature values per DC
        """
        # Inventory: dc_id -> {sku_id: quantity}
        self.inventory: Dict[int, Dict[str, int]] = defaultdict(_inventory_default_factory)
        
        # Initialize from DataFrame
        if isinstance(initial_inventory.index, pd.MultiIndex):
            for (sku_id, dc_id), row in initial_inventory.iterrows():
                self.inventory[int(dc_id)][str(sku_id)] = int(row.get('onhand_inventory', 0))
        else:
            # Assume single-level index with dc_id column
            for idx, row in initial_inventory.iterrows():
                dc_id = int(row.get('dc_id', idx))
                sku_id = str(row.get('sku_ID', row.get('sku_id', '')))
                self.inventory[dc_id][sku_id] = int(row.get('onhand_inventory', 0))
        
        # Current time
        self.now: datetime = datetime.min
        
        # Baseline dynamic features from historical snapshot
        self._baseline_dynamic_features: Optional[pd.DataFrame] = None
        
        # Dynamic features maintained directly in state
        # shipped_quantity_last_2h: (dc_id, sku_id) -> list of (ship_time, quantity, order_id)
        self.shipped_quantity_last_2h: Dict[Tuple[int, str], List[Tuple[datetime, int, str]]] = defaultdict(list)
        
        # waiting_quantity: (dc_id, sku_id) -> quantity waiting in queue
        self.waiting_quantity: Dict[Tuple[int, str], int] = defaultdict(int)
        self.waiting_total_by_dc: Dict[int, int] = defaultdict(int)
        
        # Track order arrivals for waiting_orders count
        self.waiting_order_ids: Dict[int, set] = defaultdict(set)  # dc_id -> set of order_ids
        
        # Pending shipments keyed by dc_id
        self.pending_shipments: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        
        historical_state: Optional[Dict[str, object]] = None
        baseline_df: Optional[pd.DataFrame] = None
        if isinstance(initial_dynamic_features, dict):
            historical_state = initial_dynamic_features
            baseline_df = historical_state.get('baseline_features')  # type: ignore[assignment]
        else:
            baseline_df = initial_dynamic_features
        
        if baseline_df is not None and isinstance(baseline_df, pd.DataFrame):
            self._initialize_dynamic_features(baseline_df)
        if historical_state:
            self._seed_historical_state(historical_state)
    
    def _initialize_dynamic_features(self, snapshot_df: pd.DataFrame):
        """Initialize dynamic features from snapshot DataFrame."""
        if snapshot_df.empty:
            return
        
        snapshot = snapshot_df.copy()
        if 'snapshot_time' in snapshot.columns:
            snapshot.drop(columns=['snapshot_time'], inplace=True, errors='ignore')
        
        if 'dc_id' not in snapshot.columns:
            return
        
        snapshot = snapshot.set_index('dc_id')
        self._baseline_dynamic_features = snapshot
    
    def _seed_historical_state(self, historical_state: Dict[str, object]):
        """Populate waiting queues, pending shipments, and shipped history from snapshot."""
        waiting_qty = historical_state.get('waiting_quantity') or {}
        for key, qty in waiting_qty.items():
            dc_id, sku_id = key
            self.waiting_quantity[(int(dc_id), str(sku_id))] += int(round(float(qty)))
        
        waiting_totals = historical_state.get('waiting_total_by_dc') or {}
        for dc_id, qty in waiting_totals.items():
            self.waiting_total_by_dc[int(dc_id)] += int(round(float(qty)))
        
        waiting_orders = historical_state.get('waiting_order_ids') or {}
        for dc_id, orders in waiting_orders.items():
            self.waiting_order_ids[int(dc_id)].update(orders)
        
        pending = historical_state.get('pending_shipments') or {}
        for dc_id, events in pending.items():
            normalized_events = []
            for event in events:
                completion_time = event.get('completion_time')
                if isinstance(completion_time, pd.Timestamp):
                    completion_time = completion_time.to_pydatetime()
                allocations = [
                    (str(sku), int(round(float(qty))))
                    for sku, qty in event.get('allocations', [])
                ]
                normalized_events.append({
                    'order_id': event.get('order_id'),
                    'completion_time': completion_time,
                    'allocations': allocations,
                })
            self.pending_shipments[int(dc_id)].extend(normalized_events)
        
        shipped_history = historical_state.get('shipped_history') or {}
        for key, shipments in shipped_history.items():
            dc_id, sku_id = key
            normalized = []
            for ship_time, qty, order_id in shipments:
                if isinstance(ship_time, pd.Timestamp):
                    ship_time = ship_time.to_pydatetime()
                normalized.append((ship_time, int(round(float(qty))), order_id))
            self.shipped_quantity_last_2h[(int(dc_id), str(sku_id))].extend(normalized)
    
    def set_time(self, now: datetime):
        """Set current simulation time and clean up old shipped quantities."""
        if now < self.now:
            raise ValueError("Simulation state time cannot move backwards")
        
        self._process_completed_shipments(now)
        self.now = now
        self._cleanup_recent_shipments(now)
    
    def _cleanup_recent_shipments(self, now: datetime):
        """Drop shipment history entries older than 2 hours relative to `now`."""
        two_h_ago = now - timedelta(hours=2)
        for key in list(self.shipped_quantity_last_2h.keys()):
            self.shipped_quantity_last_2h[key] = [
                (ts, qty, order_id)
                for ts, qty, order_id in self.shipped_quantity_last_2h[key]
                if ts > two_h_ago
            ]
            if not self.shipped_quantity_last_2h[key]:
                del self.shipped_quantity_last_2h[key]
    
    def has_inventory(self, dc_id: int, sku_id: str, quantity: int) -> bool:
        """Check if DC has sufficient inventory."""
        return self.inventory.get(dc_id, {}).get(sku_id, 0) >= quantity
    
    def apply_decision(
        self,
        order_id: str,
        decision: OrderDecision,
        process_time_minutes: Union[float, Dict[int, float], None] = None,
    ):
        """
        Apply a fulfillment decision to the state.
        
        Args:
            order_id: Order ID
            decision: Fulfillment decision
            process_time_minutes: Either a scalar processing time (minutes) applied to all DCs
                or a dict mapping dc_id -> minutes for per-DC processing delays.
        """
        if not decision.allocations:
            return
        
        if isinstance(process_time_minutes, dict):
            dc_process_map = {
                int(dc): max(float(minutes), 0.0) for dc, minutes in process_time_minutes.items()
            }
            default_process_time = 0.0
        else:
            default_process_time = max(float(process_time_minutes or 0.0), 0.0)
            dc_process_map = {}
        
        allocations_by_dc: Dict[int, List[ItemAllocation]] = defaultdict(list)
        
        for alloc in decision.allocations:
            dc_id, _ = alloc.option_id
            allocations_by_dc[dc_id].append(alloc)
            
            # Decrement inventory immediately
            if dc_id in self.inventory and alloc.sku_id in self.inventory[dc_id]:
                self.inventory[dc_id][alloc.sku_id] = max(
                    0, self.inventory[dc_id][alloc.sku_id] - alloc.quantity
                )
        
        for dc_id, dc_allocations in allocations_by_dc.items():
            proc_minutes = dc_process_map.get(dc_id, default_process_time)
            completion_time = (
                self.now + timedelta(minutes=proc_minutes) if proc_minutes > 0 else self.now
            )
            
            # Register waiting quantities immediately
            for alloc in dc_allocations:
                self.add_waiting_order(order_id, dc_id, alloc.sku_id, alloc.quantity)
            
            event = {
                'order_id': order_id,
                'completion_time': completion_time,
                'allocations': [(alloc.sku_id, alloc.quantity) for alloc in dc_allocations],
            }
            
            if proc_minutes > 0:
                self.pending_shipments[dc_id].append(event)
            else:
                self._finalize_shipment(dc_id, event)
        
        # Add waiting quantities for unfilled items
        if decision.unfilled:
            for sku_id, unfilled_qty in decision.unfilled.items():
                # Find which DCs have this SKU in inventory (for waiting tracking)
                # For simplicity, we'll track waiting at order level, not per-DC
                # This could be refined if needed
                pass
    
    def add_waiting_order(self, order_id: str, dc_id: int, sku_id: str, quantity: int):
        """Add quantity to waiting queue for a DC-SKU pair."""
        self.waiting_quantity[(dc_id, sku_id)] += quantity
        self.waiting_total_by_dc[dc_id] += quantity
        self.waiting_order_ids[dc_id].add(order_id)
    
    def get_total_waiting_quantity(self, dc_id: int) -> int:
        """Return the total queued units for the given DC."""
        return self.waiting_total_by_dc.get(dc_id, 0)
    
    def _process_completed_shipments(self, up_to_time: datetime):
        """Release any shipments whose processing time has elapsed."""
        for dc_id in list(self.pending_shipments.keys()):
            remaining_events = []
            for event in self.pending_shipments[dc_id]:
                completion_time = event['completion_time']
                if completion_time <= up_to_time:
                    self._finalize_shipment(dc_id, event)
                else:
                    remaining_events.append(event)
            if remaining_events:
                self.pending_shipments[dc_id] = remaining_events
            else:
                del self.pending_shipments[dc_id]
    
    def _finalize_shipment(self, dc_id: int, event: Dict[str, object]):
        """Remove waiting quantities and record shipment metrics."""
        completion_time: datetime = event['completion_time']  # type: ignore[assignment]
        order_id: str = event['order_id']  # type: ignore[assignment]
        allocations: List[Tuple[str, int]] = event['allocations']  # type: ignore[assignment]
        
        for sku_id, qty in allocations:
            key = (dc_id, sku_id)
            self.waiting_quantity[key] = max(0, self.waiting_quantity.get(key, 0) - qty)
            if self.waiting_quantity[key] == 0:
                self.waiting_quantity.pop(key, None)
            
            self.waiting_total_by_dc[dc_id] = max(
                0, self.waiting_total_by_dc.get(dc_id, 0) - qty
            )
            if self.waiting_total_by_dc[dc_id] == 0:
                self.waiting_total_by_dc.pop(dc_id, None)
            
            self.shipped_quantity_last_2h[(dc_id, sku_id)].append((completion_time, qty, order_id))
        
        if order_id in self.waiting_order_ids.get(dc_id, set()):
            self.waiting_order_ids[dc_id].discard(order_id)
            if not self.waiting_order_ids[dc_id]:
                del self.waiting_order_ids[dc_id]
    
    def build_dc_event_snapshot(
        self,
        now: Optional[datetime] = None,
        dc_ids: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        """
        Build DC event snapshot for dynamic features from current state.
        
        Args:
            now: Time point for snapshot (default: self.now)
            
        Returns:
            DataFrame indexed by dc_id with dynamic features
        """
        if now is None:
            now = self.now
        
        inventory_dcs = set(self.inventory.keys())
        waiting_dcs = set(self.waiting_order_ids.keys()) | {
            dc for dc, _ in self.waiting_quantity.keys()
        } | set(self.waiting_total_by_dc.keys())
        shipped_dcs = {dc for (dc, _), _ in self.shipped_quantity_last_2h.items()}
        required_dcs = set(int(dc) for dc in (dc_ids or []))
        all_dcs = sorted(inventory_dcs | waiting_dcs | shipped_dcs | required_dcs)
        if not all_dcs:
            return pd.DataFrame(columns=['dc_id'] + cfg.DYNAMIC_FEATURES)
        
        snapshot_rows = []
        two_h_ago = now - timedelta(hours=2)
        
        for dc_id in all_dcs:
            # Count shipped orders in last 2h (unique order IDs)
            shipped_orders_set = set()
            shipped_skus_set = set()
            total_shipped_quantity = 0
            
            for (shipped_dc, shipped_sku), shipments in self.shipped_quantity_last_2h.items():
                if shipped_dc == dc_id:
                    for ship_time, qty, order_id in shipments:
                        if ship_time > two_h_ago and ship_time <= now:
                            shipped_skus_set.add(shipped_sku)
                            total_shipped_quantity += qty
                            shipped_orders_set.add(order_id)
            
            shipped_orders_last_2h = len(shipped_orders_set)
            shipped_skus_last_2h = len(shipped_skus_set)
            
            # Count waiting orders (unique order IDs)
            waiting_orders = len(self.waiting_order_ids.get(dc_id, set()))
            waiting_skus = self.waiting_total_by_dc.get(dc_id, 0)
            
            row = {
                'dc_id': dc_id,
                'waiting_orders': waiting_orders,
                'waiting_skus': waiting_skus,
                'shipped_orders_last_2h': shipped_orders_last_2h,
                'shipped_skus_last_2h': shipped_skus_last_2h,
            }
            
            if self._baseline_dynamic_features is not None:
                try:
                    baseline_row = self._baseline_dynamic_features.loc[dc_id]
                    row = self._apply_baseline(row, baseline_row)
                except KeyError:
                    pass
            snapshot_rows.append(row)
        
        return pd.DataFrame(snapshot_rows)
    
    def _apply_baseline(self, row: Dict[str, float], baseline_row: pd.Series) -> Dict[str, float]:
        """Blend baseline dynamic features into the current snapshot row."""
        blended = dict(row)
        for feat in cfg.DYNAMIC_FEATURES:
            if feat not in baseline_row.index:
                continue
            try:
                baseline_val = float(baseline_row[feat])
            except (TypeError, ValueError):
                continue
            blended[feat] = max(blended.get(feat, 0.0), baseline_val)
        return blended
