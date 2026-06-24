"""Processing time estimator for simulation queueing logic."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import src.config as cfg


MinutesPerUnit = float


@dataclass
class _LevelSpec:
    name: str
    key_builder: Callable[[int, int, int], Tuple[int, int, int]]


class ProcessingTimeModel:
    """Estimate per-DC processing delays using historical rates with queue adjustments."""
    
    def __init__(self, rates_df: Optional[pd.DataFrame] = None):
        sentinel = cfg.PROCESSING_SENTINEL_VALUE
        self._tables: Dict[str, Dict[Tuple[int, int, int], MinutesPerUnit]] = {}
        self._samples: Dict[str, Dict[Tuple[int, int, int], int]] = {}
        if rates_df is not None and not rates_df.empty:
            for row in rates_df.itertuples():
                level = getattr(row, 'level')
                key = (
                    int(getattr(row, 'dc_id', sentinel)),
                    int(getattr(row, 'order_hour', sentinel)),
                    int(getattr(row, 'is_weekend', sentinel)),
                )
                minutes_per_unit = float(getattr(row, 'avg_minutes_per_unit'))
                if minutes_per_unit <= 0:
                    continue
                self._tables.setdefault(level, {})[key] = minutes_per_unit
                samples = int(getattr(row, 'samples', 0))
                self._samples.setdefault(level, {})[key] = samples
        
        self._fallback_levels = [
            _LevelSpec('dc_hour_weekend', lambda d, h, w: (d, h, w)),
            _LevelSpec('dc_hour', lambda d, h, w: (d, h, sentinel)),
            _LevelSpec('dc_weekend', lambda d, h, w: (d, sentinel, w)),
            _LevelSpec('dc', lambda d, h, w: (d, sentinel, sentinel)),
            _LevelSpec('hour_weekend', lambda d, h, w: (sentinel, h, w)),
            _LevelSpec('hour', lambda d, h, w: (sentinel, h, sentinel)),
            _LevelSpec('weekend', lambda d, h, w: (sentinel, sentinel, w)),
            _LevelSpec('global', lambda d, h, w: (sentinel, sentinel, sentinel)),
        ]
    
    def has_data(self) -> bool:
        """Return True if any historical rate is available."""
        return bool(self._tables)
    
    def estimate_minutes(self, dc_id: int, order_time, quantity: int, waiting_units: int = 0) -> float:
        """
        Estimate processing minutes for a DC, adjusting for queued work.
        
        Args:
            dc_id: Fulfillment DC
            order_time: Timestamp of the order arrival
            quantity: Total units allocated to this DC in the current order
            waiting_units: Units already waiting in this DC's queue (from state)
        """
        minutes_per_unit = self._lookup_minutes_per_unit(dc_id, order_time)
        qty = max(int(quantity), 0)
        queue_units = max(int(waiting_units), 0)
        
        base_minutes = max(qty * minutes_per_unit, cfg.PROCESSING_MIN_SERVICE_MINUTES)
        queue_delay = queue_units * minutes_per_unit * cfg.PROCESSING_QUEUE_SAFETY_FACTOR
        total = base_minutes + queue_delay
        return min(total, cfg.PROCESSING_MAX_SERVICE_MINUTES)
    
    def _lookup_minutes_per_unit(self, dc_id: int, order_time) -> float:
        if not self._tables:
            return cfg.PROCESSING_FALLBACK_MINUTES_PER_UNIT
        
        timestamp = pd.Timestamp(order_time) if order_time is not None else None
        hour = int(timestamp.hour) if timestamp is not None else 0
        is_weekend = int(timestamp.weekday() >= 5) if timestamp is not None else 0
        
        for level_spec in self._fallback_levels:
            table = self._tables.get(level_spec.name)
            if not table:
                continue
            key = level_spec.key_builder(dc_id, hour, is_weekend)
            minutes_per_unit = table.get(key)
            if minutes_per_unit is None:
                continue
            sample_count = self._samples.get(level_spec.name, {}).get(key, 0)
            if sample_count < cfg.PROCESSING_RATE_MIN_SAMPLES and level_spec.name != 'global':
                continue
            return max(minutes_per_unit, cfg.PROCESSING_FALLBACK_MINUTES_PER_UNIT)
        
        return cfg.PROCESSING_FALLBACK_MINUTES_PER_UNIT
