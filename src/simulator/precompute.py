"""Precomputed artifacts store for fast simulation."""

from pathlib import Path
from typing import Dict, List, Optional, Set
import logging
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pyarrow.feather as feather  # type: ignore
except Exception:
    feather = None

import src.config as cfg


class PrecomputeStore:
    """Store for precomputed eligibility, costs, and static features."""
    
    def __init__(self, precompute_dir: Optional[Path] = None):
        """
        Initialize the precompute store.
        
        Args:
            precompute_dir: Directory containing precomputed artifacts.
        """
        self.precompute_dir = precompute_dir or cfg.SIM_PRECOMPUTE_DIR
        self._static_features: Optional[pd.DataFrame] = None
        self._coverage: Optional[pd.DataFrame] = None
        self._distances: Optional[pd.DataFrame] = None
        self._costs: Optional[pd.DataFrame] = None
        self._option_static: Optional[pd.DataFrame] = None
        self._processing_rates: Optional[pd.DataFrame] = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all precomputed artifacts."""
        # Static features
        static_path = self.precompute_dir / 'orders_static_features.parquet'
        if static_path.exists():
            try:
                self._static_features = pd.read_parquet(static_path)
                if 'order_id' in self._static_features.columns:
                    self._static_features.set_index('order_id', inplace=True)
            except Exception as e:
                logger.warning("Failed to read %s (%s). Continuing without static features.", static_path, e)
                self._static_features = None
        
        # Coverage
        coverage_path = self.precompute_dir / 'coverage_allowed_states.parquet'
        if coverage_path.exists():
            try:
                self._coverage = pd.read_parquet(coverage_path)
            except Exception as e:
                logger.warning("Failed to read %s (%s). Falling back to rule-based eligibility.", coverage_path, e)
                self._coverage = None
        
        # Distances
        dist_path = self.precompute_dir / 'dc_zip5_distance.feather'
        if dist_path.exists() and feather is not None:
            try:
                self._distances = feather.read_feather(dist_path)
                if 'dc_id' in self._distances.columns and 'customer_zip5' in self._distances.columns:
                    self._distances.set_index(['dc_id', 'customer_zip5'], inplace=True)
            except Exception as e:
                logger.warning("Failed to read %s (%s). Will compute distances on the fly.", dist_path, e)
                self._distances = None
        elif dist_path.exists() and feather is None:
            logger.warning("pyarrow not available; cannot read %s. Will compute distances on the fly.", dist_path)
        
        # Costs (optional)
        cost_path = self.precompute_dir / 'cost_dc_zip5_cs.feather'
        if cost_path.exists() and cfg.SIM_USE_PRECOMPUTED_COSTS and feather is not None:
            try:
                self._costs = feather.read_feather(cost_path)
                if 'dc_id' in self._costs.columns and 'customer_zip5' in self._costs.columns and 'carrier_service_id' in self._costs.columns:
                    self._costs.set_index(['dc_id', 'customer_zip5', 'carrier_service_id'], inplace=True)
            except Exception as e:
                logger.warning("Failed to read %s (%s). Will compute costs on the fly.", cost_path, e)
                self._costs = None
        elif cost_path.exists() and cfg.SIM_USE_PRECOMPUTED_COSTS and feather is None:
            logger.warning("pyarrow not available; cannot read %s. Will compute costs on the fly.", cost_path)
        
        # Option static flags
        option_static_path = self.precompute_dir / 'option_static_flags.parquet'
        if option_static_path.exists():
            try:
                self._option_static = pd.read_parquet(option_static_path)
                if 'option_id' in self._option_static.columns:
                    self._option_static.set_index('option_id', inplace=True)
            except Exception as e:
                logger.warning("Failed to read %s (%s). Continuing without option static flags.", option_static_path, e)
                self._option_static = None
        
        # Processing rates
        processing_rates_path = self.precompute_dir / 'dc_processing_rates.parquet'
        if processing_rates_path.exists():
            try:
                self._processing_rates = pd.read_parquet(processing_rates_path)
            except Exception as e:
                logger.warning("Failed to read %s (%s). Continuing without processing rates.", processing_rates_path, e)
                self._processing_rates = None
    
    def get_static_features(self, order_id: str) -> Optional[Dict[str, float]]:
        """
        Get static features for an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Dictionary of static features or None if not found
        """
        if self._static_features is None:
            return None
        
        try:
            row = self._static_features.loc[order_id]
            return row.to_dict()
        except KeyError:
            return None
    
    @staticmethod
    def _coverage_with_option_columns(coverage: pd.DataFrame) -> pd.DataFrame:
        if {'dc_id', 'carrier_service_id'}.issubset(coverage.columns):
            return coverage
        index_names = set(name for name in coverage.index.names if name is not None)
        if {'dc_id', 'carrier_service_id'}.issubset(index_names):
            return coverage.reset_index()
        return coverage

    @staticmethod
    def _allowed_states_mask(states: pd.Series, dest_state: str) -> pd.Series:
        states_clean = states.fillna('').astype(str).str.upper()
        dest_state_upper = str(dest_state or '').strip().upper()
        all_mask = states_clean.eq('ALL')
        if not dest_state_upper:
            return all_mask

        token_mask = states_clean.str.split('|').apply(
            lambda tokens: dest_state_upper in {token.strip() for token in tokens}
        )
        return all_mask | token_mask

    def get_eligible_options(self, dest_zip5: str, dest_state: str) -> Optional[List[tuple]]:
        """
        Get eligible options for a destination.
        
        New coverage artifacts are full option-level tables with one row per
        (dc_id, carrier_service_id). Older dest_zip3 tables are still supported.
        Restricted-only dc_zip3 artifacts are intentionally ignored so runtime
        falls back to source coverage rules instead of dropping unrestricted
        options.
        
        Args:
            dest_zip5: Destination ZIP5
            dest_state: Destination state
            
        Returns:
            List of (dc_id, carrier_service_id) tuples or None if not precomputed
        """
        if self._coverage is None:
            return None
        
        coverage = self._coverage
        coverage_scope = coverage.get('coverage_scope')
        if coverage_scope is not None and coverage_scope.astype(str).eq('all_options').all():
            coverage = self._coverage_with_option_columns(coverage)
            if not {'dc_id', 'carrier_service_id', 'allowed_states'}.issubset(coverage.columns):
                return None
            eligible = coverage[self._allowed_states_mask(coverage['allowed_states'], dest_state)]
            return [
                (int(row.dc_id), int(row.carrier_service_id))
                for row in eligible.itertuples(index=False)
            ]

        # Legacy dest_zip3 table.
        dest_zip3 = dest_zip5[:3] if dest_zip5 and len(dest_zip5) >= 3 else None
        if dest_zip3 is None:
            return []
        
        if 'dest_zip3' in self._coverage.columns:
            eligible = self._coverage[
                (self._coverage['dest_zip3'] == dest_zip3) &
                self._allowed_states_mask(self._coverage['allowed_states'], dest_state)
            ]
        else:
            # Old restricted-only dc_zip3 artifacts are not full eligibility
            # tables. Fall back to JSON rules to preserve unrestricted options.
            return None
        
        if eligible.empty:
            return []
        
        return [
            (int(row['dc_id']), int(row['carrier_service_id']))
            for _, row in eligible.iterrows()
        ]
    
    def get_distance_km(self, dc_id: int, dest_zip5: str) -> Optional[float]:
        """
        Get distance in km between DC and destination.
        
        Args:
            dc_id: DC ID
            dest_zip5: Destination ZIP5
            
        Returns:
            Distance in km or None if not found
        """
        if self._distances is None or not cfg.SIM_USE_PRECOMPUTED_DIST:
            return None
        
        try:
            return float(self._distances.loc[(dc_id, dest_zip5), 'distance_km'])
        except (KeyError, IndexError):
            return None
    
    def get_cost(self, dc_id: int, carrier_service_id: int, dest_zip5: str) -> Optional[float]:
        """
        Get precomputed cost for an option.
        
        Args:
            dc_id: DC ID
            carrier_service_id: Carrier service ID
            dest_zip5: Destination ZIP5
            
        Returns:
            Unit cost or None if not precomputed
        """
        if self._costs is None:
            return None
        
        try:
            return float(self._costs.loc[(dc_id, dest_zip5, carrier_service_id), 'unit_cost'])
        except (KeyError, IndexError):
            return None
    
    def get_option_static(self, option_id: tuple) -> Optional[Dict]:
        """
        Get static flags/attributes for an option.
        
        Args:
            option_id: (dc_id, carrier_service_id) tuple
            
        Returns:
            Dictionary of static attributes or None
        """
        if self._option_static is None:
            return None
        
        try:
            row = self._option_static.loc[option_id]
            return row.to_dict()
        except KeyError:
            return None

    def get_processing_rates(self) -> Optional[pd.DataFrame]:
        """Return the processing rate lookup table, if available."""
        return self._processing_rates
