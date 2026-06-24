"""Global options catalog for fulfillment options."""

from pathlib import Path
from typing import List, Optional
import logging
import pandas as pd

import src.config as cfg
from src.simulator.entities import FulfillmentOption, Order, OptionId

logger = logging.getLogger(__name__)


class OptionsCatalog:
    """Global catalog of all fulfillment options."""
    
    def __init__(self, precompute_store=None):
        """
        Initialize the catalog.
        
        Args:
            precompute_store: Optional precompute store for eligibility lookups.
        """
        self.precompute_store = precompute_store
        self._options: List[FulfillmentOption] = []
        self._index: dict[OptionId, FulfillmentOption] = {}
        self._load_catalog()
    
    def _load_catalog(self):
        """Load options from precomputed catalog or build from source data."""
        # Prefer the caller-provided precompute directory (so simulations can run with
        # user-writable rebuilt artifacts when shared data/derived/sim is read-only).
        base_dir = getattr(self.precompute_store, 'precompute_dir', None) if self.precompute_store else None
        catalog_path = (Path(base_dir) / 'options_catalog.parquet') if base_dir else (cfg.SIM_PRECOMPUTE_DIR / 'options_catalog.parquet')
        
        if catalog_path.exists():
            try:
                df = pd.read_parquet(catalog_path)
            except ImportError as e:
                # Common failure mode in minimal environments: missing parquet engine (pyarrow/fastparquet).
                logger.warning(
                    "Unable to read %s (%s). Falling back to building options catalog from source data.",
                    catalog_path,
                    e,
                )
                df = None
            if df is not None:
                for _, row in df.iterrows():
                    opt = FulfillmentOption(
                        option_id=(int(row['dc_id']), int(row['carrier_service_id'])),
                        dc_id=int(row['dc_id']),
                        carrier_service_id=int(row['carrier_service_id']),
                        dc_zip3=str(row['dc_zip3']),
                        dc_lat=float(row['dc_lat']),
                        dc_lng=float(row['dc_lng']),
                    )
                    self._options.append(opt)
                    self._index[opt.option_id] = opt
                return
        # Fallback: build from source data
        self._build_from_source()
    
    def _build_from_source(self):
        """Build catalog from source data files."""
        import src.config as cfg
        from src.data_utils import load_dc_carrier_metadata, get_observed_dcs_from_preprocessed
        
        metadata_df, _ = load_dc_carrier_metadata()
        observed_dcs = get_observed_dcs_from_preprocessed()
        if observed_dcs:
            metadata_df = metadata_df[metadata_df['dc_id'].isin(observed_dcs)].copy()
        
        for _, row in metadata_df.iterrows():
            opt = FulfillmentOption(
                option_id=(int(row['dc_id']), int(row['carrier_service_id'])),
                dc_id=int(row['dc_id']),
                carrier_service_id=int(row['carrier_service_id']),
                dc_zip3=str(row.get('zip3', '')),
                dc_lat=float(row.get('lat', 0.0)),
                dc_lng=float(row.get('lon', 0.0)),
            )
            self._options.append(opt)
            self._index[opt.option_id] = opt
    
    @property
    def all_options(self) -> List[FulfillmentOption]:
        """Get all fulfillment options."""
        return self._options
    
    @property
    def index(self) -> dict[OptionId, FulfillmentOption]:
        """Get index mapping option_id -> FulfillmentOption."""
        return self._index
    
    def eligible_for_order(self, order: Order) -> List[OptionId]:
        """
        Get eligible options for an order.
        
        Args:
            order: The order
            
        Returns:
            List of eligible option IDs
        """
        if self.precompute_store:
            eligible = self.precompute_store.get_eligible_options(
                order.dest_zip5, order.dest_state
            )
            if eligible is not None:
                return eligible
        
        # Fallback: compute eligibility from rules
        return self._compute_eligibility(order)
    
    def _compute_eligibility(self, order: Order) -> List[OptionId]:
        """Compute eligibility from coverage rules."""
        import json
        from src.data_utils import _normalize_zip3, _aggregate_limited_coverage_to_zip3
        
        try:
            with open(cfg.LIMITED_COVERAGE_PATH, 'r') as f:
                raw_coverage = json.load(f)
        except FileNotFoundError:
            # No coverage restrictions, all options eligible
            return [opt.option_id for opt in self._options]
        
        aggregated = _aggregate_limited_coverage_to_zip3(raw_coverage)
        eligible = []
        dest_state_upper = order.dest_state.upper()
        
        for opt in self._options:
            zip3 = _normalize_zip3(opt.dc_zip3)
            if zip3 is None:
                continue
            
            # Check if carrier has coverage for this DC zip3
            if opt.carrier_service_id not in aggregated:
                # No restrictions, eligible
                eligible.append(opt.option_id)
                continue
            
            allowed_states = aggregated[opt.carrier_service_id].get(zip3, set())
            if not allowed_states or dest_state_upper in allowed_states:
                eligible.append(opt.option_id)
        
        return eligible

