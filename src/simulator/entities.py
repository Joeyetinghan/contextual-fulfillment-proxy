"""Domain entities for order fulfillment simulation."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


OptionId = Tuple[int, int]  # (dc_id, carrier_service_id)


@dataclass(frozen=True)
class FulfillmentOption:
    """A fulfillment option representing a (DC, carrier-service) pair."""
    option_id: OptionId
    dc_id: int
    carrier_service_id: int
    dc_zip3: str
    dc_lat: float
    dc_lng: float


@dataclass
class OrderItem:
    """An item in an order."""
    sku_id: str
    quantity: int


@dataclass
class ItemAllocation:
    """Allocation of a SKU quantity to a fulfillment option."""
    sku_id: str
    option_id: OptionId
    quantity: int  # shipped via this option


@dataclass
class OrderDecision:
    """Fulfillment decision for an order."""
    allocations: List[ItemAllocation]  # splits allowed
    unfilled: Optional[Dict[str, int]] = None  # sku_id -> lost qty


@dataclass
class Order:
    """An order with static attributes and items."""
    order_id: str
    dest_zip5: str
    dest_state: str
    dest_lat: float
    dest_lng: float
    static_features: Dict[str, float]
    items: List[OrderItem]
    promise_delivery_days: int
    order_time: Optional[object] = None  # datetime or timestamp
    eligible_options: Optional[List[OptionId]] = None
    customer_dc: Optional[int] = None  # closest DC identifier
