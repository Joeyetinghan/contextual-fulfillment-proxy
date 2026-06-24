"""Discrete-event simulation package for order fulfillment optimization."""

from src.simulator.entities import (
    Order,
    OrderItem,
    FulfillmentOption,
    ItemAllocation,
    OrderDecision,
    OptionId,
)
from src.simulator.catalog import OptionsCatalog
from src.simulator.precompute import PrecomputeStore
from src.simulator.state import SimulationState
from src.simulator.engine import SimulationEngine
from src.simulator.delivery_sampler import OutcomeSampler

__all__ = [
    'Order',
    'OrderItem',
    'FulfillmentOption',
    'ItemAllocation',
    'OrderDecision',
    'OptionId',
    'OptionsCatalog',
    'PrecomputeStore',
    'SimulationState',
    'SimulationEngine',
    'OutcomeSampler',
]

