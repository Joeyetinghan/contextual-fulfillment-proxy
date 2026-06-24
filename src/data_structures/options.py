from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass
class OptionMatrix:
    """Lightweight representation of fulfillment options backed by NumPy arrays."""

    option_ids: np.ndarray
    dc_ids: np.ndarray
    carrier_service_ids: np.ndarray
    base_costs: np.ndarray
    _option_index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.option_ids = np.asarray(self.option_ids, dtype=int)
        self.dc_ids = np.asarray(self.dc_ids, dtype=int)
        self.carrier_service_ids = np.asarray(self.carrier_service_ids, dtype=int)
        self.base_costs = np.asarray(self.base_costs, dtype=float)
        if not (len(self.option_ids) == len(self.dc_ids) == len(self.carrier_service_ids) == len(self.base_costs)):
            raise ValueError("OptionMatrix arrays must be equal length.")
        self._option_index = {int(opt): idx for idx, opt in enumerate(self.option_ids.tolist())}

    @property
    def size(self) -> int:
        return int(self.option_ids.size)

    def option_list(self) -> list[int]:
        return self.option_ids.astype(int).tolist()

    def option_index(self) -> dict[int, int]:
        return self._option_index

    def option_to_dc(self) -> dict[int, int]:
        return {int(opt): int(dc) for opt, dc in zip(self.option_ids, self.dc_ids, strict=False)}

    def option_to_carrier(self) -> dict[int, int]:
        return {int(opt): int(cs) for opt, cs in zip(self.option_ids, self.carrier_service_ids, strict=False)}

    def carrier_groups(self) -> dict[int, np.ndarray]:
        groups: dict[int, np.ndarray] = {}
        for carrier_id in np.unique(self.carrier_service_ids):
            mask = self.carrier_service_ids == carrier_id
            groups[int(carrier_id)] = self.option_ids[mask].astype(int)
        return groups

    def dc_groups(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for dc_id in np.unique(self.dc_ids):
            mask = self.dc_ids == dc_id
            groups[int(dc_id)] = self.option_ids[mask].astype(int).tolist()
        return groups

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'option_id': self.option_ids,
            'dc_id': self.dc_ids,
            'carrier_service_id': self.carrier_service_ids,
            'base_cost': self.base_costs,
        })

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "OptionMatrix":
        if df is None or df.empty:
            raise ValueError("Cannot build OptionMatrix from empty DataFrame.")
        return cls(
            option_ids=df['option_id'].to_numpy(dtype=int, copy=True),
            dc_ids=df['dc_id'].to_numpy(dtype=int, copy=True),
            carrier_service_ids=df['carrier_service_id'].to_numpy(dtype=int, copy=True),
            base_costs=df['base_cost'].to_numpy(dtype=float, copy=True),
        )

    @classmethod
    def from_mappings(
        cls,
        option_ids: Iterable[int],
        dc_ids: Iterable[int],
        carrier_service_ids: Iterable[int],
        base_costs: Iterable[float],
    ) -> "OptionMatrix":
        return cls(
            option_ids=np.fromiter(option_ids, dtype=int),
            dc_ids=np.fromiter(dc_ids, dtype=int),
            carrier_service_ids=np.fromiter(carrier_service_ids, dtype=int),
            base_costs=np.fromiter(base_costs, dtype=float),
        )


def ensure_option_matrix(options: OptionMatrix | pd.DataFrame | Mapping[str, Iterable]) -> OptionMatrix:
    """Convert supported input types into an OptionMatrix."""
    if isinstance(options, OptionMatrix):
        return options
    if isinstance(options, pd.DataFrame):
        return OptionMatrix.from_dataframe(options)
    raise TypeError(f"Unsupported options type: {type(options)}")
