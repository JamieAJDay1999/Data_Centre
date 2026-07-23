from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OperationalState:
    """Physical state at the boundary before the next committed interval."""

    ups_energy_kwh: float
    tes_energy_kwh: float
    it_temperature_c: float
    rack_temperature_c: float
    cold_aisle_temperature_c: float
    hot_aisle_temperature_c: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OperationalState":
        return cls(**{name: float(values[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class WorkloadCohort:
    """Outstanding flexible work with an absolute latest execution interval."""

    cohort_id: str
    arrival_utc: str
    latest_start_utc: str
    remaining_cpu_hours: float
    tranche: int

    @property
    def arrival(self) -> pd.Timestamp:
        return pd.Timestamp(self.arrival_utc)

    @property
    def latest_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.latest_start_utc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "WorkloadCohort":
        return cls(
            cohort_id=str(values["cohort_id"]),
            arrival_utc=str(values["arrival_utc"]),
            latest_start_utc=str(values["latest_start_utc"]),
            remaining_cpu_hours=float(values["remaining_cpu_hours"]),
            tranche=int(values["tranche"]),
        )


@dataclass
class HorizonResult:
    committed: pd.DataFrame
    next_state: OperationalState
    next_workload: list[WorkloadCohort]
    solver: dict[str, Any]
    audits: dict[str, float]
