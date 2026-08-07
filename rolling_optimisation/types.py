from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    planned: pd.DataFrame
    next_state: OperationalState
    terminal_state: OperationalState
    next_workload: list[WorkloadCohort]
    solver: dict[str, Any]
    audits: dict[str, float]
    workload_trace: pd.DataFrame = field(default_factory=pd.DataFrame)
    terminal_workload: list[WorkloadCohort] = field(default_factory=list)


@dataclass(frozen=True)
class FlexibilityRequest:
    """Grid-power request imposed on part of a solved rolling horizon."""

    baseline_grid_import_kw: tuple[float, ...]
    start_step: int
    duration_steps: int
    delta_kw: float
    baseline_total_cpu: tuple[float, ...] | None = None
    event_initial_state: OperationalState | None = None
    tolerance_kw: float = 0.1
    recovery_state: OperationalState | None = None
    recovery_temperature_tolerance_c: float = 0.05
    recovery_workload: tuple[WorkloadCohort, ...] | None = None
    recovery_workload_tolerance_cpu_h: float = 1e-7
