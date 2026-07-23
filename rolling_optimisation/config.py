from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from inputs.parameters_optimisation import ModelParameters

from .types import OperationalState


@dataclass(frozen=True)
class RollingConfig:
    """Resolved configuration for one linked annual scenario."""

    scenario_id: str = "central_optimised"
    mode: str = "optimised"
    dt_seconds: int = 900
    lookahead_steps: int = 12
    solver_name: str = "auto"
    solver_time_limit_s: int = 300
    mip_gap: float = 0.001
    baseline_cold_aisle_setpoint_c: float = 22.5
    state_tolerance: float = 1e-5
    flow_tolerance_kw: float = 1e-3
    workload_tolerance_cpu_h: float = 1e-7

    def __post_init__(self) -> None:
        if self.mode not in {"optimised", "baseline"}:
            raise ValueError("mode must be 'optimised' or 'baseline'")
        if self.dt_seconds != 900:
            raise ValueError("The first rolling implementation supports 15-minute steps only")
        if self.lookahead_steps < 12:
            raise ValueError("lookahead_steps must cover the maximum 3-hour workload delay")

    @property
    def dt_hours(self) -> float:
        return self.dt_seconds / 3600.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self, input_hash: str, code_hash: str) -> str:
        payload = {
            "config": self.to_dict(),
            "input_hash": input_hash,
            "code_hash": code_hash,
            "schema": 1,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def default_initial_state(config: RollingConfig) -> OperationalState:
    params = ModelParameters(dt_seconds=config.dt_seconds)
    return OperationalState(
        ups_energy_kwh=params.e_start_kwh,
        tes_energy_kwh=params.TES_initial_charge_kWh,
        it_temperature_c=params.T_IT_initial_Celsius,
        rack_temperature_c=params.T_Rack_initial_Celsius,
        cold_aisle_temperature_c=params.T_cAisle_initial,
        hot_aisle_temperature_c=params.T_hAisle_initial,
    )
