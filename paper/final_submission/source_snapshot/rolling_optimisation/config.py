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
    maximum_accepted_gap: float = 0.01
    fail_on_gap_exceeded: bool = False
    it_power_segments: int = 4
    it_power_representation: str = "DLOG"
    it_power_breakpoint_exponent: float = 1.5
    price_treatment: str = "signed"
    grid_import_limit_kw: float | None = None
    ups_reserve_kwh: float | None = None
    ups_throughput_cost_gbp_per_kwh: float = 0.0
    tes_throughput_cost_gbp_per_kwh_th: float = 0.0
    terminal_ups_value_gbp_per_kwh: float = 0.0
    terminal_tes_value_gbp_per_kwh_th: float = 0.0
    ups_capacity_multiplier: float = 1.0
    tes_capacity_multiplier: float = 1.0
    flexible_workload_multiplier: float = 1.0
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
        if self.solver_time_limit_s <= 0:
            raise ValueError("solver_time_limit_s must be positive")
        if not 0 <= self.mip_gap < 1:
            raise ValueError("mip_gap must be in [0, 1)")
        if not 0 <= self.maximum_accepted_gap < 1:
            raise ValueError("maximum_accepted_gap must be in [0, 1)")
        if self.maximum_accepted_gap < self.mip_gap:
            raise ValueError("maximum_accepted_gap cannot be smaller than mip_gap")
        if self.it_power_segments < 2:
            raise ValueError("it_power_segments must be at least two")
        if self.it_power_breakpoint_exponent <= 0:
            raise ValueError("it_power_breakpoint_exponent must be positive")
        logarithmic = self.it_power_representation in {"DLOG", "LOG"}
        if logarithmic and (
            self.it_power_segments & (self.it_power_segments - 1)
        ):
            raise ValueError(
                "it_power_segments must be a power of two for a logarithmic representation"
            )
        if self.it_power_representation not in {
            "CUSTOM",
            "DLOG",
            "LOG",
            "INC",
            "CC",
        }:
            raise ValueError(
                "it_power_representation must be CUSTOM, DLOG, LOG, INC, or CC"
            )
        if self.price_treatment not in {"signed", "floor_zero", "shift_year_min"}:
            raise ValueError(
                "price_treatment must be signed, floor_zero, or shift_year_min"
            )
        optional_positive = {
            "grid_import_limit_kw": self.grid_import_limit_kw,
            "ups_reserve_kwh": self.ups_reserve_kwh,
        }
        for name, value in optional_positive.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        nonnegative = {
            "ups_throughput_cost_gbp_per_kwh": self.ups_throughput_cost_gbp_per_kwh,
            "tes_throughput_cost_gbp_per_kwh_th": self.tes_throughput_cost_gbp_per_kwh_th,
            "terminal_ups_value_gbp_per_kwh": self.terminal_ups_value_gbp_per_kwh,
            "terminal_tes_value_gbp_per_kwh_th": self.terminal_tes_value_gbp_per_kwh_th,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        positive_multipliers = {
            "ups_capacity_multiplier": self.ups_capacity_multiplier,
            "tes_capacity_multiplier": self.tes_capacity_multiplier,
            "flexible_workload_multiplier": self.flexible_workload_multiplier,
        }
        for name, value in positive_multipliers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

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
            "schema": 2,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def model_parameters(config: RollingConfig) -> ModelParameters:
    """Resolve central physical parameters and the sensitivity multipliers."""

    params = ModelParameters(dt_seconds=config.dt_seconds)
    ups_initial_fraction = params.e_start_kwh / params.e_nom_kwh
    params.e_nom_kwh *= config.ups_capacity_multiplier
    params.e_min_kwh = params.soc_min * params.e_nom_kwh
    params.e_max_kwh = params.soc_max * params.e_nom_kwh
    params.e_start_kwh = ups_initial_fraction * params.e_nom_kwh

    tes_initial_fraction = params.TES_initial_charge_kWh / params.TES_capacity_kWh
    params.TES_kwh_cap *= config.tes_capacity_multiplier
    params.TES_capacity_kWh = params.TES_kwh_cap
    params.TES_initial_charge_kWh = (
        tes_initial_fraction * params.TES_capacity_kWh
    )
    return params


def default_initial_state(config: RollingConfig) -> OperationalState:
    params = model_parameters(config)
    return OperationalState(
        ups_energy_kwh=params.e_start_kwh,
        tes_energy_kwh=params.TES_initial_charge_kWh,
        it_temperature_c=params.T_IT_initial_Celsius,
        rack_temperature_c=params.T_Rack_initial_Celsius,
        cold_aisle_temperature_c=params.T_cAisle_initial,
        hot_aisle_temperature_c=params.T_hAisle_initial,
    )
