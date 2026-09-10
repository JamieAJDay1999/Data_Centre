"""Audit stored corrected results without importing or executing a solver."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/final_annual_results'


def main():
    shares = pd.read_csv(ROOT / 'static/data/inputs/shiftability_profile.csv')
    values = shares[[str(k) for k in range(1, 5)]].to_numpy()
    assert values.shape == (96, 4)
    assert np.allclose(values.sum(axis=1), 1, atol=1e-12)
    assert np.array_equal(values, np.repeat(values[::4], 4, axis=0))
    original = (ROOT / 'paper/first_edit.tex').read_text(encoding='utf-8')
    intended = []
    for k in range(1, 5):
        line = next(x for x in original.splitlines() if x.startswith(r'\rev{$k=' + str(k) + '$} &'))
        intended.append([float(re.search(r'\d+', x).group()) / 100 for x in line.split('&')[2:26]])
    assert np.array_equal(values[::4], np.array(intended).T)
    endpoints = pd.read_csv(OUT / 'annual_endpoints.csv')
    assert len(endpoints) == 14 and endpoints.scenario.nunique() == 14
    quality, annual = [], []
    for row in endpoints.itertuples():
        folder = ROOT / 'static/data/rolling_year_outputs' / row.scenario
        summary = json.loads((folder / 'annual_summary.json').read_text())
        metadata = json.loads((folder / 'run_metadata.json').read_text())
        data = pd.read_csv(folder / 'annual_committed.csv')
        stamps = pd.to_datetime(data.timestamp_utc, utc=True)
        assert len(data) == 35040 and data.local_date.nunique() == 365
        assert stamps.is_unique and (stamps.diff().dropna() == pd.Timedelta(minutes=15)).all()
        cost = float((data.grid_import_kw * data.settlement_price_gbp_per_mwh * .25 / 1000).sum())
        energy = float(data.grid_import_kw.sum() * .25)
        assert abs(cost - row.annual_cost_gbp) < 1e-6
        assert abs(energy - row.grid_energy_kwh) < 1e-5
        assert abs(cost - summary['settlement_cost_gbp']) < 1e-6
        for key in ('maximum_daily_boundary_residual', 'maximum_initial_state_residual',
                    'maximum_workload_conservation_residual_cpu_h', 'maximum_tes_overlap_kw',
                    'maximum_ups_overlap_kw', 'final_workload_unserved_after_planned_lookahead_cpu_h'):
            assert abs(summary[key]) < 1e-7, (row.scenario, key)
        files = sorted((folder / 'checkpoints').glob('*.json'))
        assert len(files) == 365
        previous = None
        for file in files:
            checkpoint = json.loads(file.read_text())
            assert checkpoint['fingerprint'] == metadata['fingerprint']
            if previous is not None:
                assert checkpoint['predecessor_checkpoint_hash'] == previous
            previous = hashlib.sha256(file.read_bytes()).hexdigest()
            day_path = folder / 'days' / (file.stem + '.csv')
            assert hashlib.sha256(day_path.read_bytes()).hexdigest() == checkpoint['committed_csv_sha256']
            solver = checkpoint['solver']
            assert solver['found_feasible_incumbent']
            if solver['termination_condition'] != 'optimal':
                quality.append(dict(scenario=row.scenario, date=file.stem,
                                    gap_percent=100 * solver['relative_gap'],
                                    absolute_bound_width_gbp=solver['upper_bound_gbp'] - solver['lower_bound_gbp'],
                                    termination=solver['termination_condition'],
                                    accepted_gap=solver['meets_accepted_gap']))
        annual.append(dict(scenario=row.scenario, intervals=len(data), settlement_gbp=cost,
                           grid_energy_kwh=energy, fingerprint=metadata['fingerprint']))
    pd.DataFrame(quality).to_csv(OUT / 'corrected_solver_horizons.csv', index=False)
    components = pd.read_csv(OUT / 'annual_cost_components.csv')
    for col in ['baseline_cost_gbp', 'optimised_cost_gbp', 'cost_change_gbp']:
        assert abs(components[col].iloc[:-1].sum() - components[col].iloc[-1]) < 1e-6
    grid = pd.read_csv(ROOT / 'reports/representative_day_flexibility/flexibility_results.csv')
    assert len(grid) == 288 and not grid.duplicated(['start_step', 'magnitude_kw']).any()
    assert grid.validated.all() and grid.duration_is_lower_bound.sum() == 8
    assert grid.boundary_verified.sum() == 280
    positive = grid[grid.duration_steps > 0]
    assert positive.solver_meets_accepted_gap.all()
    assert positive.solver_relative_gap.max() <= .01 + 1e-10
    reference = pd.read_csv(ROOT / 'static/data/representative_day_flexibility/2025-09-22_baseline_planned.csv')
    component_rows = []
    for row in positive.itertuples():
        name = f'2025-09-22_start_{row.start_step:02d}_magnitude_{"pos" if row.magnitude_kw > 0 else "neg"}_{abs(row.magnitude_kw):.0f}.csv'
        event = pd.read_csv(ROOT / 'static/data/representative_day_flexibility' / name).iloc[:row.duration_steps]
        base = reference.iloc[row.start_step:row.start_step + row.duration_steps]
        assert np.array_equal(pd.to_datetime(event.timestamp_utc, utc=True).to_numpy(),
                              pd.to_datetime(base.timestamp_utc, utc=True).to_numpy())
        delta = {k: event[k].to_numpy() - base[k].to_numpy() for k in
                 ['p_it_total_kw', 'p_chiller_hvac_kw', 'p_chiller_tes_kw', 'p_ups_charge_kw', 'p_ups_discharge_kw', 'grid_import_kw']}
        net_ups = delta['p_ups_charge_kw'] - delta['p_ups_discharge_kw']
        total = delta['p_it_total_kw'] + delta['p_chiller_hvac_kw'] + delta['p_chiller_tes_kw'] + net_ups
        assert np.max(np.abs(total - delta['grid_import_kw'])) < 1e-7
        assert np.max(np.abs(total - row.magnitude_kw)) <= .100001
        component_rows.append(dict(start_hour=row.start_hour, magnitude_kw=row.magnitude_kw,
                                   duration_hours=row.duration_hours, lower_bound=row.duration_is_lower_bound,
                                   it_mean_kw=delta['p_it_total_kw'].mean(), crac_mean_kw=delta['p_chiller_hvac_kw'].mean(),
                                   tes_chiller_mean_kw=delta['p_chiller_tes_kw'].mean(), ups_net_mean_kw=net_ups.mean()))
    pd.DataFrame(component_rows).to_csv(OUT / 'corrected_event_components.csv', index=False)
    end = json.loads((ROOT / 'static/data/rolling_year_outputs/2025_optimised_cohort_trace/checkpoints/2025-12-31.json').read_text())
    report = dict(status='passed', hourly_tranche_mapping_matches_first_edit=True,
                  annual_scenarios=annual, checked_checkpoint_count=14 * 365,
                  event_cells=288, positive_event_dispatches=len(positive),
                  boundary_or_cap_verified_cells=280, lower_bound_cells=8,
                  annual_gap_exceptions=[r for r in quality if r['gap_percent'] > 1],
                  central_nonoptimal_horizons=[r for r in quality if r['scenario'] == '2025_optimised_cohort_trace'],
                  corrected_year_end_audit=end['audits'])
    (OUT / 'corrected_evidence_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'annual_scenarios'}, indent=2))


if __name__ == '__main__':
    main()
