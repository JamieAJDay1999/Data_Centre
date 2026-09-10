"""Verify the self-contained paper package using Python's standard library."""
from pathlib import Path
import csv
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent


def rows(name):
    with (HERE / 'data' / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = json.loads((HERE / 'evidence/asset_manifest.json').read_text())
    for item in manifest:
        path = HERE / item['package']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
    shares = rows('shiftability_profile.csv')[:96]
    assert len(shares) == 96
    for i, row in enumerate(shares):
        assert int(row['time_slot']) == i + 1
        assert abs(sum(float(row[str(k)]) for k in range(1, 5)) - 1) < 1e-12
        assert all(row[str(k)] == shares[i // 4 * 4][str(k)] for k in range(1, 5))
    endpoints = rows('annual_endpoints.csv')
    assert len(endpoints) == 14
    central = next(r for r in endpoints if r['scenario'] == '2025_optimised_cohort_trace')
    baseline = next(r for r in endpoints if r['parameter'] == 'baseline')
    saving = float(baseline['annual_cost_gbp']) - float(central['annual_cost_gbp'])
    assert abs(saving - float(central['saving_gbp'])) < 1e-7
    assert abs(saving / float(baseline['annual_cost_gbp']) * 100 - float(central['saving_percent'])) < 1e-9
    text = '\n'.join(p.read_text(encoding='utf-8') for p in
                     [HERE/'main.tex', *sorted((HERE/'sections').glob('*.tex')),
                      *sorted((HERE/'supplement').glob('*.tex'))])
    for obsolete in ['522,702.95', '27,800.64', '5.050', '5.05\\%', '0.642875', '0.13487744',
                     'thirteen', '11.25 h', '7.75 h', '2,481.41', '1,292.28']:
        assert obsolete not in text, obsolete
    assert f'{saving:,.2f}' in text
    for parameter in ['flex', 'ups', 'tes']:
        cases = sorted([r for r in endpoints if r['parameter'] == parameter] +
                       ([] if parameter == 'flex' else [dict(central, multiplier='1.0')]),
                       key=lambda r: float(r['multiplier']))
        assert all(float(a['saving_percent']) < float(b['saving_percent']) for a, b in zip(cases, cases[1:]))
    grid = rows('flexibility_results.csv')
    assert len(grid) == 288
    assert sum(r['duration_is_lower_bound'] == 'True' for r in grid) == 8
    for start, power, duration in [(12, -100, 6), (13, -150, 5), (6, -100, 1),
                                   (6, -200, 1), (15, -100, 3), (15, -200, 3),
                                   (6, 25, 10.25), (6, 75, 7.25)]:
        record = next(r for r in grid if float(r['start_hour']) == start and float(r['magnitude_kw']) == power)
        assert float(record['duration_hours']) == duration
    for folder in [HERE, HERE/'sections', HERE/'supplement', HERE/'tables']:
        for file in folder.glob('*.tex'):
            source = file.read_text(encoding='utf-8')
            for name in re.findall(r'\\input\{([^}]+)\}', source):
                assert (HERE/name).is_file(), (file, name)
            for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source):
                assert (HERE/'figures'/name).is_file(), (file, name)
    print(f'PASS: {len(manifest)} asset hashes; 14 annual cases; 288 event cells; corrected prose and all TeX dependencies.')


if __name__ == '__main__':
    main()
