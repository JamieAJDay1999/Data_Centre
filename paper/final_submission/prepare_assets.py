"""Snapshot existing evidence and make publication tables; never imports/runs a solver."""
from pathlib import Path
import csv, json, shutil, hashlib, re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in ('figures', 'tables', 'data', 'evidence', 'source_snapshot'):
    (HERE/folder).mkdir(exist_ok=True)
manifest = []
def copy(source, target):
    source, target = ROOT/source, HERE/target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest.append({'source': str(source.relative_to(ROOT)).replace('\\','/'),
                     'package': str(target.relative_to(HERE)).replace('\\','/'),
                     'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
for name in ('Figure_1.png','Figure_4a.png','Figure_4b.png','Figure_5.png','Figure_6.png','Figure_7.png','Figure_8.png','Figure_9.png'):
    copy('paper/images/'+name, 'figures/'+name)
for name in ('load_profiles.csv','shiftability_profile.csv'):
    copy('static/data/inputs/'+name, 'data/'+name)
for name in ('annual_endpoints.csv','annual_cost_components.csv','representative_day_selection.csv','intermediate_quality_reruns.csv'):
    copy('reports/final_annual_results/'+name, 'data/'+name)
copy('reports/representative_day_flexibility/flexibility_results.csv','data/flexibility_results.csv')
copy('reports/representative_day_flexibility/summary.json','evidence/event_summary.json')
copy('reports/terminal_treatment/terminal_treatment_report.md','evidence/earlier_terminal_assessment.md')
for name in ('model.py','types.py','timeline.py','config.py','run_representative_day_flexibility.py'):
    copy('rolling_optimisation/'+name,'source_snapshot/rolling_optimisation/'+name)
copy('inputs/parameters_optimisation.py','source_snapshot/inputs/parameters_optimisation.py')
copy('paper/generate_annual_results.py','source_snapshot/generate_annual_results.py')
def rows(name):
    with (HERE/'data'/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
annual = rows('annual_endpoints.csv')
for row in annual:
    case = row['scenario']
    for name in ('annual_summary.json','run_metadata.json'):
        copy(f'static/data/rolling_year_outputs/{case}/{name}',f'evidence/{case}/{name}')
    summary = json.loads((HERE/f'evidence/{case}/annual_summary.json').read_text())
    assert summary['local_dates']==365 and summary['committed_intervals']==35040
    assert abs(float(row['annual_cost_gbp'])-summary['settlement_cost_gbp'])<1e-6
# Retain the exact quarter-hour price series, and a compact event-day dispatch.
central = ROOT/'static/data/rolling_year_outputs/2025_optimised_cohort_trace'
with (central/'annual_committed.csv').open(newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f)
    columns=['timestamp_utc','local_date','settlement_price_gbp_per_mwh']
    with (HERE/'data/price_timeline.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=columns); writer.writeheader()
        for row in reader: writer.writerow({k:row[k] for k in columns})
for case in ('2025_baseline_reformulated','2025_optimised_cohort_trace'):
    copy(f'static/data/rolling_year_outputs/{case}/days/2025-09-22.csv',f'data/{case}_representative_day.csv')

def write(name,text): (HERE/'tables'/name).write_text(text,encoding='utf-8')
def table(caption,label,spec,header,body,wide=False):
    env='table*' if wide else 'table'
    return '\\begin{'+env+'}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{@{}'+spec+'@{}}\\toprule\n'+header+' \\\\\\midrule\n'+'\n'.join(body)+'\n\\bottomrule\\end{tabular}\n\\end{'+env+'}\n'
comp=rows('annual_cost_components.csv')
body=[f"{r['component']} & {float(r['baseline_cost_gbp']):,.0f} & {float(r['optimised_cost_gbp']):,.0f} & {float(r['cost_change_gbp']):+,.0f} \\\\" for r in comp]
write('annual_components.tex',table('Annual signed electricity-cost composition (GBP). Changes are coordinated minus reference; displayed values are rounded.','tab:cost-components','lrrr','Component & Reference & Coordinated & Change',body,True))
def case_label(label): return label[:-1]+r'$\times$' if label.endswith('x') else label
body=[f"{case_label(r['label'])} & {float(r['annual_cost_gbp']):,.0f} & {float(r['saving_percent']):.3f} & {float(r['change_from_central_pp']):+.3f} & {float(r['grid_energy_kwh'])/1e6:.3f} \\\\" for r in annual]
write('annual_endpoints.tex',table('Full-year cost and resource sensitivity. Change is percentage points from the central coordinated saving; baseline denotes reference operation.','tab:annual-endpoints','lrrrr','Case & Cost (GBP) & Saving (\\%) & Change (pp) & Energy (GWh)',body))
body=[f"{case_label(r['label'])} & {r['non_optimal_horizons']} & {float(r['maximum_recorded_gap_percent']):.3f} \\\\" for r in annual]
write('solver_quality.tex',table('Original annual-chain solver records. Non-optimal denotes the stored termination classification; diagnostic reruns are reported separately.','tab:solver-quality','lrr','Case & Non-optimal horizons & Maximum recorded gap (\\%)',body))
base=float(next(r for r in annual if r['label']=='Flexible workload 1.0x')['annual_cost_gbp'])
increments=[(750,150,base-float(next(r for r in annual if r['label']=='UPS capacity 1.25x')['annual_cost_gbp'])),(900,300,base-float(next(r for r in annual if r['label']=='UPS capacity 1.5x')['annual_cost_gbp']))]
def crf(r,n):return r*(1+r)**n/((1+r)**n-1) if r else 1/n
body=[f'{e} & {delta} & {saving:,.2f} & {saving/crf(.08,10):,.0f} \\\\' for e,delta,saving in increments]
write('ups_screen.tex',table('UPS enlargement screen: constant annual benefit, ten years, 8\\% real discount rate and no additional non-electricity costs.','tab:ups-screen','rrrr','Capacity & Added energy & Added saving & Break-even\\\\(kWh) & (kWh) & (GBP/year) & investment (GBP)',body))
body=[]
for n in (5,10,15):
    for rate in (.04,.08,.12):
        body.append(f'{n} & {100*rate:.0f} & {crf(rate,n):.5f} & {increments[0][2]/crf(rate,n)/150:.2f} & {increments[1][2]/crf(rate,n)/300:.2f} \\\\')
write('financial_range.tex',table('Illustrative supported investment per added nameplate kWh, in GBP/kWh, before additional non-electricity costs.','tab:financial-range','rrrrr','Life (years) & Real rate (\\%) & CRF & 750 kWh case & 900 kWh case',body))
loads=[r for r in rows('load_profiles.csv') if 1<=int(r['time_slot'])<=96]
shares=[r for r in rows('shiftability_profile.csv') if 1<=int(r['time_slot'])<=96]
assert len(loads)==len(shares)==96
lines=[]
for a,b in zip(loads,shares):
    assert a['time_slot']==b['time_slot']
    assert abs(sum(float(b[str(k)]) for k in range(1,5))-1)<1e-8
    i=int(a['time_slot'])-1
    lines.append(f'{i//4:02d}:{15*(i%4):02d} & '+ ' & '.join(f'{float(x):.2f}' for x in [a['inflexible_load'],a['flexible_load']]+[b[str(k)] for k in range(1,5)])+r' \\')
header=r'Local start & Inflexible & Flexible & $\alpha_1$ & $\alpha_2$ & $\alpha_3$ & $\alpha_4$ \\'
write('workload_inputs.tex',r'\begin{longtable}{@{}lrrrrrr@{}}'+'\n'+r'\caption{Actual quarter-hour workload inputs. Utilisations are fractions of rated CPU capacity; tranche shares are fractions of flexible work.}\label{tab:workload-inputs}\\\toprule'+'\n'+header+r'\midrule\endfirsthead'+'\n'+r'\toprule'+'\n'+header+r'\midrule\endhead'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{longtable}'+'\n')
# Exact constants, calculated from the defining arithmetic in the snapshotted parameter file.
cit=100*10*20*600+100*140*420
free=2*.605*1.2-10*.0868*.434*.679
crack=free*100*1.16*1005.45+cit
area=2*3*28+2*3*10+28*10
gcold=1/(1/(16*area)+round((.7/1000)/(area/38),4))
thermal=[('$C_{\\mathrm{IT}}$',cit/1000,'kJ/K'),('$C_{\\mathrm R}$',crack/1000,'kJ/K'),('$C_{\\mathrm{CA}}$',2000*1.16*1005.45/1000,'kJ/K'),('$C_{\\mathrm{HA}}$',1000*1.16*1005.45/1000,'kJ/K'),('$G_{\\mathrm{conv}}$',round(100*10*20*12000/2202,3)/1000,'kW/K'),('$G_{\\mathrm{cold}}$',gcold/1000,'kW/K'),('$c_p$',1.00545,'kJ/(kg K)'),('$\\dot m$',100,'kg/s'),('$\\kappa$',.7663,'--'),('$T^{\\mathrm{out}}$',22,'$^{\\circ}$C')]
write('thermal_parameters.tex',table('Thermal constants from the implemented parameter definitions (rounded here to six decimal places).','tab:thermal','lrl','Parameter & Value & Unit',[f'{symbol} & {value:.6f} & {unit} \\\\' for symbol,value,unit in thermal]))
copy('paper/references.bib','references.bib')
bib=(HERE/'references.bib').read_text(encoding='utf-8')
bib=bib.replace('Yangyang Fu and Xiao Han','Yangyang Fu and Xu Han')
bib=bib.replace('Liping Liu and Xiaotian Shen and Ziliang Chen and Qiang Sun and Roland Wennersten','Luyao Liu and Xinwei Shen and Zhigang Chen and Qie Sun and Ronald Wennersten')
bib=bib.replace('10.1016/j.future.2017.08.001','10.1016/j.future.2016.05.010')
bib=bib.replace('10.1109/EEM.2019.8916334','10.1109/EEM.2019.8916344')
bib=bib.replace('10.1109/TPWRS.2022.3214118','10.1109/TPWRS.2022.3173250')
bib=bib.replace('smart girds','smart grids')
# Publisher-specific further corrections, retained separately from source snapshots.
corrections_path=HERE/'evidence/reference_corrections.json'
if corrections_path.exists():
    for key,fields in json.loads(corrections_path.read_text()).items():
        pattern=r'(@\w+\{\s*'+re.escape(key)+r',[\s\S]*?)(?=\n@|\Z)'
        match=re.search(pattern,bib)
        if not match:raise ValueError(key)
        entry=match.group(1)
        for field,value in fields.items():
            field_pattern=r'(?m)^\s*'+re.escape(field)+r'\s*=.*$'
            line='  '+field+' = {'+value+'},'
            if re.search(field_pattern,entry):entry=re.sub(field_pattern,lambda _:line,entry)
            else:entry=entry.replace('\n}', '\n'+line+'\n}',1)
        bib=bib[:match.start()]+entry+bib[match.end():]
bib+='\n@misc{sam_crf,\n author={{National Renewable Energy Laboratory}},\n title={{System Advisor Model: LCOE Calculator, capital recovery factor}},\n year={n.d.},\n howpublished={SAM Help},\n url={https://samrepo.nlr.gov/help/fin_lcoefcr.html},\n note={Accessed 8 September 2026}\n}\n'
(HERE/'references.bib').write_text(bib,encoding='utf-8')
(HERE/'evidence/asset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f'Prepared {len(manifest)} snapshots; verified all 14 annual costs; generated tables from stored inputs/results. No solver executed.')
