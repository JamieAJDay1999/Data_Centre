"""Read-only evidence checks, PDF contact sheets and a portable source archive."""
from pathlib import Path
import csv,json,re,hashlib,zipfile
from PIL import Image,ImageDraw
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def rows(name):return list(csv.DictReader((HERE/'data'/name).open(encoding='utf-8')))
annual=rows('annual_endpoints.csv'); events=rows('flexibility_results.csv')
base=next(r for r in annual if r['label']=='Baseline')
central=next(r for r in annual if r['label']=='Flexible workload 1.0x')
assert len(annual)==14 and len(events)==288
for row in annual:
    scenario=row['scenario']
    raw=json.loads((ROOT/f'static/data/rolling_year_outputs/{scenario}/annual_summary.json').read_text())
    assert raw['local_dates']==365 and raw['committed_intervals']==35040
    assert abs(raw['settlement_cost_gbp']-float(row['annual_cost_gbp']))<1e-6
    expected=100*(float(base['annual_cost_gbp'])-float(row['annual_cost_gbp']))/float(base['annual_cost_gbp'])
    assert abs(expected-float(row['saving_percent']))<1e-9
def duration(hour,power):
    return float(next(r for r in events if float(r['start_hour'])==hour and float(r['magnitude_kw'])==power)['duration_hours'])
for h,p,d in [(6,-100,1.25),(6,-200,1.25),(12,-100,6),(12,-150,6),(15,-100,3),(15,-200,3),(6,25,11.25),(6,50,8.75),(6,75,7.75),(3,25,12),(10,25,12),(11,25,12),(12,25,12)]:assert duration(h,p)==d,(h,p)
assert sum(r['duration_is_lower_bound']=='True' for r in events)==13
assert abs(float(base['annual_cost_gbp'])-float(central['annual_cost_gbp'])-27800.6437982245)<1e-6
crf=.08*1.08**10/(1.08**10-1)
for label,expected in [('UPS capacity 1.25x',8671.30),('UPS capacity 1.5x',16650.46)]:
    scenario=next(r for r in annual if r['label']==label)
    assert abs((float(central['annual_cost_gbp'])-float(scenario['annual_cost_gbp']))/crf-expected)<.01
verification={'annual_scenarios_verified':14,'annual_days_per_case':365,'annual_intervals_per_case':35040,'event_cells':288,'quoted_event_durations_checked':13,'unresolved_next_step_cells':13,'new_optimisation_runs':0,'pdfs':{}}
for name in ('main','supplement'):
    log=(HERE/f'build/{name}.log').read_text(errors='replace')
    assert 'undefined references' not in log and 'undefined on input line' not in log
    assert 'multiply defined' not in log
    reader=PdfReader(HERE/f'build/{name}.pdf')
    text='\n'.join(p.extract_text() or '' for p in reader.pages)
    assert len(text)>1000
    verification['pdfs'][name]={'pages':len(reader.pages),'unresolved_references':False,'overfull_box_warnings':len(re.findall('Overfull',log))}
    (HERE/f'build/{name}_extracted.txt').write_text(text,encoding='utf-8')
    pngs=sorted((HERE/'build/qa').glob(name+'-*.png'))
    for start in range(0,len(pngs),12):
        selected=pngs[start:start+12]
        sheet=Image.new('RGB',(4*390,3*540),'#d8d8d8');draw=ImageDraw.Draw(sheet)
        for j,png in enumerate(selected):
            im=Image.open(png).convert('RGB');im.thumbnail((380,510))
            x=(j%4)*390+5;y=(j//4)*540+25
            sheet.paste(im,(x,y));draw.text((x,y-20),png.stem,fill='black')
        sheet.save(HERE/f'build/qa/{name}_contact_{start//12+1}.png')
(HERE/'review/verification.json').write_text(json.dumps(verification,indent=2))
out=ROOT/'output/overleaf';out.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(out/'data_centre_minimal_revision.zip','w',zipfile.ZIP_DEFLATED) as z:
    for path in HERE.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(HERE)
        if relative.parts[0] in ('build',) or '__pycache__' in relative.parts:continue
        if path.suffix=='.py':continue  # local reconstruction helpers depend on historical repository inputs
        z.write(path,str(relative))
print(json.dumps(verification,indent=2))
