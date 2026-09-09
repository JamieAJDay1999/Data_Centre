"""Check cited DOI metadata against publisher-deposited Crossref records. Read-only."""
from pathlib import Path
import re,json,urllib.request,urllib.parse,concurrent.futures,time,sys
sys.stdout.reconfigure(encoding='utf-8')
here=Path(__file__).resolve().parent
tex='\n'.join(p.read_text(encoding='utf-8') for p in here.rglob('*.tex') if 'build' not in p.parts)
cited={key.strip() for group in re.findall(r'\\cite(?:[a-z]*)?(?:\[[^\]]*\])*\{([^}]+)\}',tex) for key in group.split(',')}
bib=(here/'references.bib').read_text(encoding='utf-8')
entries={m.group(1):m.group(0) for m in re.finditer(r'@\w+\{\s*([^,]+),[\s\S]*?(?=\n@|\Z)',bib)}
def check(key):
    entry=entries.get(key,'')
    match=re.search(r'\bdoi\s*=\s*\{([^}]+)',entry,re.I)
    if not match:return {'key':key,'status':'no DOI; existing source citation'}
    doi=match.group(1)
    try:
        time.sleep(1)
        request=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''),headers={'User-Agent':'ManuscriptBibliographyCheck/1.0'})
        with urllib.request.urlopen(request,timeout=12) as response: item=json.load(response)['message']
        return {'key':key,'doi':doi,'status':'verified record','title':item.get('title'), 'authors':item.get('author'), 'published':item.get('published'), 'container':item.get('container-title'),'volume':item.get('volume'),'issue':item.get('issue'),'pages':item.get('page'),'article_number':item.get('article-number')}
    except Exception as exc:return {'key':key,'doi':doi,'status':str(exc)}
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool: findings=list(pool.map(check,sorted(cited)))
(here/'evidence/bibliography_audit.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
for item in findings:
    authors='; '.join(a.get('given','')+' '+a.get('family','') for a in item.get('authors',[]))
    print(item['key'],item['status'],item.get('title',''),authors,item.get('published',''))
