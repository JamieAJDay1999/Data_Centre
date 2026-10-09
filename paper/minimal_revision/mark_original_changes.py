"""Add blue revision markup against the original submission, without deleted text."""
from pathlib import Path
import re,subprocess,shutil
here=Path(__file__).resolve().parent
root=here.parents[1]
clean=here/'main_clean.tex'
if not clean.exists():shutil.copy2(here/'main.tex',clean)
def unwrap_rev(text):
    # Old review wrappers must not turn an otherwise unchanged paragraph blue.
    start=text.index(r'\begin{document}')
    head,body=text[:start],text[start:]
    while r'\rev{' in body:
        a=body.index(r'\rev{');i=a+5;depth=1
        while depth:
            if body[i]=='{' and body[i-1]!='\\':depth+=1
            elif body[i]=='}' and body[i-1]!='\\':depth-=1
            i+=1
        body=body[:a]+body[a+5:i-1]+body[i:]
    return head+body
old=unwrap_rev((root/'paper/original_submission.tex').read_text(encoding='utf-8'))
new=unwrap_rev(clean.read_text(encoding='utf-8'))
for name in ('annual_operation.tex','ups_investment.tex'):
    new=new.replace(r'\input{'+name+'}',(here/name).read_text(encoding='utf-8'))
work=here/'build/original_markup';work.mkdir(parents=True,exist_ok=True)
(work/'old.tex').write_text(old,encoding='utf-8')
(work/'new.tex').write_text(new,encoding='utf-8')
preamble=r'''
\providecommand{\DIFadd}[1]{{\color{blue}#1}}
\providecommand{\DIFdel}[1]{}
\providecommand{\DIFaddbegin}{}
\providecommand{\DIFaddend}{}
\providecommand{\DIFdelbegin}{}
\providecommand{\DIFdelend}{}
\providecommand{\DIFaddFL}[1]{\DIFadd{#1}}
\providecommand{\DIFdelFL}[1]{}
\providecommand{\DIFaddbeginFL}{}
\providecommand{\DIFaddendFL}{}
\providecommand{\DIFdelbeginFL}{}
\providecommand{\DIFdelendFL}{}
'''
(work/'preamble.tex').write_text(preamble,encoding='utf-8')
cmd=['C:/Program Files/Git/usr/bin/perl.exe','C:/Users/jamie/AppData/Local/Programs/MiKTeX/scripts/latexdiff/latexdiff-so','--no-del','--math-markup=whole','--graphics-markup=none','--append-textcmd=HighlightItem,IEEEaftertitletext','--preamble='+str(work/'preamble.tex'),str(work/'old.tex'),str(work/'new.tex')]
run=subprocess.run(cmd,capture_output=True,encoding='utf-8',timeout=90)
(work/'latexdiff.log').write_text(run.stderr,encoding='utf-8')
if run.returncode:raise RuntimeError(run.stderr[-3000:])
marked=run.stdout

def balanced_command_span(text, command_start, before):
    """Return the last balanced command span beginning before *before*."""
    a=text.rfind(command_start,0,before)
    if a < 0: raise ValueError(command_start)
    i=a+len(command_start);depth=1
    while depth:
        if text[i]=='{' and text[i-1]!='\\':depth+=1
        elif text[i]=='}' and text[i-1]!='\\':depth-=1
        i+=1
    return a,i,text[a+len(command_start):i-1]

def unwrap_add_commands(text):
    """Remove colour wrappers while retaining their complete contents."""
    while True:
        matches=list(re.finditer(r'\\DIFadd(?:FL)?\{',text))
        if not matches:return text
        match=matches[-1]
        a=match.start();body=match.end();i=body;depth=1
        while depth:
            if text[i]=='{' and text[i-1]!='\\':depth+=1
            elif text[i]=='}' and text[i-1]!='\\':depth-=1
            i+=1
        text=text[:a]+text[body:i-1]+text[i:]

def unmark_unchanged_equations(text):
    # These labels map to current Equations 6--7, 12--14, 16--19,
    # 21--24, 32--34, 37 and 41. Their mathematics is unchanged.
    unchanged={
        'eq:energy_base','eq:energy_optimised',
        'eq:15','eq:16','eq:17','eq:19','eq:20','eq:21','eq:22',
        'eq:cooling_power_rel','eq:CRAC_cop','eq:tes_charge_cop',
        'eq:chiller_max_power','eq:tes_ch_limit','eq:tes_dis_limit',
        'eq:tes_binary_logic','eq:temp_constraints','eq:43',
    }
    for label in unchanged:
        marker=r'\label{'+label+'}'
        pos=text.index(marker)
        env=max(text.rfind(r'\begin{align',0,pos),text.rfind(r'\begin{flalign',0,pos))
        previous=text.rfind(r'\\',env,pos)
        start=max(env,previous)
        if previous>=env:
            prior=text.rfind(r'\\',env,previous)
            if r'\notag' in text[prior:previous]:start=max(env,prior)
        candidates=[x for x in (text.find(r'\\',pos),text.find(r'\end{align',pos),text.find(r'\end{flalign',pos)) if x>=0]
        end=min(candidates)
        text=text[:start]+unwrap_add_commands(text[start:end])+text[end:]
    return text

def table_block(text,label):
    pos=text.index(r'\label{'+label+'}')
    start=max(text.rfind(r'\begin{longtable',0,pos),text.rfind(r'\begin{table',0,pos))
    ends=[x for x in (text.find(r'\end{longtable}',pos),text.find(r'\end{table}',pos),text.find(r'\end{table*}',pos)) if x>=0]
    return text[start:min(ends)]

def table_rows(text,label):
    rows={}
    for line in table_block(text,label).splitlines():
        if '&' not in line or not line.rstrip().endswith(r'\\'):continue
        key=line.split('&',1)[0].strip()
        rows[key]=line
    return rows

def normal_cell(cell):
    return re.sub(r'\s+','',cell.removesuffix(r'\\'))

def wrap_cell(cell):
    leading=cell[:len(cell)-len(cell.lstrip())]
    trailing=cell[len(cell.rstrip()):]
    core=cell.strip()
    terminator=''
    if core.endswith(r'\\'):
        core=core[:-2].rstrip();terminator=r' \\'
    return leading+r'\DIFaddFL{'+core+'}'+terminator+trailing

def mark_table_differences(text,label,columns=None):
    before=table_rows(old,label);after=table_rows(new,label)
    for key,newline in after.items():
        if key not in before:continue
        oldcells=before[key].split('&');newcells=newline.split('&')
        changed=[i for i,(a,b) in enumerate(zip(oldcells,newcells)) if normal_cell(a)!=normal_cell(b)]
        if columns is not None:changed=[i for i in changed if i in columns]
        if not changed:continue
        parts=re.split(r'(&)',newline)
        for index in changed:parts[2*index]=wrap_cell(parts[2*index])
        assert text.count(newline)==1,(label,key,text.count(newline))
        text=text.replace(newline,''.join(parts),1)
    return text

def mark_caption(text,label):
    clean_pos=new.index(r'\label{'+label+'}')
    _,_,caption=balanced_command_span(new,r'\caption{',clean_pos)
    pos=text.index(r'\label{'+label+'}')
    a,b,_=balanced_command_span(text,r'\caption{',pos)
    return text[:a]+r'\caption{\DIFaddFL{'+caption.strip()+'}}'+text[b:]

def mark_complete_author_list(text):
    clean_pos=new.index(r'\author{')
    _,_,authors=balanced_command_span(new,r'\author{',clean_pos+len(r'\author{'))
    pos=text.index(r'\author{')
    a,b,_=balanced_command_span(text,r'\author{',pos+len(r'\author{'))
    return text[:a]+r'\author{\DIFadd{'+authors.strip()+'}}'+text[b:]

# Correct latexdiff's display/table blind spots without changing manuscript content.
marked=unmark_unchanged_equations(marked)
marked=mark_table_differences(marked,'tb:nomenclature')
# Keep the fully marked nomenclature on two pages; the colour wrappers otherwise
# push its final row onto an almost empty third page.
marked=marked.replace(r'\renewcommand{\arraystretch}{1.15}',
                      r'\renewcommand{\arraystretch}{1.08}',1)
marked=mark_table_differences(marked,'tab:workload-rates',{4,8,12})
marked=mark_table_differences(marked,'tb:energy_price',set(range(1,25)))
marked=marked.replace(r'\textbf{Tranche} &',r'\DIFaddFL{\textbf{Tranche}} &',1)
for tranche in range(1,5):
    source=f'$k={tranche}$ &'
    marked=marked.replace(source,r'\DIFaddFL{'+f'$k={tranche}$'+'} &',1)
marked=mark_caption(marked,'fig:workload_ratios')
marked=mark_complete_author_list(marked)
# Do not split the two mandatory arguments of the running-header command.
a=marked.index('% The paper headers');b=marked.index('% -------- Abstract',a)
ca=new.index('% The paper headers');cb=new.index('% -------- Abstract',ca)
marked=marked[:a]+new[ca:cb]+marked[b:]
# The new workflow has editable text inside TikZ, which latexdiff treats as a picture.
marked=marked.replace(r'\usetikzlibrary{arrows.meta,positioning}',r'\usetikzlibrary{arrows.meta,positioning}'+'\n'+r'\tikzset{every node/.append style={text=blue}}')
# Highlight helper must not count/write diff commands as literal text.
a=marked.index(r'\newcommand{\HighlightItem}');b=marked.index(r'\begin{document}',a)
marked=marked[:a]+r'\newcommand{\HighlightItem}[1]{\item #1}'+'\n'+preamble+'\n'+marked[b:]
# Mark newly cited or corrected bibliography entries as well as prose citations.
cited={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',old) for k in group.split(',')}
def entries(path):
    return {m.group(1):re.sub(r'\s+','',m.group(0)) for m in re.finditer(r'@\w+\{\s*([^,]+),[\s\S]*?(?=\n@|\Z)',path.read_text(encoding='utf-8'))}
before=entries(root/'paper/references.bib');after=entries(here/'references.bib')
same=[k for k in after if k in cited and before.get(k)==after[k]]
bibmarks='\n'+''.join(r'\expandafter\def\csname originalref@'+k+r'\endcsname{1}'+'\n' for k in same)
bibmarks+=r'''
\usepackage{etoolbox}
\AtBeginEnvironment{thebibliography}{%
\let\unmarkedbibitem\bibitem
\renewcommand{\bibitem}[1]{\color{black}\unmarkedbibitem{#1}%
\ifcsname originalref@#1\endcsname\color{black}\else\color{blue}\fi}}
\AtEndEnvironment{thebibliography}{\color{black}}
'''
marked=marked.replace(r'\begin{document}',bibmarks+'\n'+r'\begin{document}',1)
(here/'main.tex').write_text(marked,encoding='utf-8')
assert r'\section{Results}' in marked or r'\section{\DIFadd{Results' in marked
print('All revisions marked blue; unchanged equations restored to black; clean source preserved.')
