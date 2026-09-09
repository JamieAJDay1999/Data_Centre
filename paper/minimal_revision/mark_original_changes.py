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
print('Blue additions-only markup generated against paper/original_submission.tex; clean source preserved.')
