"""Create upload sources and numbered artwork from the current manuscript."""
from pathlib import Path
import re, shutil, subprocess, zipfile, json, hashlib
from pypdf import PdfReader, PdfWriter

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent
WORK=PAPER.parent/'.codex_tmp/submission_20261009'
STAGE=WORK/'manuscript_sources'
# Start from an empty stage so files removed from the paper do not linger.
if STAGE.exists():shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
ART=HERE/'artwork'
if ART.exists():shutil.rmtree(ART)
ART.mkdir()
# Editorial Manager builds the review PDF from the uploaded LaTeX. It flattens
# folders, may not run BibTeX and can treat every .tex file as a manuscript, so
# the package is one flat main.tex (inputs inlined), a prebuilt main.bbl and
# figures named by their final numbers. The canonical sources are unchanged.
FIGURES={'Figure_1.pdf':'Figure_01.pdf','Figure_2.pdf':'Figure_02.pdf','Figure_3.pdf':'Figure_03.pdf',
         'Figure_4a.pdf':'Figure_05a.pdf','Figure_4b.pdf':'Figure_05b.pdf','Figure_5.pdf':'Figure_06.pdf',
         'Figure_6.pdf':'Figure_07.pdf','Figure_7.pdf':'Figure_08.pdf','Figure_8.pdf':'Figure_09.pdf',
         'Figure_9.pdf':'Figure_10.pdf'}
tex=(PAPER/'main.tex').read_text(encoding='utf8')
for name in ['annual_operation.tex','ups_investment.tex']:
    tex=tex.replace('\\input{'+name+'}',(PAPER/name).read_text(encoding='utf8'))
for source,target in FIGURES.items():
    tex=tex.replace('{images/'+source+'}','{'+target+'}')
    shutil.copy2(PAPER/'images'/source,STAGE/target)
assert '\\input{' not in tex and 'images/' not in tex, 'Unflattened input remains'
(STAGE/'main.tex').write_text(tex,encoding='utf8')
shutil.copy2(PAPER/'references.bib',STAGE/'references.bib')
for name in ['IEEEtran.cls','IEEEtran.bst']:
    path=subprocess.check_output(['kpsewhich',name],text=True).strip()
    shutil.copy2(path,STAGE/name)
# Prebuild the bibliography, then verify the flat package compiles cleanly.
for command in (['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],['bibtex','main'],
                ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']):
    process=subprocess.run(command,cwd=STAGE,capture_output=True)
    if process.returncode:raise RuntimeError(process.stdout.decode(errors='replace')[-2000:])
# Editorial Manager rejects files sharing a base name (main.tex/main.bbl), so
# embed the generated bibliography in main.tex; no .bib, .bst or .bbl is then needed.
bibliography='\\bibliographystyle{IEEEtran}\n\\bibliography{references}'
tex=(STAGE/'main.tex').read_text(encoding='utf8')
assert tex.count(bibliography)==1, 'Bibliography commands not found'
tex=tex.replace(bibliography,(STAGE/'main.bbl').read_text(encoding='utf8').strip())
(STAGE/'main.tex').write_text(tex,encoding='utf8')
for name in ['main.bbl','references.bib','IEEEtran.bst']:
    (STAGE/name).unlink()
for leftover in STAGE.iterdir():
    if leftover.suffix in {'.aux','.log','.blg','.out','.pdf'} and leftover.name not in set(FIGURES.values()):
        leftover.unlink()
for _ in range(3):
    process=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=STAGE,capture_output=True)
    if process.returncode:raise RuntimeError(process.stdout.decode(errors='replace')[-2000:])
log=(STAGE/'main.log').read_text(errors='replace')
assert 'undefined' not in log and 'Rerun to get' not in log, 'Flat package has unresolved references'
for leftover in STAGE.iterdir():
    if leftover.suffix in {'.aux','.log','.blg','.out','.pdf'} and leftover.name not in set(FIGURES.values()):
        leftover.unlink()
with zipfile.ZipFile(HERE/'manuscript_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(STAGE.iterdir()):
        z.write(p,p.name)
for source,target in [('main','manuscript_clean'),('main_marked','manuscript_marked'),('supplement','supplementary_information')]:
    shutil.copy2(PAPER/'pdf'/f'{source}.pdf',HERE/f'{target}.pdf')
mapping={1:'Figure_1.pdf',2:'Figure_2.pdf',3:'Figure_3.pdf',6:'Figure_5.pdf',
         7:'Figure_6.pdf',8:'Figure_7.pdf',9:'Figure_8.pdf',10:'Figure_9.pdf'}
for number,source in mapping.items():
    shutil.copy2(PAPER/'images'/source,ART/f'Figure_{number:02d}{Path(source).suffix}')
for panel in ['a','b']:
    shutil.copy2(PAPER/'images'/f'Figure_4{panel}.pdf',ART/f'Figure_05{panel}.pdf')

# Extract the existing TikZ workflow as a separate vector artwork file.
source=(PAPER/'main.tex').read_text(encoding='utf8')
picture=source.split(r'\begin{tikzpicture}',1)[1].split(r'\end{tikzpicture}',1)[0]
# Resolve the three manuscript table references in the standalone artwork.
# These numbers match the clean manuscript's nomenclature/workload tables.
for label,number in {'tb:nomenclature':'I','tab:workload-rates':'III','tab:shiftable':'IV'}.items():
    picture=picture.replace(r'\ref{'+label+'}',number)
assert r'\ref{' not in picture, 'Resolve any new cross-references before exporting artwork.'
tex=r'''\documentclass[tikz,border=5pt]{standalone}
\usepackage{amsmath,graphicx,tikz}
\usetikzlibrary{arrows.meta,positioning}
\setlength{\textwidth}{516pt}
\begin{document}
\begin{tikzpicture}'''+picture+r'''\end{tikzpicture}
\end{document}
'''
workflow=WORK/'workflow_artwork.tex';workflow.write_text(tex,encoding='utf8')
process=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',workflow.name],
    cwd=WORK,capture_output=True)
if process.returncode:raise RuntimeError(process.stdout.decode(errors='replace')[-2000:])
workflow_pdf=PdfReader(WORK/'workflow_artwork.pdf')
assert len(workflow_pdf.pages)==1 and len(workflow_pdf.pages[0].extract_text())>500
assert '??' not in workflow_pdf.pages[0].extract_text()
shutil.copy2(WORK/'workflow_artwork.pdf',ART/'Figure_04.pdf')
(ART/'README.txt').write_text('''Artwork is numbered to match the revised manuscript.
Figure 5 consists of comparable panels 05a and 05b, assembled by main.tex.
All figures are vector PDFs.
Captions are included in the editable manuscript. The source archive retains
its existing internal image names; these standalone upload names use the actual
figure numbering after the workflow diagram was inserted.
''',encoding='utf8')
with zipfile.ZipFile(HERE/'artwork.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ART.iterdir()):z.write(p,p.name)
print('Created manuscript source ZIP, PDFs and numbered artwork.')
