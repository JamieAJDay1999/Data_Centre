"""Prepare supplementary additions, review files and an input plot; no model run."""
from pathlib import Path
import re,shutil,html,difflib,json,csv,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'paper/final_submission'

# Supplement contains additions; none of the original component material is relocated.
inputs=(OLD/'supplement/inputs.tex').read_text(encoding='utf-8')
inputs=inputs[inputs.index(r'\section{Workload and electricity-price inputs}'):]
inputs=inputs.replace('An older literature-range table is not reproduced because its reported ranges do not consistently enclose the adopted values and it does not specify the actual solver inputs.','The main paper retains its earlier literature-range table as context, with the adopted columns updated to hourly means. Those range summaries do not replace the exact quarter-hour inputs.')
procedures=(OLD/'supplement/procedures.tex').read_text(encoding='utf-8')
procedures=procedures.replace('in S5','in S3')
procedures=procedures.replace('If $\\boldsymbol x$ contains',r'''In this supplement, $\mathcal C_d=T_d$ and $\mathcal H_d=T_d^{\mathrm{ext}}$ in the main-paper notation. A cohort $c$ identifies one arrival/tranche pair $(t,k)$; $r_c$ and $\ell_c$ are its arrival and final eligible interval start, $A_{c,d}$ its available work, and $R_{c,H}$ its residual at boundary $H$. Superscript $\mathrm{op}$ denotes Scenario 2, and $\mathrm{ref}$ denotes Scenario 1. Powers and stored energies use the main-paper units.

If $\boldsymbol x$ contains''')
evidence=(OLD/'supplement/evidence.tex').read_text(encoding='utf-8')
a=evidence.index('Figure~\\ref{fig:dispatch-extra}');b=evidence.index('The annual component-cost differences',a)
evidence=evidence[:a]+r'''The annual endpoints and event grid are supplied as \path{data/annual_endpoints.csv} and \path{data/flexibility_results.csv}. The representative-day component schedule remains in the main paper.

'''+evidence[b:]
evidence=evidence.replace('in S5','in S3').replace('in S7','in S5')
(HERE/'supplement/inputs.tex').write_text(inputs,encoding='utf-8')
(HERE/'supplement/procedures.tex').write_text(procedures,encoding='utf-8')
(HERE/'supplement/evidence.tex').write_text(evidence,encoding='utf-8')
sup=(OLD/'supplement.tex').read_text(encoding='utf-8')
sup=sup.replace(r'\graphicspath{{figures/}}',r'\graphicspath{{images/}}')
sup=sup.replace(r'\input{supplement/model.tex}'+'\n','')
sup=sup.replace('This document specifies the inputs, formulation and numerical interpretation of the annual and event results.','This document supplies additional input records, annual and event procedures, numerical checks and investment assumptions. The main paper retains its IT, UPS and cooling formulation and existing descriptive material.')
(HERE/'supplement.tex').write_text(sup,encoding='utf-8')

# Current input figure: direct rendering of stored data, without running the simulation.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader((HERE/'data/load_profiles.csv').open()))[:96]
x=[j/4 for j in range(96)]
inflex=[100*float(r['inflexible_load']) for r in rows]
flex=[100*float(r['flexible_load']) for r in rows]
fig,ax=plt.subplots(figsize=(8,3.5))
ax.bar(x,inflex,width=.25,align='edge',label='Inflexible',color='#454545')
ax.bar(x,flex,bottom=inflex,width=.25,align='edge',label='Flexible',color='#0072B2')
ax.set(xlim=(0,24),ylim=(0,100),xlabel='Local time (h)',ylabel='CPU utilisation (%)')
ax.set_xticks(range(0,25,3));ax.legend(ncol=2,frameon=False,loc='upper left')
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(HERE/'images/Figure_2.png',dpi=240);plt.close(fig)

# Review source mirrors the final main text plus inline additions; no generated PDF needed.
original=(ROOT/'paper/first_edit.tex').read_text(encoding='utf-8')
current=(HERE/'main.tex').read_text(encoding='utf-8')
expanded=current
for name in ('annual_operation.tex','ups_investment.tex'):
    expanded=expanded.replace(r'\input{'+name+'}',(HERE/name).read_text(encoding='utf-8'))
(HERE/'review/expanded_main.tex').write_text(expanded,encoding='utf-8')
patch=''.join(difflib.unified_diff(original.splitlines(True),expanded.splitlines(True),fromfile='first_edit.tex',tofile='minimal_revision (additions expanded)'))
(HERE/'review/first_edit_to_minimal.patch').write_text(patch,encoding='utf-8')
diff=difflib.HtmlDiff(tabsize=4,wrapcolumn=95).make_file(original.splitlines(),expanded.splitlines(),fromdesc='first_edit.tex',todesc='Minimal revision (new passages expanded)',context=True,numlines=2,charset='utf-8')
diff=diff.replace('</head>','<style>body{font-family:Arial,sans-serif}table.diff{font:12px Consolas,monospace;width:100%}.diff_header{background:#eee}td{vertical-align:top}.diff_add{background:#dff3df}.diff_sub{background:#f9d9d9}.diff_chg{background:#fff0bf}</style></head>')
(HERE/'review/comparison.html').write_text(diff,encoding='utf-8')

# Preservation checks distinguish prose from the nomenclature embedded in Introduction.
def section(text,start,end):return text[text.index(start):text.index(end,text.index(start))]
intro_old=section(original,r'\section{Introduction}',r'\section{Literature Review}')
intro_new=section(expanded,r'\section{Introduction}',r'\section{Literature Review}')
def without_nomenclature(text):
    a=text.index(r'\afterpage{');b=text.index('% --- END OF \\afterpage BLOCK ---',a)+len('% --- END OF \\afterpage BLOCK ---')
    return text[:a]+text[b:]
preservation={}
for label,o,n in [('Introduction prose',without_nomenclature(intro_old),without_nomenclature(intro_new)),('Literature Review',section(original,r'\section{Literature Review}',r'\section{Methodology}'),section(expanded,r'\section{Literature Review}',r'\section{Methodology}'))]:
    ot=o.split();nt=n.split();sm=difflib.SequenceMatcher(None,ot,nt,autojunk=False)
    preservation[label]={'original_tokens':len(ot),'new_tokens':len(nt),'original_tokens_retained_percent':round(100*sum(m.size for m in sm.get_matching_blocks())/len(ot),1)}
oldlabels=set(re.findall(r'\\label\{([^}]+)\}',original));newlabels=set(re.findall(r'\\label\{([^}]+)\}',expanded))
missing=sorted(oldlabels-newlabels)
assert not missing,missing
expected_sections=['Introduction','Literature Review','Methodology','Case Studies: Integrated DC Model','Results','Conclusion']
actual_sections=re.findall(r'^\\section\{([^}]+)\}',expanded,re.M)
assert actual_sections==expected_sections,actual_sections
assert (ROOT/'paper/first_edit.tex').read_text(encoding='utf-8')==original
preservation['all_original_labels_retained']=True
preservation['top_level_sections']=actual_sections
preservation['first_edit_sha256']=hashlib.sha256((ROOT/'paper/first_edit.tex').read_bytes()).hexdigest()
(HERE/'review/preservation_check.json').write_text(json.dumps(preservation,indent=2))
print(json.dumps(preservation,indent=2))
