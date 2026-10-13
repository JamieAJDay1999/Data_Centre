"""Regenerate changes against the manuscript submitted as APEN-D-25-21553.

The baseline is the source of the Editorial Manager PDF retained in
author_records (Takci first; 10.02% single-day saving; SCIP). The 6.3% draft in
the same folder was never submitted and must not be used as the baseline.
"""
from pathlib import Path
import re, subprocess, argparse, json, hashlib

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
WORK = ROOT / '.codex_tmp/submission_20261009'

def unwrap(text, command):
    token = '\\' + command + '{'
    while token in text:
        start = text.index(token)
        i = start + len(token)
        depth, end = 1, i
        while depth:
            if text[end] == '{' and text[end-1] != '\\': depth += 1
            elif text[end] == '}' and text[end-1] != '\\': depth -= 1
            end += 1
        text = text[:start] + text[i:end-1] + text[end:]
    return text

def expand(text):
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand((PAPER/m[1]).read_text(encoding='utf-8')),text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perl',default='C:/Program Files/Git/usr/bin/perl.exe')
    parser.add_argument('--latexdiff',default='C:/Users/jamie/AppData/Local/Programs/MiKTeX/scripts/latexdiff/latexdiff-so')
    args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    old_path = HERE/'author_records/submitted_manuscript_source.tex'
    if not old_path.exists():
        old_path.write_bytes(subprocess.check_output(['git','show','a72f90c:paper/original_submission.tex'],cwd=ROOT))
    new = (PAPER/'main.tex').read_text(encoding='utf-8')
    old = old_path.read_text(encoding='utf-8-sig')
    # Remove pre-existing revision wrappers, which do not represent this comparison.
    old = unwrap(old,'rev')
    normalized_new = unwrap(expand(new),'rev')
    (WORK/'diff_old.tex').write_text(old,encoding='utf-8')
    (WORK/'diff_new.tex').write_text(normalized_new,encoding='utf-8')
    proc = subprocess.run([args.perl,args.latexdiff,'--type=CFONT','--graphics-markup=none',
        '--math-markup=off',str(WORK/'diff_old.tex'),str(WORK/'diff_new.tex')],
        cwd=PAPER,capture_output=True,timeout=120)
    (WORK/'latexdiff.log').write_bytes(proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stderr.decode('utf-8','replace')[-3000:])
    result = proc.stdout.decode('utf-8').replace('\r\n', '\n')
    body = result.split(r'\begin{document}',1)[1]
    # Keep the current title/author/header machinery exactly; the scientific
    # comparison begins with the abstract. The author order changed from the
    # submitted version (Takci first), so the author line is marked as changed.
    front = normalized_new.split(r'\begin{document}',1)[1].split(r'\begin{abstract}',1)[0]
    author = re.search(r'\\author\{(.*?)\}\n', front)
    assert author and 'Takci' in author[1], 'Author line not found'
    front = front.replace(author[0], '\\author{\\color{blue}' + author[1] + '}\n')
    body = front + r'\begin{abstract}' + body.split(r'\begin{abstract}',1)[1]
    # Suppress deletions. A removed passage can leave a space before following
    # punctuation ("approximation ."), so drop that space at deletion sites.
    body = re.sub(r'\\DIFdelbegin(?:FL)?\b.*?\\DIFdelend(?:FL)?\b', '\x00', body, flags=re.S)
    body = re.sub(r'[ \t]*\x00[ \t]*(?=[.,;:)])', '', body)
    body = body.replace('\x00', '')
    body = re.sub(r'(?<!\\)%[^\n]*', '', body)
    # \DIFaddend is a control word, so TeX discards the space after it. When the
    # added text does not itself end in a space, words merge ("nearlyinstantaneous").
    body = re.sub(r'([^ \n])\}\\DIFaddend(FL)?( +)(?=[A-Za-z0-9$\\(])', r'\1}\\DIFaddend\2{}\3', body)
    # Added punctuation should attach to the preceding word ("dichotomy :").
    body = re.sub(r'[ \t]+(\\DIFaddbegin(?:FL)? \\DIFadd(?:FL)?\{[.,;:])', r'\1', body)
    # Inline differencing of aligned maths can split operators or alignment
    # delimiters. Keep each current display intact and colour changed displays.
    displays = re.compile(r'\\begin\{(equation\*?|align\*?|flalign\*?|gather\*?|multline\*?)\}.*?\\end\{\1\}', re.S)
    compact = lambda s: re.sub(r'\s+', '', s)
    old_displays = {compact(m.group()) for m in displays.finditer(old)}
    def colour_display(match):
        value = re.sub(r'\n\s*\n', '\n', match.group())
        if compact(value) in old_displays:
            return value
        return '\\begingroup\\color{blue}\n' + value + '\n\\endgroup'
    body = displays.sub(colour_display, body)
    preamble = normalized_new.split(r'\begin{document}',1)[0]
    macros = r'''
% Additions and replacements relative to submitted manuscript APEN-D-25-21553
% are blue. Deleted passages are suppressed.
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
    (PAPER/'main_marked.tex').write_text(preamble+macros+r'\begin{document}'+body,encoding='utf-8')
    print('Marked manuscript generated against submitted APEN-D-25-21553 source (10.02% daily saving).')

if __name__ == '__main__': main()



