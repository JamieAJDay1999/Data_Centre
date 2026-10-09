"""Overlay the agreed September wording edits on the original blue diff."""
from pathlib import Path
import json
import hashlib
import re

HERE = Path(__file__).resolve().parent


def mark_latest_changes(marked):
    # Project visible source back onto its original offsets, omitting only diff
    # wrappers and comments. Splitting at wrappers preserves earlier blue spans.
    chars, offsets = [], []

    def scan(start, end):
        i = start
        while i < end:
            if marked[i] == '%' and (i == 0 or marked[i - 1] != '\\'):
                newline = marked.find('\n', i, end)
                i = end if newline < 0 else newline
                continue
            match = re.match(r'\\DIFadd(?:FL)?\{', marked[i:end])
            if match:
                body = i + match.end()
                j, depth = body, 1
                while depth:
                    if marked[j] == '{' and marked[j - 1] != '\\':
                        depth += 1
                    elif marked[j] == '}' and marked[j - 1] != '\\':
                        depth -= 1
                    j += 1
                scan(body, j - 1)
                i = j
                continue
            match = re.match(r'\\DIF(?:add|del)(?:begin|end)(?:FL)?\b', marked[i:end])
            if match:
                i += match.end()
                continue
            if not marked[i].isspace():
                chars.append(marked[i])
                offsets.append(i)
            i += 1

    scan(marked.index(r'\begin{document}'), len(marked))
    # latexdiff inserts separator spaces beside its invisible commands. Remove
    # these when the next visible character is punctuation, including at a
    # colour boundary. The normalised text and its green selections stay intact.
    for index in range(len(chars) - 1, 0, -1):
        if chars[index] not in ',.;:!?':
            continue
        start, end = offsets[index - 1] + 1, offsets[index]
        gap = marked[start:end]
        if '%' not in gap and any(c.isspace() for c in gap):
            marked = marked[:start] + re.sub(r'\s+', '', gap) + marked[end:]
    chars.clear()
    offsets.clear()
    scan(marked.index(r'\begin{document}'), len(marked))
    flattened = ''.join(chars)
    ranges = []
    for phrase in json.loads((HERE / 'review/latest_green_phrases.json').read_text(encoding='utf-8')):
        needle = re.sub(r'\s+', '', phrase)
        matches = [m.start() for m in re.finditer(re.escape(needle), flattened)]
        assert len(matches) == 1, (phrase, len(matches))
        ranges.append((matches[0], matches[0] + len(needle)))
    editorial_path = HERE / 'review/editorial_green_spans.json'
    if editorial_path.exists():
        editorial = json.loads(editorial_path.read_text(encoding='utf-8'))
        assert editorial['sha256'] == hashlib.sha256(flattened.encode()).hexdigest(), 'Editorial green offsets do not match the current text'
        ranges.extend(tuple(span) for span in editorial['spans'])
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    spans = []
    for start, end in merged:
        selected = offsets[start:end]
        begin = previous = selected[0]
        for current in selected[1:]:
            if marked[previous + 1:current].strip():
                spans.append((begin, previous + 1))
                begin = current
            previous = current
        spans.append((begin, previous + 1))
    for start, end in sorted(spans, reverse=True):
        fragment = marked[start:end]
        depth = 0
        for token in re.findall(r'(?<!\\)[{}]', fragment):
            depth += 1 if token == '{' else -1
            assert depth >= 0, fragment
        assert depth == 0, fragment
        marked = marked[:start] + r'\textcolor{LatestGreen}{' + fragment + '}' + marked[end:]
    marked = marked.replace(r'\begin{document}',
                            r'\definecolor{LatestGreen}{RGB}{0,0,255}' + '\n'
                            + r'\pdfstringdefDisableCommands{\def\textcolor#1#2{#2}\def\DIFadd#1{#1}\def\DIFaddbegin{}\def\DIFaddend{}}'
                            + '\n' + r'\begin{document}', 1)
    return marked


if __name__ == '__main__':
    path = HERE / 'main.tex'
    path.write_text(mark_latest_changes(path.read_text(encoding='utf-8')), encoding='utf-8')
