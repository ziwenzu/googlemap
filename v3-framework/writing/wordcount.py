"""Count the words of the main text the way Political Analysis does (abstract, body, figure legends and footnotes;
not the title page, words inside tables, references or the appendix) and write \\MainWordCount for the title page.

Run from writing/ after assemble.py:   python3 wordcount.py
"""
import re
from pathlib import Path

tex = Path("Main.tex").read_text(encoding="utf8")
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", Path("../results/tables/macros.tex").read_text(encoding="utf8")))


def clean(s):
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", s, flags=re.S)                  # words in tables are not counted
    s = re.sub(r"\\begin\{tablenotes\}.*?\\end\{tablenotes\}", " ", s, flags=re.S)
    s = re.sub(r"\\noindent\\emph\{\[Placeholder\..*?\]\}", " ", s, flags=re.S)               # the applications placeholder
    s = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: macros.get(m.group(1), m.group(0)), s)       # expand number macros
    s = re.sub(r"\\(parencite|textcite|cite|citep|citet)\*?(\[[^\]]*\])*\{([^}]*)\}",
               lambda m: " ".join(["Author 2000"] * len(m.group(3).split(","))), s)             # a citation is about two words
    s = re.sub(r"\\(ref|eqref|label|input|includegraphics|caption@|cntxt|pdfbookmark)(\[[^\]]*\])*\{[^}]*\}", " X ", s)
    s = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " EQ ", s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", " EQ ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " M ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}\[\]~]", " ", s)
    return [w for w in s.split() if re.search(r"[A-Za-z0-9]", w)]


abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
body = tex[tex.index(r"\section{Introduction}"):tex.index(r"\printbibliography[title={References}]")]
n_abs, n_body = len(clean(abstract)), len(clean(body))
total = n_abs + n_body
Path("../results/tables/wordcount.tex").write_text(f"\\newcommand{{\\MainWordCount}}{{{total:,}}}\n".replace(",", "{,}"))
print(f"abstract {n_abs} words; body with captions and footnotes {n_body}; total {total}")
