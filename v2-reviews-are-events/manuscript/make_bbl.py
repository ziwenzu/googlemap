"""Write natbib-compatible .bbl files in Chicago author-date style without BibTeX.

Only needed where chicago.bst is not installed.  With a full TeX distribution, run BibTeX instead; the .tex
files already say \\bibliographystyle{chicago}.
Usage: python3 make_bbl.py main supplement
"""
import re
import sys


def parse_bib(path):
    txt = open(path, encoding="utf8").read()
    txt = re.sub(r"(?m)%.*$", "", txt)
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", txt):
        kind, key = m.group(1).lower(), m.group(2)
        i, depth, start = m.end(), 1, m.end()
        while depth and i < len(txt):
            depth += {"{": 1, "}": -1}.get(txt[i], 0)
            i += 1
        body = txt[start:i - 1]
        fields, pos = {}, 0
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            if fm.start() < pos:
                continue
            j = fm.end()
            if body[j] == "{":
                d, k = 1, j + 1
                while d:
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                val = body[j + 1:k - 1]
            else:
                k = body.find(",", j)
                k = len(body) if k < 0 else k
                val = body[j:k].strip().strip('"')
            fields[fm.group(1).lower()] = " ".join(val.split())
            pos = k
        entries[key] = (kind, fields)
    return entries


def split_authors(a):
    out = []
    for name in re.split(r"\s+and\s+", a):
        if "," in name:
            last, first = [x.strip() for x in name.split(",", 1)]
        else:
            parts = name.split()
            last, first = parts[-1], " ".join(parts[:-1])
        out.append((last, first))
    return out


def plain(s):
    return re.sub(r"[{}\\'\"`^~=.v ]|\\[a-zA-Z]+", "", s).lower()


def author_list(auths):
    names = [f"{auths[0][0]}, {auths[0][1]}"] + [f"{f} {l}" for l, f in auths[1:]]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def short_label(auths):
    lasts = [l for l, _ in auths]
    if len(lasts) == 1:
        return lasts[0]
    if len(lasts) == 2:
        return f"{lasts[0]} and {lasts[1]}"
    if len(lasts) == 3:
        return f"{lasts[0]}, {lasts[1]}, and {lasts[2]}"
    return f"{lasts[0]} et~al."


def fmt(key, kind, f):
    auths = split_authors(f["author"])
    names = author_list(auths)
    names = names if names.endswith(".") else names + "."
    year, title = f["year"], f["title"]
    if kind == "article":
        s = f"{names} {year}. ``{title}.'' \\emph{{{f['journal']}}}"
        if "volume" in f:
            s += f" {f['volume']}"
            if "number" in f:
                s += f" ({f['number']})"
            if "pages" in f:
                s += f": {f['pages'].replace('--', '--')}"
            s += "."
        else:
            s += f", {f.get('note', 'forthcoming')}." + (f" doi:{f['doi']}." if "doi" in f else "")
    elif kind == "book":
        ed = {"2": "2nd"}.get(f.get("edition", ""), f.get("edition", ""))
        s = f"{names} {year}. \\emph{{{title}}}." + (f" {ed} ed." if ed else "") + f" {f['address']}: {f['publisher']}."
    else:  # inproceedings
        s = f"{names} {year}. ``{title}.'' In \\emph{{{f['booktitle']}}}" + (f", {f['pages']}" if "pages" in f else "") + "."
        if "publisher" in f:
            s += f" {f['address']}: {f['publisher']}."
    label = f"{short_label(auths)}({year}){author_list([(l, '') for l, _ in auths]).replace(', ,', ',')}"
    return f"\\bibitem[{{{short_label(auths)}({year})}}]{{{key}}}\n{s}\n"


def main(stems):
    bib = parse_bib("refs.bib")
    for stem in stems:
        aux = open(f"{stem}.aux", encoding="utf8").read()
        keys = []
        for m in re.finditer(r"\\citation\{([^}]*)\}", aux):
            for k in m.group(1).split(","):
                if k.strip() and k.strip() not in keys:
                    keys.append(k.strip())
        missing = [k for k in keys if k not in bib]
        if missing:
            raise SystemExit(f"{stem}: keys missing from refs.bib: {missing}")
        keys.sort(key=lambda k: (plain(split_authors(bib[k][1]["author"])[0][0]), plain(bib[k][1]["author"]), bib[k][1]["year"]))
        with open(f"{stem}.bbl", "w", encoding="utf8") as out:
            out.write("\\begin{thebibliography}{99}\n\\providecommand{\\natexlab}[1]{#1}\n\n")
            for k in keys:
                out.write(fmt(k, *bib[k]) + "\n")
            out.write("\\end{thebibliography}\n")
        print(f"{stem}.bbl: {len(keys)} references")


if __name__ == "__main__":
    main(sys.argv[1:] or ["main", "supplement"])
