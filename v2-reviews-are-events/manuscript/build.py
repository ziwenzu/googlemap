"""Build main.pdf and supplement.pdf.  Uses BibTeX with chicago.bst when available, otherwise make_bbl.py."""
import subprocess, shutil, sys, os

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

have_chicago = subprocess.run(["kpsewhich", "chicago.bst"], capture_output=True, text=True).stdout.strip() != ""
for stem in ("main", "supplement"):
    run(["pdflatex", "-interaction=nonstopmode", stem])
for stem in ("main", "supplement"):
    if have_chicago:
        run(["bibtex", stem])
if not have_chicago:
    import make_bbl; make_bbl.main(["main", "supplement"])
for _ in range(3):
    for stem in ("main", "supplement"):
        code, log = run(["pdflatex", "-interaction=nonstopmode", stem])
for stem in ("main", "supplement"):
    log = open(f"{stem}.log", encoding="latin1").read()
    bad = [l for l in log.splitlines() if ("undefined" in l.lower() and "warning" in l.lower()) or l.startswith("!")]
    print(stem, "OK" if not bad else bad[:12])
