#!/usr/bin/env python3
"""Compile artificial-data exhibits with the manuscript's supplied house style."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    for item in ("tables/prototype-monte-carlo.tex",
                 "figures/prototype-selection-distributions.pdf"):
        if not (root / item).exists():
            raise SystemExit("Run the prototype simulation first: " + item)
    with tempfile.TemporaryDirectory(prefix="trace-prototype-exhibits-") as temp:
        env = os.environ.copy()
        env["TEXINPUTS"] = os.pathsep.join(
            [str(root / "manuscript"), str(root / "simulation"), env.get("TEXINPUTS", "")]
        )
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
             "-outdir=" + temp, str(root / "simulation/prototype-exhibits.tex")],
            cwd=root / "manuscript", env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode:
            print(result.stdout)
            raise SystemExit(result.returncode)
        log = (Path(temp) / "prototype-exhibits.log").read_text(errors="replace")
        warnings = [line for line in log.splitlines() if "Warning" in line or "Overfull" in line]
        fls = (Path(temp) / "prototype-exhibits.fls").read_text(errors="replace")
        for line in fls.splitlines():
            if line.startswith("INPUT ") and Path(line[6:]).suffix.lower() in {
                ".png", ".jpg", ".jpeg", ".csv"
            }:
                raise SystemExit("Prohibited non-PDF/TeX exhibit input: " + line)
        dest = root / "simulation/prototype-output/prototype-exhibits.pdf"
        shutil.copy2(Path(temp) / "prototype-exhibits.pdf", dest)
        print(dest)
        print("\n".join(warnings) if warnings else "No final LaTeX warnings or overfull boxes.")


if __name__ == "__main__":
    main()
