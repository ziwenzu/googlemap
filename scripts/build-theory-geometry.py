#!/usr/bin/env python3
"""Build the analytical companion with the existing manuscript template."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["Rscript", "scripts/draw-theory-geometry.R"], cwd=root, check=True)
    subprocess.run(["python3", "scripts/check-pa-mathematics.py"], cwd=root, check=True)
    with tempfile.TemporaryDirectory(prefix="trace-theory-geometry-") as temp:
        env = os.environ.copy()
        env["TEXINPUTS"] = os.pathsep.join([str(root / "manuscript"), env.get("TEXINPUTS", "")])
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
             "-outdir=" + temp, "theory-geometry.tex"],
            cwd=root / "manuscript", env=env, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
        directory = Path(temp)
        log = (directory / "theory-geometry.log").read_text(errors="replace")
        for line in (directory / "theory-geometry.fls").read_text(errors="replace").splitlines():
            if line.startswith("INPUT ") and Path(line[6:]).suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".csv"}:
                raise SystemExit("Unexpected raster/CSV input: " + line)
        warnings = [s for s in log.splitlines() if "Warning" in s or "Overfull" in s]
        if warnings:
            print("\n".join(warnings))
        else:
            print("No final LaTeX warnings or overfull boxes.")
        target = root / "manuscript/theory-geometry.pdf"
        shutil.copy2(directory / target.name, target)
        print(target)


if __name__ == "__main__":
    main()
