#!/usr/bin/env python3
"""Build the template-matched reading copy and split exports in a temporary folder."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def validate_exhibit_inputs(recorder: str) -> None:
    """Reject non-PDF graphics and CSV tables actually loaded by the TeX build."""
    prohibited = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
        ".webp", ".svg", ".eps", ".csv",
    }
    invalid = sorted({
        line.removeprefix("INPUT ").strip()
        for line in recorder.splitlines()
        if line.startswith("INPUT ")
        and Path(line.removeprefix("INPUT ").strip()).suffix.lower() in prohibited
    })
    if invalid:
        raise ValueError(
            "Manuscript exhibits must use PDF figures and TeX tables; "
            "prohibited inputs: " + ", ".join(invalid)
        )


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    manuscript = project / "manuscript"
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise SystemExit("latexmk was not found; install a working TeX distribution.")

    # Regenerate reported numbers from saved replications, never from prose.
    subprocess.run(
        ["Rscript", "scripts/export-prototype-summary.R"], cwd=project, check=True
    )

    # The SI reads the main article's labels before the temporary build is removed.
    with tempfile.TemporaryDirectory(prefix="informative-traces-pa-") as temporary:
        build_dir = Path(temporary)
        env = os.environ.copy()
        env["TEXINPUTS"] = os.pathsep.join(
            [str(build_dir), str(manuscript), env.get("TEXINPUTS", "")]
        )
        env["BIBINPUTS"] = os.pathsep.join(
            [str(project / "sources"), env.get("BIBINPUTS", "")]
        )
        stems = ("Main", "pa-main", "pa-si", "research-plan")
        for stem in stems:
            result = subprocess.run(
                [
                    latexmk,
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-outdir={build_dir}",
                    f"{stem}.tex",
                ],
                cwd=manuscript,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode:
                print(result.stdout)
                raise SystemExit(result.returncode)
            try:
                validate_exhibit_inputs(
                    (build_dir / f"{stem}.fls").read_text(errors="replace")
                )
            except ValueError as error:
                raise SystemExit(str(error)) from error
            log = (build_dir / f"{stem}.log").read_text(errors="replace")
            warnings = [
                line for line in log.splitlines()
                if "Warning" in line or "Overfull" in line
            ]
            print(f"Built and checked: {stem}.pdf", flush=True)
            for warning in warnings:
                print(f"  {warning}", flush=True)
            if not warnings:
                print("  No LaTeX warnings or overfull boxes in the final pass.", flush=True)
            print("  Exhibit-input check passed: PDF graphics / TeX tables only.", flush=True)
        # Do not replace the reading copies with a partially successful build.
        for stem in stems:
            destination = manuscript / f"{stem}.pdf"
            shutil.copy2(build_dir / f"{stem}.pdf", destination)
            print(f"Saved: {destination}")
    print("Temporary build files removed; only PDFs and source files retained.")


if __name__ == "__main__":
    main()
