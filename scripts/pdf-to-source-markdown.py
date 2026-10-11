#!/usr/bin/env python3
"""Fallback PDF-to-Markdown conversion preserving page-indexed extracted text."""
from pathlib import Path
import subprocess
import sys
import textwrap

source, destination = map(Path, sys.argv[1:3])
result = subprocess.run(
    ["pdftotext", "-layout", str(source), "-"],
    check=True, capture_output=True, text=True,
)
if len(result.stdout.strip()) < 200:
    raise SystemExit("Insufficient text layer; OCR required before this source can be cited.")
pages = result.stdout.split("\f")
if not pages[-1].strip():
    pages.pop()
header = (
    f"# {source.stem}\n\n"
    f"Original: `sources/og/{source.name}`\n\n"
    "Conversion: Poppler pdftotext -layout fallback. PDF page numbers are retained.\n"
    "Publisher download/IP watermarks removed; common page indentation normalized.\n"
    "Quality flag: mathematical symbols and scanned OCR require claim-specific review;\n"
    "this extraction is not a verified mathematical transcription.\n\n"
)
def clean_page(page):
    lines = [line for line in page.splitlines()
             if not line.lstrip().startswith("Downloaded from https://www.cambridge.org/")]
    return textwrap.dedent("\n".join(lines)).strip()


content = header + "\n\n".join(
    f"## PDF page {number}\n\n```text\n{clean_page(page)}\n```"
    for number, page in enumerate(pages, 1)
) + "\n"
destination.write_text(content)
print(f"Extracted {len(pages)} pages to {destination}")
