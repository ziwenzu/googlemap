#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCES="$PROJECT_ROOT/sources/og"
OUTPUT="$PROJECT_ROOT/sources/md"
VENV="$PROJECT_ROOT/.venv"

FORCE=false
[[ "${1:-}" == "--all" ]] && FORCE=true

if [[ -x "$VENV/bin/opendataloader-pdf" ]] && java -version >/dev/null 2>&1; then
  PDF_BACKEND=opendataloader
elif command -v pdftotext >/dev/null 2>&1; then
  PDF_BACKEND=pdftotext
  echo "Using page-indexed pdftotext fallback; equations require manual verification."
else
  echo "PDF conversion requires OpenDataLoader + Java, or Poppler pdftotext." >&2
  exit 1
fi

mkdir -p "$OUTPUT"
converted=0

for pdf in "$SOURCES"/*.pdf; do
  [[ -f "$pdf" ]] || continue
  base="$(basename "$pdf" .pdf)"
  if [[ "$FORCE" == false && -f "$OUTPUT/$base.md" ]]; then
    continue
  fi
  echo "Converting $base.pdf"
  if [[ "$PDF_BACKEND" == opendataloader ]]; then
    "$VENV/bin/opendataloader-pdf" "$pdf" --format markdown --output-dir "$OUTPUT/"
  else
    python3 "$PROJECT_ROOT/scripts/pdf-to-source-markdown.py" "$pdf" "$OUTPUT/$base.md"
  fi
  converted=$((converted + 1))
done

for docx in "$SOURCES"/*.docx; do
  [[ -f "$docx" ]] || continue
  base="$(basename "$docx" .docx)"
  if [[ "$FORCE" == false && -f "$OUTPUT/$base.md" ]]; then
    continue
  fi
  if ! command -v pandoc >/dev/null 2>&1; then
    echo "pandoc is required to convert DOCX files." >&2
    exit 1
  fi
  echo "Converting $base.docx"
  pandoc "$docx" -t markdown -o "$OUTPUT/$base.md"
  converted=$((converted + 1))
done

if [[ $converted -eq 0 ]]; then
  echo "Nothing new to convert."
else
  echo "Converted $converted file(s) into sources/md/."
fi
