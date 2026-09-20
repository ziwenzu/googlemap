#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw/ucsd-google-local"
OUTPUT_DIR="$PROJECT_ROOT/data/interim/government-places-by-state"
COMBINED="$PROJECT_ROOT/data/interim/government-places-all.csv"
TAXONOMY="$PROJECT_ROOT/config/government-place-taxonomy.csv"
FILTER="$PROJECT_ROOT/scripts/01_filter_government_places.py"
PARTITIONS="$PROJECT_ROOT/scripts/download-ucsd-all.sh"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUTPUT_DIR"
combined_part="$COMBINED.part"
: > "$combined_part"
first=1
index=0
total="$($PARTITIONS --list | wc -l | tr -d ' ')"

while IFS= read -r state; do
  index=$((index + 1))
  state_file="${state// /_}"
  if [[ "$state" == "Other" ]]; then
    state_file="other"
  fi
  metadata="$RAW_DIR/meta-$state_file.json.gz"
  output="$OUTPUT_DIR/government-places-$state_file.csv"
  if [[ ! -f "$metadata" ]]; then
    echo "Missing metadata archive: $metadata" >&2
    echo "Run ./scripts/download-ucsd-all.sh --metadata-only or --download first." >&2
    exit 1
  fi
  printf '[%d/%d] Filtering %s\n' "$index" "$total" "$state"
  "$PYTHON" "$FILTER" \
    --metadata "$metadata" \
    --taxonomy "$TAXONOMY" \
    --output "$output"
  if [[ "$first" -eq 1 ]]; then
    cat "$output" >> "$combined_part"
    first=0
  else
    tail -n +2 "$output" >> "$combined_part"
  fi
done < <("$PARTITIONS" --list)

mv "$combined_part" "$COMBINED"
echo "Combined candidate retrieval table: $COMBINED"

