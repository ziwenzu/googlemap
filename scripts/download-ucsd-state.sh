#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 STATE [--check|--download|--metadata-only]" >&2
  echo "Example: $0 Vermont --download" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

STATE_INPUT="$1"
MODE="$2"
case "$MODE" in
  --check|--download|--metadata-only) ;;
  *) usage; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw/ucsd-google-local"
MANIFEST="$PROJECT_ROOT/data/download-manifest.tsv"
BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal"
STATE_FILE="${STATE_INPUT// /_}"

mkdir -p "$RAW_DIR"

content_length() {
  local url="$1"
  curl -fsSIL "$url" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub("\r", "", $2); n=$2} END{print n+0}'
}

human_bytes() {
  awk -v n="$1" 'BEGIN {
    split("B KiB MiB GiB TiB", u, " "); i=1;
    while (n >= 1024 && i < 5) {n/=1024; i++}
    printf "%.2f %s", n, u[i]
  }'
}

record_manifest() {
  local kind="$1"
  local url="$2"
  local path="$3"
  local bytes sha now rel
  bytes="$(wc -c < "$path" | tr -d ' ')"
  sha="$(shasum -a 256 "$path" | awk '{print $1}')"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  rel="${path#"$PROJECT_ROOT/"}"
  if awk -F '\t' -v state="$STATE_INPUT" -v kind="$kind" -v url="$url" \
    -v bytes="$bytes" -v sha="$sha" -v rel="$rel" \
    '($3 == state && $4 == kind && $5 == url && $6 == bytes && $7 == sha && $8 == rel) {found=1} END {exit !found}' \
    "$MANIFEST"; then
    echo "Manifest already records verified file: $rel"
    return
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$now" "UCSD Google Local Data (2021)" "$STATE_INPUT" "$kind" "$url" \
    "$bytes" "$sha" "$rel" "complete state archive" >> "$MANIFEST"
}

verify_existing() {
  local kind="$1"
  local url="$2"
  local path="$3"
  local expected="$4"
  local actual
  actual="$(wc -c < "$path" | tr -d ' ')"
  if [[ "$actual" -ne "$expected" ]]; then
    return 1
  fi
  gzip -t "$path"
  record_manifest "$kind" "$url" "$path"
  echo "Already complete and verified: $path ($(human_bytes "$actual"))"
}

download_one() {
  local kind="$1"
  local url="$2"
  local target="$3"
  local expected actual
  expected="$(content_length "$url")"
  if [[ "$expected" -le 0 ]]; then
    echo "Could not determine remote size for $url" >&2
    exit 1
  fi
  if [[ -f "$target" ]]; then
    if verify_existing "$kind" "$url" "$target" "$expected"; then
      return
    fi
    actual="$(wc -c < "$target" | tr -d ' ')"
    echo "Existing file failed size or gzip verification: $target ($actual bytes)" >&2
    echo "Leaving it untouched; a fresh transfer will be written to $target.part" >&2
  fi
  echo "Downloading $kind for $STATE_INPUT: $(human_bytes "$expected")"
  curl -fL --retry 4 --retry-delay 3 --continue-at - --output "$target.part" "$url"
  actual="$(wc -c < "$target.part" | tr -d ' ')"
  if [[ "$actual" -ne "$expected" ]]; then
    echo "Size mismatch for $target.part: expected $expected, got $actual" >&2
    exit 1
  fi
  mv "$target.part" "$target"
  gzip -t "$target"
  record_manifest "$kind" "$url" "$target"
}

META_URL="$BASE_URL/meta-$STATE_FILE.json.gz"
REVIEW_URL="$BASE_URL/review-$STATE_FILE.json.gz"
META_TARGET="$RAW_DIR/meta-$STATE_FILE.json.gz"
REVIEW_TARGET="$RAW_DIR/review-$STATE_FILE.json.gz"

META_BYTES="$(content_length "$META_URL")"
REVIEW_BYTES="$(content_length "$REVIEW_URL")"
TOTAL_BYTES=$((META_BYTES + REVIEW_BYTES))

echo "State: $STATE_INPUT"
echo "Metadata: $(human_bytes "$META_BYTES") — $META_URL"
echo "Reviews:  $(human_bytes "$REVIEW_BYTES") — $REVIEW_URL"
echo "Total:    $(human_bytes "$TOTAL_BYTES")"
df -h "$PROJECT_ROOT" | awk 'NR==1 || NR==2 {print}'

if [[ "$MODE" == "--check" ]]; then
  exit 0
fi

download_one "metadata" "$META_URL" "$META_TARGET"
if [[ "$MODE" == "--download" ]]; then
  download_one "reviews" "$REVIEW_URL" "$REVIEW_TARGET"
fi
