#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--check|--download|--list]" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

MODE="$1"
case "$MODE" in
  --check|--download|--list) ;;
  *) usage; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/external/official-rosters/raw"
MANIFEST="$PROJECT_ROOT/data/download-manifest.tsv"

mkdir -p "$RAW_DIR"

content_length() {
  local url="$1"
  local headers
  headers="$(curl -fsSIL "$url" 2>/dev/null || true)"
  printf '%s' "$headers" | awk 'BEGIN{IGNORECASE=1} /^HTTP\// {n=0} /^content-length:/ {gsub("\r", "", $2); n=$2} END{print n+0}'
}

human_bytes() {
  awk -v n="$1" 'BEGIN {
    if (n <= 0) {print "unknown"; exit}
    split("B KiB MiB GiB TiB", u, " "); i=1;
    while (n >= 1024 && i < 5) {n/=1024; i++}
    printf "%.2f %s", n, u[i]
  }'
}

verify_archive() {
  local path="$1"
  unzip -tq "$path" >/dev/null
}

record_manifest() {
  local source="$1"
  local roster_id="$2"
  local url="$3"
  local path="$4"
  local bytes sha now rel kind
  bytes="$(wc -c < "$path" | tr -d ' ')"
  sha="$(shasum -a 256 "$path" | awk '{print $1}')"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  rel="${path#"$PROJECT_ROOT/"}"
  kind="official_roster:$roster_id"
  if awk -F '\t' -v source="$source" -v kind="$kind" -v url="$url" \
    -v bytes="$bytes" -v sha="$sha" -v rel="$rel" \
    '($2 == source && $4 == kind && $5 == url && $6 == bytes && $7 == sha && $8 == rel) {found=1} END {exit !found}' \
    "$MANIFEST"; then
    echo "Manifest already records verified file: $rel"
    return
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$now" "$source" "National" "$kind" "$url" "$bytes" "$sha" "$rel" \
    "official roster archive; retain source terms" >> "$MANIFEST"
}

download_one() {
  local roster_id="$1"
  local source="$2"
  local url="$3"
  local filename="$4"
  local target expected actual
  target="$RAW_DIR/$filename"
  expected="$(content_length "$url")"
  printf '%s\t%s\t%s\t%s\n' "$roster_id" "$(human_bytes "$expected")" "$filename" "$url"
  if [[ "$MODE" == "--check" ]]; then
    return
  fi
  if [[ -f "$target" ]]; then
    actual="$(wc -c < "$target" | tr -d ' ')"
    if [[ "$expected" -gt 0 && "$actual" -ne "$expected" ]]; then
      echo "Existing file has an unexpected size; leaving it untouched: $target" >&2
      exit 1
    fi
    verify_archive "$target"
    record_manifest "$source" "$roster_id" "$url" "$target"
    echo "Already complete and verified: $target"
    return
  fi
  if [[ -f "$target.part" ]] && verify_archive "$target.part"; then
    actual="$(wc -c < "$target.part" | tr -d ' ')"
    if [[ "$expected" -le 0 || "$actual" -eq "$expected" ]]; then
      mv "$target.part" "$target"
      record_manifest "$source" "$roster_id" "$url" "$target"
      echo "Promoted complete verified transfer: $target"
      return
    fi
  fi
  curl -fL --retry 4 --retry-delay 3 --continue-at - --output "$target.part" "$url"
  actual="$(wc -c < "$target.part" | tr -d ' ')"
  if [[ "$expected" -gt 0 && "$actual" -ne "$expected" ]]; then
    echo "Size mismatch for $target.part: expected $expected, got $actual" >&2
    exit 1
  fi
  verify_archive "$target.part"
  mv "$target.part" "$target"
  record_manifest "$source" "$roster_id" "$url" "$target"
}

if [[ "$MODE" == "--list" ]]; then
  printf '%s\n' census_gov_units_2021 imls_pls_outlets_2021 gsa_frpp_2021
  exit 0
fi

download_one \
  "census_gov_units_2021" \
  "U.S. Census Bureau Government Units Listing (2021)" \
  "https://www2.census.gov/programs-surveys/gus/datasets/2021/govt_units_2021.ZIP" \
  "census-government-units-2021.zip"

download_one \
  "imls_pls_outlets_2021" \
  "IMLS Public Libraries Survey (FY2021)" \
  "https://www.imls.gov/sites/default/files/2023-06/pls_fy2021_csv.zip" \
  "imls-pls-fy2021-csv.zip"

download_one \
  "gsa_frpp_2021" \
  "GSA Federal Real Property Profile (FY2021)" \
  "https://inventory.data.gov/dataset/ba12ef9a-2a84-4da7-a943-8772c176d5ff/resource/ec1ee26a-d1c7-4c53-87b1-58484cf948e5/download/frpp_public_dataset_fy21_final_02242023-1.xlsx" \
  "gsa-frpp-fy2021.xlsx"
