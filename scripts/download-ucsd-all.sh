#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--check|--download|--metadata-only|--list]" >&2
  echo "Downloads all U.S. state partitions, District of Columbia, and Other." >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

MODE="$1"
case "$MODE" in
  --check|--download|--metadata-only|--list) ;;
  *) usage; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_DOWNLOADER="$SCRIPT_DIR/download-ucsd-state.sh"

STATES=(
  "Alabama"
  "Alaska"
  "Arizona"
  "Arkansas"
  "California"
  "Colorado"
  "Connecticut"
  "Delaware"
  "District of Columbia"
  "Florida"
  "Georgia"
  "Hawaii"
  "Idaho"
  "Illinois"
  "Indiana"
  "Iowa"
  "Kansas"
  "Kentucky"
  "Louisiana"
  "Maine"
  "Maryland"
  "Massachusetts"
  "Michigan"
  "Minnesota"
  "Mississippi"
  "Missouri"
  "Montana"
  "Nebraska"
  "Nevada"
  "New Hampshire"
  "New Jersey"
  "New Mexico"
  "New York"
  "North Carolina"
  "North Dakota"
  "Ohio"
  "Oklahoma"
  "Oregon"
  "Pennsylvania"
  "Rhode Island"
  "South Carolina"
  "South Dakota"
  "Tennessee"
  "Texas"
  "Utah"
  "Vermont"
  "Virginia"
  "Washington"
  "West Virginia"
  "Wisconsin"
  "Wyoming"
  "Other"
)

if [[ "$MODE" == "--list" ]]; then
  printf '%s\n' "${STATES[@]}"
  exit 0
fi

total="${#STATES[@]}"
for index in "${!STATES[@]}"; do
  state="${STATES[$index]}"
  printf '\n[%d/%d] %s — %s\n' "$((index + 1))" "$total" "$state" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$STATE_DOWNLOADER" "$state" "$MODE"
done

