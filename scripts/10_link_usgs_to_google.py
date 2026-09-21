#!/usr/bin/env python3
"""Reverse-link official USGS facilities against complete UCSD Google metadata.

The official facility frame is the search spine.  Every Google metadata record
with coordinates is compared with nearby official facilities in the same state.
Outputs remain candidates: current USGS data are not a 2021 denominator, and a
geographic/name match is not a human-adjudicated public-service label.
"""

import argparse
import csv
import gzip
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


LINKAGE_VERSION = "usgs_google_v0.2"
GRID_SIZE = 0.10
STATE_ABBREVIATIONS = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
FAMILY_PATTERNS = {
    "postal_service": r"post office|postal service|usps",
    "city_hall": (
        r"city hall|town hall|municipal|borough hall|village hall|"
        r"town office|city office|town clerk|city clerk|village clerk|"
        r"town administration|city administration"
    ),
    "court": r"court|courthouse|judicial",
    "public_health": r"hospital|medical center|health",
    "fire_emergency": r"fire|ems|ambulance|emergency",
    "police": r"police|sheriff|law enforcement|highway patrol",
    "corrections": r"prison|correction|jail|detention",
    "public_education": r"school|college|university|education",
    "parks_recreation": r"ranger|visitor center|information center|park",
    "other_government": r"government|capitol|headquarters|federal|state office",
}
GENERIC_NAME_TOKENS = {
    "THE", "OF", "AND", "AT", "OFFICE", "DEPARTMENT", "BUILDING", "CENTER",
    "STATION", "FACILITY", "SERVICE", "SERVICES", "UNITED", "STATES",
}
OUTPUT_FIELDS = [
    "linkage_version", "source_partition", "source_line", "gmap_id",
    "google_name", "google_address", "google_latitude", "google_longitude",
    "google_categories", "google_avg_rating", "google_num_of_reviews",
    "service_family", "analytic_tier_hint", "official_roster_id",
    "official_record_id", "official_name", "official_address", "official_city",
    "official_state", "official_zip", "official_latitude", "official_longitude",
    "official_owner_class", "owner_eligibility", "official_source_vintage",
    "official_source_subtype", "distance_km", "name_similarity",
    "address_similarity", "family_signal", "official_link_score",
    "official_match_status", "official_match_method",
    "official_match_multiplicity", "candidate_label",
    "classification_probability", "adjudicated_label", "exclusion_reason",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--combined-output", type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--partitions", help="Comma-separated names, e.g. Vermont,New_York")
    parser.add_argument("--max-distance-km", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(clean(item) for item in value)
    if isinstance(value, dict):
        return " | ".join(f"{clean(key)} {clean(item)}" for key, item in value.items())
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_text(value):
    text = unicodedata.normalize("NFKD", clean(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", " ", text.upper().replace("&", " AND ")).strip()


def normalize_name(value):
    tokens = [
        token for token in normalize_text(value).split()
        if token not in GENERIC_NAME_TOKENS
    ]
    return " ".join(tokens) or normalize_text(value)


def similarity(left, right, *, name=False):
    left = normalize_name(left) if name else normalize_text(left)
    right = normalize_name(right) if name else normalize_text(right)
    if not left or not right:
        return 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    containment = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return max(sequence, jaccard, containment * 0.95)


def address_similarity(google_address, official_address):
    google = normalize_text(google_address)
    official = normalize_text(official_address)
    if not google or not official:
        return 0.0
    if official in google:
        return 1.0
    google_tokens = set(google.split())
    official_tokens = set(official.split())
    overlap = len(google_tokens & official_tokens) / len(official_tokens)
    google_number = re.search(r"\b\d+\b", google)
    official_number = re.search(r"\b\d+\b", official)
    if google_number and official_number and google_number.group() != official_number.group():
        overlap *= 0.55
    return max(overlap, similarity(google, official))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    values = [as_float(value) for value in (lat1, lon1, lat2, lon2)]
    if any(value is None for value in values):
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(hav))


def grid_cell(latitude, longitude):
    latitude = as_float(latitude)
    longitude = as_float(longitude)
    if latitude is None or longitude is None:
        return None
    return math.floor(latitude / GRID_SIZE), math.floor(longitude / GRID_SIZE)


def family_signal(record, family):
    pattern = FAMILY_PATTERNS.get(family)
    if not pattern:
        return 0.0
    text = " || ".join(
        clean(record.get(field)) for field in ("name", "category", "description")
    )
    return 1.0 if re.search(pattern, text, re.I) else 0.0


def score_match(google, official, max_distance_km=3.0):
    distance = haversine_km(
        google.get("latitude"), google.get("longitude"),
        official.get("latitude"), official.get("longitude"),
    )
    if distance is None or distance > max_distance_km:
        return None
    name_score = similarity(google.get("name"), official.get("entity_name"), name=True)
    address_score = address_similarity(google.get("address"), official.get("address_line1"))
    family_score = family_signal(google, official.get("service_family"))
    geo_score = max(0.0, 1.0 - distance / max_distance_km)
    total = 0.50 * name_score + 0.20 * address_score + 0.25 * geo_score + 0.05 * family_score

    if distance <= 0.15 and family_score and name_score >= 0.55:
        status, method = "confirmed_official_candidate", "near_geo_plus_family_and_name"
    elif distance <= 0.15 and address_score >= 0.90 and name_score >= 0.70:
        status, method = "needs_review", "near_geo_plus_address_and_name_without_family"
    elif distance <= 1.0 and family_score and name_score >= 0.82:
        status, method = "confirmed_official_candidate", "family_name_plus_geo"
    elif distance <= 2.0 and family_score and name_score >= 0.90:
        status, method = "confirmed_official_candidate", "strong_family_name_plus_geo"
    elif distance <= 0.10 and (
        family_score or name_score >= 0.45 or address_score >= 0.80
    ):
        status, method = "needs_review", "near_geo_with_partial_evidence"
    elif distance <= 1.0 and name_score >= 0.55 and (
        family_score or address_score >= 0.70
    ):
        status, method = "needs_review", "partial_name_plus_family_or_address"
    elif distance <= 3.0 and name_score >= 0.75 and family_score:
        status, method = "needs_review", "family_name_plus_geo"
    else:
        return None
    return {
        "status": status,
        "method": method,
        "score": total,
        "distance_km": distance,
        "name_similarity": name_score,
        "address_similarity": address_score,
        "family_signal": family_score,
    }


def load_official(path):
    records = []
    by_state_grid = defaultdict(list)
    state_totals = Counter()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("owner_eligibility") == "exclude_private_owner":
                continue
            cell = grid_cell(row.get("latitude"), row.get("longitude"))
            state = row.get("state", "")
            if not cell or not state:
                continue
            index = len(records)
            records.append(row)
            by_state_grid[(state, cell[0], cell[1])].append(index)
            state_totals[state] += 1
    return records, by_state_grid, state_totals


def nearby_official(google, state, records, by_state_grid):
    cell = grid_cell(google.get("latitude"), google.get("longitude"))
    if not cell:
        return []
    indices = set()
    for lat_offset in (-1, 0, 1):
        for lon_offset in (-1, 0, 1):
            indices.update(
                by_state_grid.get((state, cell[0] + lat_offset, cell[1] + lon_offset), ())
            )
    return [records[index] for index in indices]


def output_row(partition, line_number, google, official, score):
    official_address = " ".join(
        value for value in (
            official.get("address_line1", ""), official.get("address_line2", "")
        ) if value
    )
    return {
        "linkage_version": LINKAGE_VERSION,
        "source_partition": partition,
        "source_line": line_number,
        "gmap_id": clean(google.get("gmap_id")),
        "google_name": clean(google.get("name")),
        "google_address": clean(google.get("address")),
        "google_latitude": clean(google.get("latitude")),
        "google_longitude": clean(google.get("longitude")),
        "google_categories": clean(google.get("category")),
        "google_avg_rating": clean(google.get("avg_rating")),
        "google_num_of_reviews": clean(google.get("num_of_reviews")),
        "service_family": official.get("service_family", ""),
        "analytic_tier_hint": official.get("analytic_tier_hint", ""),
        "official_roster_id": official.get("roster_id", ""),
        "official_record_id": official.get("source_record_id", ""),
        "official_name": official.get("entity_name", ""),
        "official_address": official_address,
        "official_city": official.get("city", ""),
        "official_state": official.get("state", ""),
        "official_zip": official.get("zip", ""),
        "official_latitude": official.get("latitude", ""),
        "official_longitude": official.get("longitude", ""),
        "official_owner_class": official.get("official_owner_class", ""),
        "owner_eligibility": official.get("owner_eligibility", ""),
        "official_source_vintage": official.get("source_vintage", ""),
        "official_source_subtype": official.get("source_subtype", ""),
        "distance_km": f"{score['distance_km']:.4f}",
        "name_similarity": f"{score['name_similarity']:.4f}",
        "address_similarity": f"{score['address_similarity']:.4f}",
        "family_signal": f"{score['family_signal']:.0f}",
        "official_link_score": f"{score['score']:.4f}",
        "official_match_status": score["status"],
        "official_match_method": score["method"],
        "official_match_multiplicity": "",
        "candidate_label": "",
        "classification_probability": "",
        "adjudicated_label": "",
        "exclusion_reason": "",
    }


def partition_name(path):
    return path.name.removeprefix("meta-").removesuffix(".json.gz")


def partition_state(partition):
    return STATE_ABBREVIATIONS.get(partition.replace("_", " "), "")


def process_partition(path, output_path, summary_path, records, by_state_grid, state_totals, max_distance):
    partition = partition_name(path)
    state = partition_state(partition)
    if not state:
        summary = {
            "partition": partition,
            "state": "",
            "status": "skipped_non_state_partition",
            "metadata_records_seen": 0,
            "candidate_links": 0,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with gzip.open(output_path, "wt", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS).writeheader()
        return summary

    best_by_google_family = {}
    metadata_seen = 0
    decode_errors = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            metadata_seen += 1
            try:
                google = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            if not google.get("gmap_id"):
                continue
            best_by_family = {}
            for official in nearby_official(google, state, records, by_state_grid):
                score = score_match(google, official, max_distance)
                if not score:
                    continue
                family = official.get("service_family", "")
                current = best_by_family.get(family)
                rank = (score["status"] == "confirmed_official_candidate", score["score"])
                if current is None or rank > current[0]:
                    best_by_family[family] = (rank, official, score)
            for family, (_, official, score) in best_by_family.items():
                row = output_row(partition, line_number, google, official, score)
                key = (row["gmap_id"], family)
                current = best_by_google_family.get(key)
                rank = (
                    row["official_match_status"] == "confirmed_official_candidate",
                    float(row["official_link_score"]),
                )
                if current is None or rank > current[0]:
                    best_by_google_family[key] = (rank, row)

    rows = [item[1] for item in best_by_google_family.values()]
    official_google_ids = defaultdict(set)
    for row in rows:
        if row["official_match_status"] == "confirmed_official_candidate":
            official_google_ids[row["official_record_id"]].add(row["gmap_id"])
    for row in rows:
        multiplicity = len(official_google_ids.get(row["official_record_id"], ()))
        row["official_match_multiplicity"] = multiplicity or ""
        if row["official_match_status"] == "confirmed_official_candidate" and multiplicity > 1:
            row["official_match_status"] = "needs_review"
            row["official_match_method"] = "ambiguous_many_google_to_one_official"
    rows.sort(key=lambda row: (row["service_family"], row["gmap_id"], row["official_record_id"]))
    with gzip.open(output_path, "wt", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    statuses = Counter(row["official_match_status"] for row in rows)
    families = Counter(row["service_family"] for row in rows)
    linked_official = {row["official_record_id"] for row in rows}
    summary = {
        "linkage_version": LINKAGE_VERSION,
        "partition": partition,
        "state": state,
        "metadata_records_seen": metadata_seen,
        "json_decode_errors": decode_errors,
        "official_facilities_eligible_for_matching": state_totals[state],
        "candidate_links": len(rows),
        "unique_google_places": len({row["gmap_id"] for row in rows}),
        "unique_official_facilities_linked": len(linked_official),
        "counts_by_status": dict(sorted(statuses.items())),
        "counts_by_service_family": dict(sorted(families.items())),
        "warning": "Automated candidate linkage; blank adjudicated_label is intentional.",
        "output": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def combine_outputs(paths, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with gzip.open(destination, "wt", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for path in paths:
            with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
                for row in csv.DictReader(source):
                    writer.writerow(row)
                    total += 1
    return total


def verify_partition_output(path, expected_rows):
    rows = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != OUTPUT_FIELDS:
            raise ValueError(f"Unexpected linkage schema in {path}")
        for _ in reader:
            rows += 1
    if rows != expected_rows:
        raise ValueError(
            f"Existing linkage row count differs from summary for {path}: "
            f"{rows} != {expected_rows}"
        )
    return rows


def main():
    args = parse_args()
    records, by_state_grid, state_totals = load_official(args.official)
    requested = None
    if args.partitions:
        requested = {value.strip() for value in args.partitions.split(",") if value.strip()}
    metadata_paths = sorted(args.metadata_dir.glob("meta-*.json.gz"))
    if requested is not None:
        metadata_paths = [path for path in metadata_paths if partition_name(path) in requested]
        missing = requested - {partition_name(path) for path in metadata_paths}
        if missing:
            raise SystemExit(f"Requested metadata partitions not found: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    output_paths = []
    for metadata_path in metadata_paths:
        partition = partition_name(metadata_path)
        output_path = args.output_dir / f"usgs-google-linkage-{partition}.csv.gz"
        state_summary = args.output_dir / f"usgs-google-linkage-{partition}-summary.json"
        if output_path.exists() and state_summary.exists() and not args.force:
            summary = json.loads(state_summary.read_text(encoding="utf-8"))
            if summary.get("linkage_version") != LINKAGE_VERSION:
                raise SystemExit(
                    f"Existing {partition} output uses "
                    f"{summary.get('linkage_version')!r}, not {LINKAGE_VERSION!r}. "
                    "Preserve the old files or rerun explicitly with --force."
                )
            verify_partition_output(output_path, summary.get("candidate_links", -1))
            summary["status"] = "verified_existing_partition"
        else:
            summary = process_partition(
                metadata_path, output_path, state_summary, records, by_state_grid,
                state_totals, args.max_distance_km,
            )
        summaries.append(summary)
        output_paths.append(output_path)
        print(json.dumps(summary, sort_keys=True), flush=True)

    combined_rows = None
    if args.combined_output:
        combined_rows = combine_outputs(output_paths, args.combined_output)
    aggregate_status = Counter()
    aggregate_family = Counter()
    for summary in summaries:
        aggregate_status.update(summary.get("counts_by_status", {}))
        aggregate_family.update(summary.get("counts_by_service_family", {}))
    combined_summary = {
        "linkage_version": LINKAGE_VERSION,
        "partitions": len(summaries),
        "metadata_records_seen": sum(row.get("metadata_records_seen", 0) for row in summaries),
        "candidate_links": sum(row.get("candidate_links", 0) for row in summaries),
        "combined_output_rows": combined_rows,
        "counts_by_status": dict(sorted(aggregate_status.items())),
        "counts_by_service_family": dict(sorted(aggregate_family.items())),
        "state_summaries": summaries,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(combined_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
