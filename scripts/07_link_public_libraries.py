#!/usr/bin/env python3
"""Link Google Maps public-library candidates to the official IMLS outlet frame.

The output is a candidate linkage table, not a final adjudication.  A strong
official link raises confidence that the Google place represents a public
library outlet.  Failure to link does not establish that a place is private or
invalid because names, addresses, coordinates, and operating status can change.
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

MATCH_FIELDS = [
    "official_match_status",
    "official_match_score",
    "official_match_method",
    "official_match_multiplicity",
    "official_roster_id",
    "official_record_id",
    "official_entity_name",
    "official_parent_entity_name",
    "official_address",
    "official_city",
    "official_state",
    "official_zip",
    "official_latitude",
    "official_longitude",
    "official_active_status",
    "name_similarity",
    "address_similarity",
    "distance_km",
    "adjudicated_label",
    "exclusion_reason",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--places", required=True, type=Path)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def open_text(path, mode="rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode.replace("t", ""), newline="", encoding="utf-8")


def normalize_text(value):
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.upper().replace("&", " AND ")
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def normalize_name(value):
    tokens = [
        token for token in normalize_text(value).split()
        if token not in {"THE", "PUBLIC", "LIBRARY", "BRANCH"}
    ]
    return " ".join(tokens) or normalize_text(value)


def similarity(left, right, *, name=False):
    left_norm = normalize_name(left) if name else normalize_text(left)
    right_norm = normalize_name(right) if name else normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(sequence, token_score)


def address_similarity(candidate_address, official_address):
    candidate = normalize_text(candidate_address)
    official = normalize_text(official_address)
    if not candidate or not official:
        return 0.0
    if official in candidate:
        return 1.0
    candidate_tokens = set(candidate.split())
    official_tokens = set(official.split())
    containment = len(candidate_tokens & official_tokens) / len(official_tokens)
    candidate_number = re.search(r"\b\d+\b", candidate)
    official_number = re.search(r"\b\d+\b", official)
    if candidate_number and official_number and candidate_number.group() != official_number.group():
        containment *= 0.65
    return max(containment, similarity(candidate, official))


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


def extract_zip(value):
    matches = re.findall(r"\b(\d{5})(?:-\d{4})?\b", str(value or ""))
    return matches[-1] if matches else ""


def candidate_state(row):
    address_match = re.search(r",\s*([A-Z]{2})\s+\d{5}(?:-\d{4})?\b", str(row.get("address", "")))
    if address_match:
        return address_match.group(1)
    partition = str(row.get("source_partition", "")).replace("_", " ")
    return STATE_ABBREVIATIONS.get(partition, partition if len(partition) == 2 else "")


def grid_cell(latitude, longitude, size=0.05):
    latitude = as_float(latitude)
    longitude = as_float(longitude)
    if latitude is None or longitude is None:
        return None
    return math.floor(latitude / size), math.floor(longitude / size)


def load_official(path):
    records = []
    by_state_zip = defaultdict(list)
    by_state_name = defaultdict(list)
    by_grid = defaultdict(list)
    with open_text(path) as handle:
        for row in csv.DictReader(handle):
            index = len(records)
            records.append(row)
            state = row.get("state", "")
            zip_code = row.get("zip", "")[:5]
            by_state_zip[(state, zip_code)].append(index)
            by_state_name[(state, normalize_name(row.get("entity_name", "")))].append(index)
            cell = grid_cell(row.get("latitude"), row.get("longitude"))
            if cell:
                by_grid[cell].append(index)
    return records, by_state_zip, by_state_name, by_grid


def candidate_pool(row, records, by_state_zip, by_state_name, by_grid):
    state = candidate_state(row)
    zip_code = extract_zip(row.get("address"))
    indices = set(by_state_zip.get((state, zip_code), ())) if zip_code else set()
    indices.update(by_state_name.get((state, normalize_name(row.get("name"))), ()))
    cell = grid_cell(row.get("latitude"), row.get("longitude"))
    if cell:
        for lat_offset in (-1, 0, 1):
            for lon_offset in (-1, 0, 1):
                for index in by_grid.get((cell[0] + lat_offset, cell[1] + lon_offset), ()):
                    if not state or records[index].get("state") == state:
                        indices.add(index)
    return [records[index] for index in indices]


def score_match(candidate, official):
    name_score = similarity(candidate.get("name"), official.get("entity_name"), name=True)
    parent_score = similarity(candidate.get("name"), official.get("parent_entity_name"), name=True)
    name_score = max(name_score, parent_score * 0.92)
    address_score = address_similarity(candidate.get("address"), official.get("address_line1"))
    distance = haversine_km(
        candidate.get("latitude"), candidate.get("longitude"),
        official.get("latitude"), official.get("longitude"),
    )
    zip_match = extract_zip(candidate.get("address")) == official.get("zip", "")[:5]
    geo_score = 0.0 if distance is None else max(0.0, 1.0 - distance / 5.0)
    total = 0.55 * name_score + 0.25 * address_score + 0.20 * geo_score

    if name_score >= 0.88 and (
        (zip_match and (distance is None or distance <= 3.0))
        or (distance is not None and distance <= 3.0)
    ):
        status, method = "confirmed_official_candidate", "name_plus_zip_or_geo"
    elif address_score >= 0.90 and distance is not None and distance <= 1.0 and name_score >= 0.55:
        status, method = "confirmed_official_candidate", "address_geo_name"
    elif distance is not None and distance <= 0.20 and name_score >= 0.65:
        status, method = "confirmed_official_candidate", "near_exact_geo_plus_name"
    elif (name_score >= 0.68 and (zip_match or (distance is not None and distance <= 5.0))) or (
        address_score >= 0.75 and distance is not None and distance <= 3.0
    ):
        status, method = "needs_review", "partial_official_link"
    else:
        status, method = "not_linked", "insufficient_agreement"
    return {
        "status": status,
        "method": method,
        "score": total,
        "name_similarity": name_score,
        "address_similarity": address_score,
        "distance_km": distance,
    }


def best_match(candidate, pool):
    scored = [(score_match(candidate, official), official) for official in pool]
    if not scored:
        return None, None
    status_order = {"confirmed_official_candidate": 2, "needs_review": 1, "not_linked": 0}
    return max(scored, key=lambda item: (status_order[item[0]["status"]], item[0]["score"]))


def match_fields(score, official, multiplicity=0):
    if not score or not official or score["status"] == "not_linked":
        return {
            "official_match_status": "not_linked",
            "official_match_score": "",
            "official_match_method": "no_plausible_official_record",
            "official_match_multiplicity": "",
            **{field: "" for field in MATCH_FIELDS[4:]},
        }
    status = score["status"]
    method = score["method"]
    if multiplicity > 1:
        status = "needs_review"
        method = "ambiguous_many_to_one_official_record"
    address = " ".join(
        value for value in (official.get("address_line1", ""), official.get("address_line2", "")) if value
    )
    return {
        "official_match_status": status,
        "official_match_score": f"{score['score']:.4f}",
        "official_match_method": method,
        "official_match_multiplicity": multiplicity,
        "official_roster_id": official.get("roster_id", ""),
        "official_record_id": official.get("source_record_id", ""),
        "official_entity_name": official.get("entity_name", ""),
        "official_parent_entity_name": official.get("parent_entity_name", ""),
        "official_address": address,
        "official_city": official.get("city", ""),
        "official_state": official.get("state", ""),
        "official_zip": official.get("zip", ""),
        "official_latitude": official.get("latitude", ""),
        "official_longitude": official.get("longitude", ""),
        "official_active_status": official.get("active_status", ""),
        "name_similarity": f"{score['name_similarity']:.4f}",
        "address_similarity": f"{score['address_similarity']:.4f}",
        "distance_km": "" if score["distance_km"] is None else f"{score['distance_km']:.3f}",
        "adjudicated_label": "",
        "exclusion_reason": "",
    }


def main():
    args = parse_args()
    records, by_state_zip, by_state_name, by_grid = load_official(args.official)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    candidate_rows = 0
    duplicate_rows = 0
    seen = set()
    linked = []
    with args.places.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        output_fields = reader.fieldnames + MATCH_FIELDS
        for row in reader:
            families = set(filter(None, row.get("all_service_families", "").split(";")))
            if row.get("service_family") != "public_library" and "public_library" not in families:
                continue
            candidate_rows += 1
            gmap_id = row.get("gmap_id", "")
            identity = gmap_id or (row.get("source_partition"), row.get("source_line"))
            if identity in seen:
                duplicate_rows += 1
                continue
            seen.add(identity)
            pool = candidate_pool(row, records, by_state_zip, by_state_name, by_grid)
            score, official = best_match(row, pool)
            linked.append((row, score, official))

    official_candidate_counts = Counter(
        official.get("source_record_id", "")
        for _, score, official in linked
        if score and official and score["status"] != "not_linked"
    )
    status_counts = Counter()
    linked_official_ids = set()
    confirmed_official_ids = set()
    with open_text(args.output, "wt") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_fields)
        writer.writeheader()
        for row, score, official in linked:
            official_id = official.get("source_record_id", "") if official else ""
            multiplicity = official_candidate_counts.get(official_id, 0)
            fields = match_fields(score, official, multiplicity)
            status_counts[fields["official_match_status"]] += 1
            if fields["official_match_status"] != "not_linked":
                linked_official_ids.add(official_id)
            if fields["official_match_status"] == "confirmed_official_candidate":
                confirmed_official_ids.add(official_id)
            writer.writerow({**row, **fields})

    summary = {
        "official_outlet_rows": len(records),
        "google_candidate_rows_before_deduplication": candidate_rows,
        "duplicate_google_candidate_rows_removed": duplicate_rows,
        "unique_google_candidates": len(seen),
        "match_status_counts": dict(sorted(status_counts.items())),
        "unique_official_outlets_linked": len(linked_official_ids),
        "unique_official_outlets_with_unambiguous_confirmed_link": len(confirmed_official_ids),
        "official_outlets_with_multiple_google_candidates": sum(
            count > 1 for count in official_candidate_counts.values()
        ),
        "interpretation": {
            "confirmed_official_candidate": "Strong automated link to an IMLS public-library outlet; retain as a candidate pending adjudication policy.",
            "needs_review": "Plausible partial link requiring manual review.",
            "not_linked": "No sufficiently agreeing IMLS record; this is not evidence that the Google place is private.",
        },
        "output": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
