#!/usr/bin/env python3
"""Stream UCSD metadata and retain rule-based government-place candidates."""

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def clean(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(clean(item) for item in value)
    if isinstance(value, dict):
        return " | ".join(f"{clean(k)} {clean(v)}" for k, v in value.items())
    return re.sub(r"\s+", " ", str(value)).strip()


def load_taxonomy(path):
    rules = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rules.append(
                {
                    **row,
                    "priority": int(row["priority"]),
                    "include": re.compile(row["include_pattern"], re.I),
                    "exclude": re.compile(row["exclude_pattern"], re.I)
                    if row["exclude_pattern"]
                    else None,
                }
            )
    return sorted(rules, key=lambda item: item["priority"])


def candidate_matches(record, rules):
    fields = {
        "name": clean(record.get("name")),
        "categories": clean(record.get("category")),
        "description": clean(record.get("description")),
        "address": clean(record.get("address")),
    }
    searchable = " || ".join(fields.values())
    matches = []
    for rule in rules:
        field_hits = {
            field: hit.group(0)
            for field, value in fields.items()
            if (hit := rule["include"].search(value))
        }
        if not field_hits:
            continue
        exclusion = rule["exclude"].search(searchable) if rule["exclude"] else None
        if exclusion:
            continue
        matches.append((rule, field_hits))
    return matches


def main():
    args = parse_args()
    if not args.metadata.exists():
        raise SystemExit(f"Missing metadata archive: {args.metadata}")
    rules = load_taxonomy(args.taxonomy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gmap_id",
        "name",
        "address",
        "latitude",
        "longitude",
        "categories",
        "avg_rating",
        "num_of_reviews",
        "service_family",
        "all_service_families",
        "government_level_hint",
        "match_priority",
        "matched_text",
        "matched_fields",
        "address_only_match",
        "classification_status",
        "manual_label",
        "manual_exclusion_reason",
        "source_line",
    ]
    seen = 0
    retained = 0
    decode_errors = 0
    with gzip.open(args.metadata, "rt", encoding="utf-8", errors="replace") as source, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for line_number, line in enumerate(source, start=1):
            seen += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            matches = candidate_matches(record, rules)
            if not matches:
                continue
            primary_rule, primary_hits = matches[0]
            families = sorted({rule["service_family"] for rule, _ in matches})
            matched_fields = sorted(primary_hits)
            writer.writerow(
                {
                    "gmap_id": clean(record.get("gmap_id")),
                    "name": clean(record.get("name")),
                    "address": clean(record.get("address")),
                    "latitude": record.get("latitude", ""),
                    "longitude": record.get("longitude", ""),
                    "categories": clean(record.get("category")),
                    "avg_rating": record.get("avg_rating", ""),
                    "num_of_reviews": record.get("num_of_reviews", ""),
                    "service_family": primary_rule["service_family"],
                    "all_service_families": ";".join(families),
                    "government_level_hint": primary_rule["level_hint"],
                    "match_priority": primary_rule["priority"],
                    "matched_text": " | ".join(
                        f"{field}: {primary_hits[field]}" for field in matched_fields
                    ),
                    "matched_fields": ";".join(matched_fields),
                    "address_only_match": "yes" if matched_fields == ["address"] else "no",
                    "classification_status": "rule_candidate",
                    "manual_label": "",
                    "manual_exclusion_reason": "",
                    "source_line": line_number,
                }
            )
            retained += 1
    print(
        json.dumps(
            {
                "metadata": str(args.metadata),
                "records_seen": seen,
                "candidates_retained": retained,
                "json_decode_errors": decode_errors,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if seen == 0:
        print("Warning: metadata archive contained no records", file=sys.stderr)


if __name__ == "__main__":
    main()
