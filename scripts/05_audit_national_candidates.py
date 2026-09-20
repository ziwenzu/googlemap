#!/usr/bin/env python3
"""Audit the nationwide rule-retrieval table without reading restricted reviews."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--places", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reported-source-places", type=int, default=4_963_111)
    return parser.parse_args()


def percent(numerator, denominator):
    if not denominator:
        return "n/a"
    return f"{100 * numerator / denominator:.2f}%"


def markdown_table(headers, rows):
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def main():
    args = parse_args()
    if not args.places.exists():
        raise SystemExit(f"Missing candidate table: {args.places}")

    rows = 0
    missing_ids = 0
    missing_names = 0
    missing_coordinates = 0
    address_only_rows = 0
    multi_family_rows = 0
    blank_manual_labels = 0
    ids = set()
    address_only_ids = set()
    partitions = Counter()
    families = Counter()
    family_ids = defaultdict(set)
    priorities = Counter()
    match_fields = Counter()
    statuses = Counter()
    id_partitions = defaultdict(set)
    id_signatures = defaultdict(set)

    with args.places.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            gmap_id = row.get("gmap_id", "").strip()
            partition = row.get("source_partition", "").strip()
            family = row.get("service_family", "").strip() or "missing"
            priority = row.get("match_priority", "").strip() or "missing"
            fields = row.get("matched_fields", "").strip() or "missing"
            status = row.get("classification_status", "").strip() or "missing"

            partitions[partition or "missing"] += 1
            families[family] += 1
            priorities[priority] += 1
            match_fields[fields] += 1
            statuses[status] += 1
            blank_manual_labels += not row.get("manual_label", "").strip()
            missing_names += not row.get("name", "").strip()
            missing_coordinates += not (
                row.get("latitude", "").strip() and row.get("longitude", "").strip()
            )
            multi_family_rows += ";" in row.get("all_service_families", "")

            if row.get("address_only_match", "").strip().lower() == "yes":
                address_only_rows += 1
                if gmap_id:
                    address_only_ids.add(gmap_id)

            if not gmap_id:
                missing_ids += 1
                continue
            ids.add(gmap_id)
            family_ids[family].add(gmap_id)
            id_partitions[gmap_id].add(partition)
            id_signatures[gmap_id].add(
                (
                    row.get("name", "").strip(),
                    row.get("latitude", "").strip(),
                    row.get("longitude", "").strip(),
                    family,
                )
            )

    # Count multiplicity directly from a second compact pass so repeated rows within
    # the same partition are not lost when partition names are stored as a set.
    id_counts = Counter()
    with args.places.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("gmap_id", "").strip():
                id_counts[row["gmap_id"].strip()] += 1
    duplicate_ids = {gmap_id for gmap_id, count in id_counts.items() if count > 1}
    duplicate_excess_rows = sum(count - 1 for count in id_counts.values() if count > 1)
    cross_partition_duplicates = sum(
        1 for gmap_id in duplicate_ids if len(id_partitions[gmap_id]) > 1
    )
    conflicting_duplicate_ids = sum(
        1 for gmap_id in duplicate_ids if len(id_signatures[gmap_id]) > 1
    )

    family_rows = []
    for family, count in sorted(families.items(), key=lambda item: (-item[1], item[0])):
        family_rows.append((family, f"{count:,}", f"{len(family_ids[family]):,}"))

    priority_rows = [
        (priority, f"{count:,}", percent(count, rows))
        for priority, count in sorted(priorities.items())
    ]
    match_rows = [
        (fields, f"{count:,}", percent(count, rows))
        for fields, count in sorted(match_fields.items(), key=lambda item: (-item[1], item[0]))
    ]

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report = f"""# Nationwide public-sector candidate retrieval audit

Generated at `{generated_at}` from `{args.places}`.

## Interpretation

This is a **high-recall rule-retrieval result, not a validated public-sector
facility frame**. Rows remain `rule_candidate` until manual adjudication and
official-roster checks establish precision and recall. Counts below must not be
used as estimates of government-office prevalence or public-service quality.

## Coverage and integrity

| Metric | Value |
| --- | ---: |
| Source partitions represented | {len(partitions):,} |
| Candidate rows | {rows:,} |
| Unique nonmissing `gmap_id` values | {len(ids):,} |
| Candidate rows / UCSD reported places | {percent(rows, args.reported_source_places)} |
| Missing `gmap_id` rows | {missing_ids:,} |
| Missing-name rows | {missing_names:,} |
| Missing-coordinate rows | {missing_coordinates:,} |
| Duplicate `gmap_id` values | {len(duplicate_ids):,} |
| Duplicate excess rows | {duplicate_excess_rows:,} |
| Duplicate IDs spanning partitions | {cross_partition_duplicates:,} |
| Duplicate IDs with conflicting name/coordinates/family | {conflicting_duplicate_ids:,} |
| Address-only match rows | {address_only_rows:,} ({percent(address_only_rows, rows)}) |
| Unique address-only IDs | {len(address_only_ids):,} |
| Rows matching multiple service families | {multi_family_rows:,} ({percent(multi_family_rows, rows)}) |
| Rows without a manual label | {blank_manual_labels:,} ({percent(blank_manual_labels, rows)}) |

The source-place denominator is the dataset publisher's reported total of
{args.reported_source_places:,}; it is not a locally reconstructed denominator.
The archive download itself is independently checked by byte count, gzip
integrity, SHA-256, and the download manifest.

## Primary service-family retrieval

{markdown_table(["Primary family", "Rows", "Unique IDs"], family_rows)}

## Retrieval priority

{markdown_table(["Priority", "Rows", "Share"], priority_rows)}

## Primary matched fields

{markdown_table(["Matched fields", "Rows", "Share"], match_rows)}

## Classification state

{markdown_table(["Status", "Rows", "Share"], [(status, f"{count:,}", percent(count, rows)) for status, count in sorted(statuses.items())])}

## Required next validation steps

1. Adjudicate all core and ambiguous candidates with independent double coding.
2. Treat address-only matches as manual-review-only until their precision is known.
3. Draw a stratified probability sample of rule negatives; candidate-only review
   cannot estimate recall.
4. Match covered service families to official rosters and retain match quality,
   candidate probability, adjudicated label, and exclusion reason.
5. Freeze the validated `gmap_id` list before extracting nationwide reviews.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_rows": rows,
                "unique_gmap_ids": len(ids),
                "source_partitions": len(partitions),
                "duplicate_gmap_ids": len(duplicate_ids),
                "address_only_rows": address_only_rows,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
