#!/usr/bin/env python3
"""Create a privacy-preserving QA report for a state pilot.

The report contains aggregate diagnostics only. It never writes review text,
reviewer hashes, or row-level review records.
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--places", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-metadata", type=Path)
    parser.add_argument("--raw-reviews", type=Path)
    return parser.parse_args()


def pct(numerator, denominator):
    if not denominator:
        return "NA"
    return f"{100 * numerator / denominator:.1f}%"


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def fmt_number(value):
    if value is None:
        return "NA"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.2f}"
    return f"{int(value):,}"


def utc_date(milliseconds):
    return dt.datetime.fromtimestamp(
        float(milliseconds) / 1000.0, tz=dt.timezone.utc
    ).date()


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_places(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["gmap_id"]].append(row)
    return rows, by_id


def inspect_reviews(path):
    review_counts = Counter()
    ratings = Counter()
    years = Counter()
    review_keys = Counter()
    row_signatures = Counter()
    unique_reviewers = set()
    text_count = 0
    response_count = 0
    response_before_review = 0
    deduplicated_ratings = Counter()
    invalid_rating = 0
    invalid_time = 0
    minimum_time = None
    maximum_time = None
    rows = 0

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows += 1
            record = json.loads(line)
            gmap_id = record.get("gmap_id")
            review_counts[gmap_id] += 1
            review_key = record.get("review_key")
            first_review_key = review_keys[review_key] == 0
            review_keys[review_key] += 1
            signature_record = {key: value for key, value in record.items() if key != "source_line"}
            row_signatures[json.dumps(signature_record, sort_keys=True, separators=(",", ":"))] += 1
            if record.get("user_hash"):
                unique_reviewers.add(record["user_hash"])
            if record.get("text"):
                text_count += 1
            if record.get("has_response"):
                response_count += 1
                try:
                    if float(record.get("response_time")) < float(record.get("time")):
                        response_before_review += 1
                except (TypeError, ValueError):
                    pass

            rating = record.get("rating")
            if isinstance(rating, (int, float)) and int(rating) == rating and 1 <= rating <= 5:
                ratings[int(rating)] += 1
                if first_review_key:
                    deduplicated_ratings[int(rating)] += 1
            else:
                invalid_rating += 1

            try:
                timestamp = utc_date(record.get("time"))
            except (TypeError, ValueError, OverflowError, OSError):
                invalid_time += 1
                continue
            minimum_time = timestamp if minimum_time is None else min(minimum_time, timestamp)
            maximum_time = timestamp if maximum_time is None else max(maximum_time, timestamp)
            years[timestamp.year] += 1

    return {
        "rows": rows,
        "review_counts": review_counts,
        "ratings": ratings,
        "years": years,
        "review_keys": review_keys,
        "row_signatures": row_signatures,
        "unique_reviewers": unique_reviewers,
        "text_count": text_count,
        "response_count": response_count,
        "response_before_review": response_before_review,
        "deduplicated_ratings": deduplicated_ratings,
        "invalid_rating": invalid_rating,
        "invalid_time": invalid_time,
        "minimum_time": minimum_time,
        "maximum_time": maximum_time,
    }


def inspect_raw_archive(path, artifact_type):
    seen_lines = set()
    seen_review_fingerprints = set()
    gmap_ids = set()
    rows = decode_errors = exact_duplicate_rows = fingerprint_duplicate_rows = 0
    missing_rating = invalid_time = 0
    minimum_time = maximum_time = None

    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
            rows += 1
            line_digest = hashlib.blake2b(raw_line.rstrip(b"\r\n"), digest_size=16).digest()
            if line_digest in seen_lines:
                exact_duplicate_rows += 1
            else:
                seen_lines.add(line_digest)
            try:
                record = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                decode_errors += 1
                continue
            if record.get("gmap_id"):
                gmap_ids.add(record["gmap_id"])
            if artifact_type != "reviews":
                continue
            if record.get("rating") is None:
                missing_rating += 1
            normalized_text = " ".join(str(record.get("text") or "").split())
            fingerprint = "|".join(
                [
                    str(record.get("gmap_id") or ""),
                    str(record.get("user_id") or ""),
                    str(record.get("time") or ""),
                    str(record.get("rating") or ""),
                    normalized_text,
                ]
            )
            fingerprint_digest = hashlib.blake2b(
                fingerprint.encode("utf-8"), digest_size=16
            ).digest()
            if fingerprint_digest in seen_review_fingerprints:
                fingerprint_duplicate_rows += 1
            else:
                seen_review_fingerprints.add(fingerprint_digest)
            try:
                timestamp = utc_date(record.get("time"))
            except (TypeError, ValueError, OverflowError, OSError):
                invalid_time += 1
                continue
            minimum_time = timestamp if minimum_time is None else min(minimum_time, timestamp)
            maximum_time = timestamp if maximum_time is None else max(maximum_time, timestamp)

    return {
        "rows": rows,
        "decode_errors": decode_errors,
        "gmap_ids": gmap_ids,
        "exact_duplicate_rows": exact_duplicate_rows,
        "fingerprint_duplicate_rows": fingerprint_duplicate_rows,
        "missing_rating": missing_rating,
        "invalid_time": invalid_time,
        "minimum_time": minimum_time,
        "maximum_time": maximum_time,
    }


def inspect_panel(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["n_reviews"] = int(row["n_reviews"])
            rows.append(row)
    return rows


def main():
    args = parse_args()
    place_rows, places_by_id = load_places(args.places)
    review = inspect_reviews(args.reviews)
    panel = inspect_panel(args.panel)
    raw_metadata = (
        inspect_raw_archive(args.raw_metadata, "metadata") if args.raw_metadata else None
    )
    raw_reviews = inspect_raw_archive(args.raw_reviews, "reviews") if args.raw_reviews else None

    unique_ids = sorted(places_by_id)
    canonical_place = {gmap_id: rows[0] for gmap_id, rows in places_by_id.items()}
    duplicate_ids = {gmap_id: rows for gmap_id, rows in places_by_id.items() if len(rows) > 1}
    conflicting_duplicates = 0
    comparison_fields = ("name", "address", "service_family", "latitude", "longitude")
    for rows in duplicate_ids.values():
        signatures = {tuple(row.get(field, "") for field in comparison_fields) for row in rows}
        conflicting_duplicates += len(signatures) > 1

    places_by_family = Counter(
        place.get("service_family", "unclassified") for place in canonical_place.values()
    )
    reviews_by_family = Counter()
    places_with_reviews_by_family = Counter()
    for gmap_id, count in review["review_counts"].items():
        family = canonical_place.get(gmap_id, {}).get("service_family", "unmatched")
        reviews_by_family[family] += count
        if count:
            places_with_reviews_by_family[family] += 1

    lifetime_counts = [review["review_counts"].get(gmap_id, 0) for gmap_id in unique_ids]
    cells = [row["n_reviews"] for row in panel]
    panel_analysis_reviews = sum(cells)
    panel_source_rows = sum(int(row.get("n_source_rows", row["n_reviews"])) for row in panel)
    places_with_reviews = sum(count > 0 for count in lifetime_counts)
    exact_duplicate_reviews = sum(count - 1 for count in review["review_keys"].values() if count > 1)
    exact_duplicate_rows = sum(count - 1 for count in review["row_signatures"].values() if count > 1)

    metadata_match = Counter()
    for gmap_id in unique_ids:
        raw_value = canonical_place[gmap_id].get("num_of_reviews", "")
        try:
            metadata_count = int(float(raw_value))
        except (TypeError, ValueError):
            metadata_match["missing"] += 1
            continue
        extracted = review["review_counts"].get(gmap_id, 0)
        if extracted == metadata_count:
            metadata_match["equal"] += 1
        elif extracted > metadata_count:
            metadata_match["extracted_gt"] += 1
        else:
            metadata_match["extracted_lt"] += 1

    rating_total = sum(review["ratings"].values())
    rating_mean = (
        sum(star * count for star, count in review["ratings"].items()) / rating_total
        if rating_total
        else None
    )
    deduplicated_rating_total = sum(review["deduplicated_ratings"].values())
    deduplicated_rating_mean = (
        sum(star * count for star, count in review["deduplicated_ratings"].items())
        / deduplicated_rating_total
        if deduplicated_rating_total
        else None
    )
    family_rows = []
    for family in sorted(places_by_family):
        family_rows.append(
            (
                family,
                f"{places_by_family[family]:,}",
                f"{places_with_reviews_by_family[family]:,}",
                f"{reviews_by_family[family]:,}",
            )
        )

    retrieval_channels = Counter(
        place.get("matched_fields", "not recorded") or "not recorded"
        for place in canonical_place.values()
    )
    retrieval_rows = [
        (fields, f"{count:,}", pct(count, len(unique_ids)))
        for fields, count in sorted(retrieval_channels.items())
    ]
    address_only_count = sum(
        place.get("address_only_match", "").casefold() == "yes"
        for place in canonical_place.values()
    )

    rating_rows = [
        (f"{star} star", f"{review['ratings'][star]:,}", pct(review["ratings"][star], rating_total))
        for star in range(1, 6)
    ]
    year_rows = [
        (str(year), f"{count:,}", pct(count, review["rows"]))
        for year, count in sorted(review["years"].items())
    ]
    duplicate_rows = []
    for gmap_id, rows in sorted(duplicate_ids.items()):
        duplicate_rows.append(
            (
                gmap_id,
                len(rows),
                ", ".join(row.get("source_line", "") for row in rows),
                "yes" if len({tuple(row.get(field, "") for field in comparison_fields) for row in rows}) > 1 else "no",
            )
        )

    if raw_metadata and raw_reviews:
        archive_section = f"""## Raw archive audit

{markdown_table(
    ["Diagnostic", "Metadata", "Reviews"],
    [
        ("Source rows", f"{raw_metadata['rows']:,}", f"{raw_reviews['rows']:,}"),
        ("JSON decode errors", f"{raw_metadata['decode_errors']:,}", f"{raw_reviews['decode_errors']:,}"),
        ("Unique place IDs", f"{len(raw_metadata['gmap_ids']):,}", f"{len(raw_reviews['gmap_ids']):,}"),
        ("Exact duplicate excess rows", f"{raw_metadata['exact_duplicate_rows']:,}", f"{raw_reviews['exact_duplicate_rows']:,}"),
        ("Duplicate review-fingerprint excess", "NA", f"{raw_reviews['fingerprint_duplicate_rows']:,}"),
        ("Missing ratings", "NA", f"{raw_reviews['missing_rating']:,}"),
        ("Invalid timestamps", "NA", f"{raw_reviews['invalid_time']:,}"),
    ],
)}

The raw reviews span **{raw_reviews['minimum_time']} to {raw_reviews['maximum_time']} UTC**. Review fingerprints use place ID, user ID, timestamp, rating, and normalized text; they are analysis keys, not platform review IDs. There are {len(raw_reviews['gmap_ids'] - raw_metadata['gmap_ids']):,} review place IDs absent from the metadata archive. Exact duplicates are extraction-quality errors and remain distinct from strategic fake reviews.
"""
    else:
        archive_section = "## Raw archive audit\n\nNot run; pass both raw archive paths to the QA script."

    report = f"""# {args.state} pilot quality-assurance report

Generated from the filtered pilot files. This report contains no review text or reviewer identifiers.

## Executive result

The pipeline completed and the source row totals previously matched the UCSD state files, but this is **not yet an analysis-ready governance sample**. The rule-based retrieval stage yielded {len(place_rows):,} candidate rows representing {len(unique_ids):,} unique Google place IDs. None has yet received a frozen manual include/exclude label, so precision, recall, and substantive estimates remain unknown. The event-only panel is also extremely sparse: {pct(sum(value < 5 for value in cells), len(cells))} of observed place-month cells contain fewer than five reviews. Monthly office rankings would therefore be largely noise without a complete exposure grid, external denominators, and substantial pooling.

## File and key integrity

{markdown_table(
    ["Diagnostic", "Value"],
    [
        ("Candidate rows", f"{len(place_rows):,}"),
        ("Unique candidate place IDs", f"{len(unique_ids):,}"),
        ("Duplicate candidate IDs", f"{len(duplicate_ids):,}"),
        ("Duplicate IDs with conflicting core metadata", f"{conflicting_duplicates:,}"),
        ("Filtered review rows", f"{review['rows']:,}"),
        ("Distinct review keys", f"{len(review['review_keys']):,}"),
        ("Repeated rows by review-key excess", f"{exact_duplicate_reviews:,}"),
        ("Exact duplicate rows, excluding source line", f"{exact_duplicate_rows:,}"),
        ("Source rows represented in panel", f"{panel_source_rows:,}"),
        ("Analysis reviews after review-key deduplication", f"{panel_analysis_reviews:,}"),
        ("Invalid ratings", f"{review['invalid_rating']:,}"),
        ("Invalid timestamps", f"{review['invalid_time']:,}"),
        ("Event-only place-month rows", f"{len(panel):,}"),
    ],
)}

The two candidate duplicates are source-level duplicate metadata records rather than two distinct facilities if the conflict flag below is `no`. They must be collapsed only after preserving provenance. Review extraction already uses a set of place IDs, so those metadata duplicates did not multiply review rows. The filtered review archive also contains {exact_duplicate_rows:,} exact duplicate excess rows. They remain in the immutable-derived stream for provenance; the descriptive panel applies the declared `review-key` deduplication rule and retains source-row counts for audit.

{markdown_table(["gmap_id", "candidate rows", "source lines", "core metadata conflict"], duplicate_rows) if duplicate_rows else "No duplicate candidate IDs were found."}

{archive_section}

## Candidate composition

The following table is a **retrieval result**, not a validated government-facility universe.

An exploratory name-level audit has already confirmed false positives and family
misses; see [`Vermont-retrieval-audit-notes.md`](Vermont-retrieval-audit-notes.md).

{markdown_table(["Service family", "Candidate places", "With reviews", "Extracted reviews"], family_rows)}

### Retrieval channels

{markdown_table(["Matched fields", "Candidate places", "Share"], retrieval_rows)}

There are {address_only_count:,} address-only candidate places. These are routed to
low-confidence manual review because an agency phrase in an address can retrieve a
private or unrelated business. Field provenance improves auditing but does not
replace probability-based precision and recall validation.

## Coverage and sparsity

{markdown_table(
    ["Diagnostic", "Value"],
    [
        ("Candidate places with at least one extracted review", f"{places_with_reviews:,} / {len(unique_ids):,} ({pct(places_with_reviews, len(unique_ids))})"),
        ("Places with at least 5 lifetime reviews", f"{sum(value >= 5 for value in lifetime_counts):,} / {len(unique_ids):,} ({pct(sum(value >= 5 for value in lifetime_counts), len(unique_ids))})"),
        ("Places with at least 10 lifetime reviews", f"{sum(value >= 10 for value in lifetime_counts):,} / {len(unique_ids):,} ({pct(sum(value >= 10 for value in lifetime_counts), len(unique_ids))})"),
        ("Places with at least 20 lifetime reviews", f"{sum(value >= 20 for value in lifetime_counts):,} / {len(unique_ids):,} ({pct(sum(value >= 20 for value in lifetime_counts), len(unique_ids))})"),
        ("Lifetime reviews per candidate: minimum", fmt_number(min(lifetime_counts) if lifetime_counts else None)),
        ("Lifetime reviews per candidate: median", fmt_number(quantile(lifetime_counts, 0.50))),
        ("Lifetime reviews per candidate: 90th percentile", fmt_number(quantile(lifetime_counts, 0.90))),
        ("Lifetime reviews per candidate: maximum", fmt_number(max(lifetime_counts) if lifetime_counts else None)),
        ("Observed place-month cells with exactly 1 review", f"{sum(value == 1 for value in cells):,} / {len(cells):,} ({pct(sum(value == 1 for value in cells), len(cells))})"),
        ("Observed place-month cells with fewer than 5 reviews", f"{sum(value < 5 for value in cells):,} / {len(cells):,} ({pct(sum(value < 5 for value in cells), len(cells))})"),
        ("Reviews per observed place-month: median", fmt_number(quantile(cells, 0.50))),
        ("Reviews per observed place-month: 90th percentile", fmt_number(quantile(cells, 0.90))),
        ("Reviews per observed place-month: maximum", fmt_number(max(cells) if cells else None)),
    ],
)}

The current panel contains only months with one or more retained reviews. It therefore cannot estimate a zero-review rate. A complete facility-month exposure grid requires a defensible listing/activity window or an external government-facility frame; treating all absent rows as zero would confuse pre-listing periods, closure, nonuse, nonposting, and deletion.

## Rating, text, and response diagnostics

The filtered reviews span **{review['minimum_time']} to {review['maximum_time']} UTC**. Their source-row mean rating is **{rating_mean:.3f}** stars; after review-key deduplication it is **{deduplicated_rating_mean:.3f}**. These are visible-review sentiment, not selection-corrected satisfaction estimates.

{markdown_table(["Rating", "Reviews", "Share"], rating_rows)}

{markdown_table(
    ["Diagnostic", "Value"],
    [
        ("Distinct reviewer HMACs", f"{len(review['unique_reviewers']):,}"),
        ("Reviews with text", f"{review['text_count']:,} / {review['rows']:,} ({pct(review['text_count'], review['rows'])})"),
        ("Reviews with owner response", f"{review['response_count']:,} / {review['rows']:,} ({pct(review['response_count'], review['rows'])})"),
        ("Owner responses timestamped before review", f"{review['response_before_review']:,} / {review['response_count']:,} ({pct(review['response_before_review'], review['response_count'])})"),
    ],
)}

### Review years

{markdown_table(["UTC year", "Reviews", "Share"], year_rows)}

## Metadata-count diagnostic

Metadata `num_of_reviews` and the extracted review archive agree for {metadata_match['equal']:,} of {len(unique_ids):,} unique candidate places. The extracted count is higher for {metadata_match['extracted_gt']:,}, lower for {metadata_match['extracted_lt']:,}, and unavailable for {metadata_match['missing']:,}. Disagreement is not automatically an error: metadata and review archives may reflect different crawl times, deletions, listing merges, or source coverage. It is a diagnostic for manual review, not a reconciliation rule.

## Go/no-go decision

**Engineering go; measurement hold.** The streaming and aggregation code works on the complete Vermont state archive. Do not estimate or publish a Vermont governance index yet. Advance only after:

1. double-coded manual adjudication of all core candidates, ambiguous cases, and a probability sample of retrieved negatives;
2. linkage to official facility rosters to estimate frame-specific recall;
3. a decision to use annual or multi-year windows unless a much denser state/service family supports finer time;
4. implementation of contamination and selection sensitivity analyses; and
5. acquisition of at least one external anchor (probability exit audit, administrative encounter denominator, or defensible probability survey) for point-identification claims.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(args.output), "candidate_places": len(unique_ids), "reviews": review["rows"]}, indent=2))


if __name__ == "__main__":
    main()
