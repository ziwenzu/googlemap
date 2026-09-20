#!/usr/bin/env python3
"""Create a descriptive place-month panel from the filtered review stream."""

import argparse
import csv
import datetime as dt
import gzip
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--places", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--deduplicate",
        choices=("none", "review-key"),
        default="review-key",
        help="The analysis panel defaults to one row per stable review fingerprint.",
    )
    return parser.parse_args()


def month_from_milliseconds(value):
    if value is None:
        return None
    try:
        timestamp = float(value) / 1000.0
        return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).strftime("%Y-%m")
    except (TypeError, ValueError, OverflowError):
        return None


def load_places(path):
    places = {}
    comparison_fields = ("name", "address", "service_family", "latitude", "longitude")
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gmap_id = row["gmap_id"]
            if gmap_id in places:
                prior = places[gmap_id]
                if any(prior.get(field, "") != row.get(field, "") for field in comparison_fields):
                    raise ValueError(f"Conflicting duplicate place metadata for {gmap_id}")
                continue
            places[gmap_id] = row
    return places


def main():
    args = parse_args()
    places = load_places(args.places)
    aggregates = defaultdict(
        lambda: {
            "n_source_rows": 0,
            "n_duplicate_rows": 0,
            "n_reviews": 0,
            "rating_sum": 0.0,
            "n_rating": 0,
            "n_1star": 0,
            "n_2star": 0,
            "n_3star": 0,
            "n_4star": 0,
            "n_5star": 0,
            "n_text": 0,
            "n_response": 0,
            "users": set(),
        }
    )
    seen_review_keys = set()
    source_rows = duplicate_rows = 0
    with gzip.open(args.reviews, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            month = month_from_milliseconds(record.get("time"))
            if month is None:
                continue
            key = (record.get("gmap_id"), month)
            bucket = aggregates[key]
            source_rows += 1
            bucket["n_source_rows"] += 1
            review_key = record.get("review_key")
            if args.deduplicate == "review-key" and review_key and review_key in seen_review_keys:
                duplicate_rows += 1
                bucket["n_duplicate_rows"] += 1
                continue
            if review_key:
                seen_review_keys.add(review_key)
            bucket["n_reviews"] += 1
            rating = record.get("rating")
            if isinstance(rating, (int, float)) and 1 <= rating <= 5:
                rating_int = int(rating)
                bucket["rating_sum"] += float(rating)
                bucket["n_rating"] += 1
                bucket[f"n_{rating_int}star"] += 1
            if record.get("text"):
                bucket["n_text"] += 1
            if record.get("has_response"):
                bucket["n_response"] += 1
            if record.get("user_hash"):
                bucket["users"].add(record["user_hash"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gmap_id",
        "month",
        "service_family",
        "name",
        "address",
        "latitude",
        "longitude",
        "n_source_rows",
        "n_duplicate_rows",
        "n_reviews",
        "n_unique_reviewers",
        "mean_rating",
        "negative_share",
        "positive_share",
        "text_share",
        "owner_response_share",
        "n_1star",
        "n_2star",
        "n_3star",
        "n_4star",
        "n_5star",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (gmap_id, month), bucket in sorted(aggregates.items()):
            place = places.get(gmap_id, {})
            n = bucket["n_reviews"]
            nr = bucket["n_rating"]
            writer.writerow(
                {
                    "gmap_id": gmap_id,
                    "month": month,
                    "service_family": place.get("service_family", ""),
                    "name": place.get("name", ""),
                    "address": place.get("address", ""),
                    "latitude": place.get("latitude", ""),
                    "longitude": place.get("longitude", ""),
                    "n_source_rows": bucket["n_source_rows"],
                    "n_duplicate_rows": bucket["n_duplicate_rows"],
                    "n_reviews": n,
                    "n_unique_reviewers": len(bucket["users"]),
                    "mean_rating": bucket["rating_sum"] / nr if nr else "",
                    "negative_share": (bucket["n_1star"] + bucket["n_2star"]) / nr if nr else "",
                    "positive_share": (bucket["n_4star"] + bucket["n_5star"]) / nr if nr else "",
                    "text_share": bucket["n_text"] / n if n else "",
                    "owner_response_share": bucket["n_response"] / n if n else "",
                    "n_1star": bucket["n_1star"],
                    "n_2star": bucket["n_2star"],
                    "n_3star": bucket["n_3star"],
                    "n_4star": bucket["n_4star"],
                    "n_5star": bucket["n_5star"],
                }
            )
    print(
        json.dumps(
            {
                "place_month_rows": len(aggregates),
                "source_review_rows": source_rows,
                "analysis_review_rows": source_rows - duplicate_rows,
                "duplicate_rows_omitted": duplicate_rows,
                "deduplicate": args.deduplicate,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
