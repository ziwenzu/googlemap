#!/usr/bin/env python3
"""Stream reviews for candidate places and remove direct reviewer identifiers."""

import argparse
import csv
import gzip
import hashlib
import hmac
import json
import secrets
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--places", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--salt-file", type=Path, default=Path(".secrets/reviewer_salt"))
    parser.add_argument(
        "--place-mode",
        choices=("all-candidates", "adjudicated"),
        default="all-candidates",
        help="Use every retrieval candidate for engineering, or only adjudicated eligible labels.",
    )
    parser.add_argument(
        "--eligible-labels",
        default="core,secondary",
        help="Comma-separated manual_label values admitted when --place-mode=adjudicated.",
    )
    return parser.parse_args()


def load_place_ids(path, mode, eligible_labels):
    allowed = {label.strip().casefold() for label in eligible_labels.split(",") if label.strip()}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if mode == "all-candidates":
        return {row["gmap_id"] for row in rows if row.get("gmap_id")}
    return {
        row["gmap_id"]
        for row in rows
        if row.get("gmap_id") and row.get("manual_label", "").strip().casefold() in allowed
    }


def get_salt(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
        path.chmod(0o600)
    return path.read_text(encoding="utf-8").strip()


def keyed_hash(value, salt):
    if not value:
        return None
    return hmac.new(salt.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()


def normalize_text(value):
    return " ".join(str(value or "").split())


def hash_user(user_id, salt):
    if not user_id:
        return None
    return keyed_hash(str(user_id), salt)


def main():
    args = parse_args()
    if not args.reviews.exists():
        raise SystemExit(f"Missing review archive: {args.reviews}")
    place_ids = load_place_ids(args.places, args.place_mode, args.eligible_labels)
    if not place_ids:
        raise SystemExit(
            "No eligible place IDs found. For adjudicated mode, complete manual_label first."
        )
    salt = get_salt(args.salt_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_state = args.reviews.name.removeprefix("review-").removesuffix(".json.gz")
    seen = retained = decode_errors = 0
    with gzip.open(args.reviews, "rt", encoding="utf-8", errors="replace") as source, gzip.open(
        args.output, "wt", encoding="utf-8"
    ) as destination:
        for source_line, line in enumerate(source, start=1):
            seen += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            if record.get("gmap_id") not in place_ids:
                continue
            response = record.get("resp") or {}
            normalized_text = normalize_text(record.get("text"))
            review_fingerprint = "|".join(
                [
                    str(record.get("gmap_id") or ""),
                    str(record.get("user_id") or ""),
                    str(record.get("time") or ""),
                    str(record.get("rating") or ""),
                    normalized_text,
                ]
            )
            safe_record = {
                "review_key": keyed_hash(review_fingerprint, salt),
                "source_state": source_state,
                "source_line": source_line,
                "gmap_id": record.get("gmap_id"),
                "user_hash": hash_user(record.get("user_id"), salt),
                "time": record.get("time"),
                "rating": record.get("rating"),
                "text": record.get("text"),
                "text_hash": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
                if normalized_text
                else None,
                "text_length": len(normalized_text),
                "has_pictures": bool(record.get("pics")),
                "has_response": bool(response),
                "response_time": response.get("time"),
                "response_text": response.get("text"),
            }
            destination.write(json.dumps(safe_record, ensure_ascii=False) + "\n")
            retained += 1
    print(
        json.dumps(
            {
                "review_archive": str(args.reviews),
                "candidate_places": len(place_ids),
                "place_mode": args.place_mode,
                "records_seen": seen,
                "reviews_retained": retained,
                "json_decode_errors": decode_errors,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
