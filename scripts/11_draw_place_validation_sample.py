#!/usr/bin/env python3
"""Draw a weighted, blinded validation sample for public-sector place coding."""

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from pathlib import Path


VALIDATION_VERSION = "place_validation_v0.1"
MASTER_FIELDS = [
    "sample_id", "validation_version", "sample_stratum", "stratum_population",
    "stratum_sample_n", "selection_probability", "analysis_weight",
    "prediction_source", "source_partition", "gmap_id", "google_name",
    "google_address", "google_latitude", "google_longitude", "google_categories",
    "predicted_service_family", "official_match_status", "official_link_score",
    "official_name", "official_source_subtype", "official_owner_class",
]
CODER_FIELDS = [
    "sample_id", "source_partition", "google_name", "google_address",
    "google_latitude", "google_longitude", "google_categories",
    "public_sector_label", "service_family", "public_facing", "analytic_tier",
    "exclusion_reason", "coder_notes",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linkage", required=True, type=Path)
    parser.add_argument("--rule-candidates", required=True, type=Path)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--partitions", required=True, help="Comma-separated metadata partition names")
    parser.add_argument("--per-stratum", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--master-output", required=True, type=Path)
    parser.add_argument("--coder-a-output", required=True, type=Path)
    parser.add_argument("--coder-b-output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def open_csv(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def clean(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(clean(item) for item in value)
    if isinstance(value, dict):
        return " | ".join(f"{clean(key)} {clean(item)}" for key, item in value.items())
    return re.sub(r"\s+", " ", str(value)).strip()


def partition_name(path):
    return path.name.removeprefix("meta-").removesuffix(".json.gz")


def sample_id(stratum, gmap_id):
    value = f"{VALIDATION_VERSION}|{stratum}|{gmap_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def best_linkage_rows(path, partitions):
    status_rank = {"confirmed_official_candidate": 2, "needs_review": 1}
    best = {}
    with open_csv(path) as handle:
        for row in csv.DictReader(handle):
            if row.get("source_partition") not in partitions:
                continue
            gmap_id = row.get("gmap_id", "")
            if not gmap_id:
                continue
            rank = (
                status_rank.get(row.get("official_match_status"), 0),
                float(row.get("official_link_score") or 0),
            )
            if gmap_id not in best or rank > best[gmap_id][0]:
                best[gmap_id] = (rank, row)
    return {gmap_id: value[1] for gmap_id, value in best.items()}


def rule_rows(path, partitions):
    best = {}
    with open_csv(path) as handle:
        for row in csv.DictReader(handle):
            if row.get("source_partition") not in partitions:
                continue
            gmap_id = row.get("gmap_id", "")
            if gmap_id and gmap_id not in best:
                best[gmap_id] = row
    return best


def from_linkage(row):
    return {
        "prediction_source": "usgs_reverse_linkage",
        "source_partition": row.get("source_partition", ""),
        "gmap_id": row.get("gmap_id", ""),
        "google_name": row.get("google_name", ""),
        "google_address": row.get("google_address", ""),
        "google_latitude": row.get("google_latitude", ""),
        "google_longitude": row.get("google_longitude", ""),
        "google_categories": row.get("google_categories", ""),
        "predicted_service_family": row.get("service_family", ""),
        "official_match_status": row.get("official_match_status", ""),
        "official_link_score": row.get("official_link_score", ""),
        "official_name": row.get("official_name", ""),
        "official_source_subtype": row.get("official_source_subtype", ""),
        "official_owner_class": row.get("official_owner_class", ""),
    }


def from_rule(row):
    return {
        "prediction_source": "keyword_rule_only",
        "source_partition": row.get("source_partition", ""),
        "gmap_id": row.get("gmap_id", ""),
        "google_name": row.get("name", ""),
        "google_address": row.get("address", ""),
        "google_latitude": row.get("latitude", ""),
        "google_longitude": row.get("longitude", ""),
        "google_categories": row.get("categories", ""),
        "predicted_service_family": row.get("service_family", ""),
        "official_match_status": "not_linked",
        "official_link_score": "",
        "official_name": "",
        "official_source_subtype": "",
        "official_owner_class": "",
    }


def from_metadata(partition, record):
    return {
        "prediction_source": "general_google_negative",
        "source_partition": partition,
        "gmap_id": clean(record.get("gmap_id")),
        "google_name": clean(record.get("name")),
        "google_address": clean(record.get("address")),
        "google_latitude": clean(record.get("latitude")),
        "google_longitude": clean(record.get("longitude")),
        "google_categories": clean(record.get("category")),
        "predicted_service_family": "",
        "official_match_status": "not_candidate",
        "official_link_score": "",
        "official_name": "",
        "official_source_subtype": "",
        "official_owner_class": "",
    }


def reservoir_general_negatives(metadata_dir, partitions, excluded_ids, sample_n, rng):
    reservoir = []
    population = 0
    seen = set()
    decode_errors = 0
    for path in sorted(metadata_dir.glob("meta-*.json.gz")):
        partition = partition_name(path)
        if partition not in partitions:
            continue
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    decode_errors += 1
                    continue
                gmap_id = clean(record.get("gmap_id"))
                if not gmap_id or gmap_id in excluded_ids or gmap_id in seen:
                    continue
                seen.add(gmap_id)
                population += 1
                item = from_metadata(partition, record)
                if len(reservoir) < sample_n:
                    reservoir.append(item)
                else:
                    replace = rng.randrange(population)
                    if replace < sample_n:
                        reservoir[replace] = item
    return population, reservoir, decode_errors


def draw(items, n, rng):
    items = list(items)
    if len(items) <= n:
        return items
    return rng.sample(items, n)


def add_sampling_fields(rows, stratum, population):
    sample_n = len(rows)
    probability = sample_n / population if population else 0
    weight = 1 / probability if probability else 0
    output = []
    for row in rows:
        output.append(
            {
                "sample_id": sample_id(stratum, row["gmap_id"]),
                "validation_version": VALIDATION_VERSION,
                "sample_stratum": stratum,
                "stratum_population": population,
                "stratum_sample_n": sample_n,
                "selection_probability": f"{probability:.10f}",
                "analysis_weight": f"{weight:.6f}",
                **row,
            }
        )
    return output


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coder_rows(master_rows, rng):
    rows = [
        {
            "sample_id": row["sample_id"],
            "source_partition": row["source_partition"],
            "google_name": row["google_name"],
            "google_address": row["google_address"],
            "google_latitude": row["google_latitude"],
            "google_longitude": row["google_longitude"],
            "google_categories": row["google_categories"],
            "public_sector_label": "",
            "service_family": "",
            "public_facing": "",
            "analytic_tier": "",
            "exclusion_reason": "",
            "coder_notes": "",
        }
        for row in master_rows
    ]
    rng.shuffle(rows)
    return rows


def main():
    args = parse_args()
    if args.per_stratum < 1:
        raise SystemExit("--per-stratum must be positive")
    partitions = {value.strip() for value in args.partitions.split(",") if value.strip()}
    rng = random.Random(args.seed)
    links = best_linkage_rows(args.linkage, partitions)
    rules = rule_rows(args.rule_candidates, partitions)

    confirmed = [
        from_linkage(row) for row in links.values()
        if row.get("official_match_status") == "confirmed_official_candidate"
    ]
    needs_review = [
        from_linkage(row) for row in links.values()
        if row.get("official_match_status") == "needs_review"
    ]
    rule_only = [
        from_rule(row) for gmap_id, row in rules.items() if gmap_id not in links
    ]
    excluded = set(links) | set(rules)
    general_population, general_sample, decode_errors = reservoir_general_negatives(
        args.metadata_dir, partitions, excluded, args.per_stratum, rng
    )

    strata = {
        "official_confirmed_candidate": (len(confirmed), draw(confirmed, args.per_stratum, rng)),
        "official_needs_review": (len(needs_review), draw(needs_review, args.per_stratum, rng)),
        "keyword_rule_only": (len(rule_only), draw(rule_only, args.per_stratum, rng)),
        "general_google_negative": (general_population, general_sample),
    }
    master = []
    for stratum, (population, rows) in strata.items():
        master.extend(add_sampling_fields(rows, stratum, population))
    master.sort(key=lambda row: (row["sample_stratum"], row["sample_id"]))
    write_csv(args.master_output, MASTER_FIELDS, master)
    write_csv(args.coder_a_output, CODER_FIELDS, coder_rows(master, random.Random(args.seed + 1)))
    write_csv(args.coder_b_output, CODER_FIELDS, coder_rows(master, random.Random(args.seed + 2)))

    summary = {
        "validation_version": VALIDATION_VERSION,
        "seed": args.seed,
        "partitions": sorted(partitions),
        "requested_per_stratum": args.per_stratum,
        "total_sample": len(master),
        "strata": {
            name: {"population": population, "sample": len(rows)}
            for name, (population, rows) in strata.items()
        },
        "json_decode_errors": decode_errors,
        "coding_rules": {
            "public_sector_label": ["yes", "no", "uncertain"],
            "public_facing": ["yes", "no", "uncertain"],
            "analytic_tier": ["core", "secondary", "separate_sector", "exclude", "uncertain"],
            "instruction": (
                "Coders work independently and cannot see stratum or automated predictions. "
                "Join coder files to the master only after both files are frozen."
            ),
        },
        "outputs": {
            "master": str(args.master_output),
            "coder_a": str(args.coder_a_output),
            "coder_b": str(args.coder_b_output),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
