#!/usr/bin/env python3
"""Normalize raw USGS structure features while preserving evidentiary limits."""

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


SERVICE_URL = "https://carto.nationalmap.gov/arcgis/rest/services/structures/MapServer"
OWNER_CLASSES = {
    0: "uncertain",
    1: "federal",
    2: "tribal",
    3: "state",
    4: "regional",
    5: "county",
    6: "municipal",
    7: "private",
}
FIELDS = [
    "roster_id",
    "source_record_id",
    "source_subrecord_id",
    "record_grain",
    "evidence_role",
    "government_level",
    "service_family",
    "entity_name",
    "parent_entity_name",
    "address_line1",
    "address_line2",
    "city",
    "state",
    "zip",
    "zip4",
    "county",
    "latitude",
    "longitude",
    "active_status",
    "public_access_status",
    "source_subtype",
    "source_vintage",
    "source_file",
    "source_url",
    "redistribution_status",
    "official_owner_class",
    "owner_eligibility",
    "analytic_tier_hint",
    "ownership_rule",
    "source_originator",
    "source_load_date",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers-config", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def clean(value):
    return " ".join(str(value or "").split())


def load_layers(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["layer_id"]): row for row in csv.DictReader(handle)}


def coordinate(value, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{value:.7f}" if minimum <= value <= maximum else ""


def attribute_value(attributes, name):
    return (
        attributes.get(name)
        if name in attributes
        else attributes.get(name.lower(), attributes.get(name.upper()))
    )


def owner_decision(owner_class, rule):
    if rule == "official_type":
        return "eligible_official_type"
    if owner_class == "private":
        return "exclude_private_owner"
    if owner_class == "uncertain":
        return "needs_review_unknown_owner"
    return "eligible_public_owner"


def public_access_status(eligibility):
    if eligibility == "exclude_private_owner":
        return "not_public_sector_owner"
    if eligibility == "needs_review_unknown_owner":
        return "unknown_owner_and_public_access"
    return "unknown_structure_not_public_access_evidence"


def source_record_id(attributes, layer_id):
    return clean(
        attribute_value(attributes, "PERMANENT_IDENTIFIER")
        or attribute_value(attributes, "GLOBALID")
        or f"{layer_id}:{attribute_value(attributes, 'OBJECTID') or ''}"
    )


def normalize_row(raw, layer):
    feature = raw["feature"]
    attributes = feature.get("attributes") or {}
    geometry = feature.get("geometry") or {}
    owner_code = attribute_value(attributes, "ADMINTYPE")
    try:
        owner_code = int(owner_code)
    except (TypeError, ValueError):
        owner_code = 0
    owner_class = OWNER_CLASSES.get(owner_code, "uncertain")
    eligibility = owner_decision(owner_class, layer["ownership_rule"])
    downloaded = clean(raw.get("downloaded_at_utc"))
    zip_code = clean(attribute_value(attributes, "ZIPCODE"))
    address = clean(attribute_value(attributes, "ADDRESS"))
    building = clean(attribute_value(attributes, "ADDRESSBUILDINGNAME"))
    return {
        "roster_id": "usgs_national_structures_current",
        "source_record_id": source_record_id(attributes, raw["source_layer_id"]),
        "source_subrecord_id": clean(attribute_value(attributes, "OBJECTID")),
        "record_grain": "named_structure_point",
        "evidence_role": "facility_discovery",
        "government_level": owner_class if owner_class != "private" else "",
        "service_family": layer["service_family"],
        "entity_name": clean(attribute_value(attributes, "NAME")),
        "parent_entity_name": "",
        "address_line1": address,
        "address_line2": building if building and building != address else "",
        "city": clean(attribute_value(attributes, "CITY")),
        "state": clean(attribute_value(attributes, "STATE")),
        "zip": zip_code[:5],
        "zip4": zip_code[6:10] if len(zip_code) >= 10 else "",
        "county": "",
        "latitude": coordinate(geometry.get("y"), -90, 90),
        "longitude": coordinate(geometry.get("x"), -180, 180),
        "active_status": "unknown_current_snapshot",
        "public_access_status": public_access_status(eligibility),
        "source_subtype": layer["layer_name"],
        "source_vintage": downloaded[:10] or "current_snapshot",
        "source_file": f"usgs-national-structures-layer-{raw['source_layer_id']:02d}.jsonl.gz",
        "source_url": f"{SERVICE_URL}/{raw['source_layer_id']}",
        "redistribution_status": "public_domain",
        "official_owner_class": owner_class,
        "owner_eligibility": eligibility,
        "analytic_tier_hint": layer["analytic_tier_hint"],
        "ownership_rule": layer["ownership_rule"],
        "source_originator": clean(attribute_value(attributes, "SOURCE_ORIGINATOR")),
        "source_load_date": clean(attribute_value(attributes, "LOADDATE")),
    }


def main():
    args = parse_args()
    layers = load_layers(args.layers_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    by_layer = Counter()
    by_service = Counter()
    by_owner = Counter()
    by_eligibility = Counter()
    blank_names = 0
    blank_coordinates = 0
    seen_ids = set()
    with gzip.open(args.output, "wt", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        for layer_id, layer in layers.items():
            raw_path = args.raw_dir / f"usgs-national-structures-layer-{layer_id:02d}.jsonl.gz"
            if not raw_path.exists():
                raise SystemExit(f"Missing raw USGS layer: {raw_path}")
            with gzip.open(raw_path, "rt", encoding="utf-8") as source:
                for line in source:
                    raw = json.loads(line)
                    row = normalize_row(raw, layer)
                    key = (row["roster_id"], row["source_record_id"])
                    if key in seen_ids:
                        raise SystemExit(f"Duplicate official source ID: {key}")
                    seen_ids.add(key)
                    writer.writerow(row)
                    total += 1
                    by_layer[f"{layer_id}:{layer['layer_name']}"] += 1
                    by_service[row["service_family"]] += 1
                    by_owner[row["official_owner_class"]] += 1
                    by_eligibility[row["owner_eligibility"]] += 1
                    blank_names += not bool(row["entity_name"])
                    blank_coordinates += not bool(row["latitude"] and row["longitude"])

    summary = {
        "roster_id": "usgs_national_structures_current",
        "record_grain": "named_structure_point",
        "evidence_role": "facility_discovery",
        "records": total,
        "counts_by_layer": dict(sorted(by_layer.items())),
        "counts_by_service_family": dict(sorted(by_service.items())),
        "counts_by_owner_class": dict(sorted(by_owner.items())),
        "counts_by_owner_eligibility": dict(sorted(by_eligibility.items())),
        "blank_names": blank_names,
        "blank_coordinates": blank_coordinates,
        "warning": (
            "Current USGS structures are discovery evidence, not a time-aligned "
            "2021 denominator or proof that the public can receive service on site."
        ),
        "output": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
