#!/usr/bin/env python3
"""Stream nationwide metadata using only an explicit name/category rule list.

No human labels, external requests, official roster matching, or LLM inference.
Raw archives and prior outputs are never overwritten.
"""
import argparse
import csv
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PARTITIONS = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|"
    "District_of_Columbia|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|"
    "Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
    "Mississippi|Missouri|Montana|Nebraska|Nevada|New_Hampshire|New_Jersey|"
    "New_Mexico|New_York|North_Carolina|North_Dakota|Ohio|Oklahoma|Oregon|"
    "Pennsylvania|Rhode_Island|South_Carolina|South_Dakota|Tennessee|Texas|"
    "Utah|Vermont|Virginia|Washington|West_Virginia|Wisconsin|Wyoming|other"
).split("|")
FIELDS = [
    "gmap_id", "name", "address", "latitude", "longitude", "categories",
    "source_partition", "source_partitions", "source_line", "avg_rating",
    "num_of_reviews", "service_family", "all_service_families",
    "matched_rules", "rules_version", "candidate", "classification_status",
    "model_probability", "adjudicated_label", "exclusion_reason",
]

def normalize(value):
    value = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(value.replace("\u2019", "'").replace("\u2018", "'").casefold().split())

def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def load_rules(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("match_fields") != ["name", "category"]:
        raise ValueError("Only name and category matching is supported")
    family_ids = [family["id"] for family in config["families"]]
    if len(family_ids) != len(set(family_ids)) or not family_ids:
        raise ValueError("Family IDs must be nonempty and unique")
    category_index = {}
    name_patterns = []
    for family in config["families"]:
        family["_name"] = [re.compile(p) for p in family["name_patterns"]]
        for category in family["categories"]:
            category_index.setdefault(normalize(category), []).append(family["id"])
        name_patterns.extend(family["name_patterns"])
    for rule in config["exclusions"]:
        rule["_name"] = [re.compile(p) for p in rule["name_patterns"]]
        rule["_category"] = [re.compile(p) for p in rule["category_patterns"]]
    config["_category_index"] = category_index
    config["_name_gate"] = re.compile("|".join("(?:%s)" % p for p in name_patterns))
    return config

def classify(record, config):
    name = normalize(record.get("name"))
    raw_categories = record.get("category") or []
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    categories = [normalize(c) for c in raw_categories if isinstance(c, str)]
    hits = []
    for category in categories:
        for family in config["_category_index"].get(category, []):
            hits.append({"family": family, "field": "category", "text": category})
    if config["_name_gate"].search(name):
        for family in config["families"]:
            for pattern in family["_name"]:
                match = pattern.search(name)
                if match:
                    hits.append({"family": family["id"], "field": "name",
                                 "pattern": pattern.pattern, "text": match.group()})
                    break
    if not hits:
        return [], [], []
    present = {hit["family"] for hit in hits}
    families = [f["id"] for f in config["families"] if f["id"] in present]
    exclusions = []
    for rule in config["exclusions"]:
        if any(p.search(name) for p in rule["_name"]) or any(
                p.search(c) for c in categories for p in rule["_category"]):
            exclusions.append(rule["id"])
    return families, hits, exclusions

def output_row(record, partition, line, config, families, hits, exclusions):
    categories = record.get("category") or []
    if isinstance(categories, str):
        categories = [categories]
    return {
        "gmap_id": str(record.get("gmap_id") or "").strip(),
        "name": record.get("name") or "", "address": record.get("address") or "",
        "latitude": record.get("latitude"), "longitude": record.get("longitude"),
        "categories": " | ".join(str(c) for c in categories),
        "source_partition": partition, "source_partitions": partition,
        "source_line": line, "avg_rating": record.get("avg_rating"),
        "num_of_reviews": record.get("num_of_reviews"),
        "service_family": families[0], "all_service_families": ";".join(families),
        "matched_rules": json.dumps(hits, ensure_ascii=False, sort_keys=True),
        "rules_version": config["version"], "candidate": 1,
        "classification_status": "rule_excluded" if exclusions else "rule_selected",
        "model_probability": "", "adjudicated_label": "",
        "exclusion_reason": ";".join(exclusions),
    }

def add_unique(target, row, config):
    key = row["gmap_id"]
    if key not in target:
        target[key] = row
        return False
    kept = target[key]
    kept["source_partitions"] = ";".join(sorted(
        set(kept["source_partitions"].split(";")) | {row["source_partition"]}))
    present = set(kept["all_service_families"].split(";")) | set(row["all_service_families"].split(";"))
    families = [f["id"] for f in config["families"] if f["id"] in present]
    kept["service_family"] = families[0]
    kept["all_service_families"] = ";".join(families)
    hits = json.loads(kept["matched_rules"]) + json.loads(row["matched_rules"])
    hits = list({json.dumps(h, sort_keys=True, ensure_ascii=False): h for h in hits}.values())
    kept["matched_rules"] = json.dumps(hits, ensure_ascii=False, sort_keys=True)
    return True

def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def run(metadata_dir, rules_path, output_dir, partitions=None):
    config = load_rules(rules_path)
    partitions = sorted(set(partitions or PARTITIONS))
    invalid = set(partitions) - set(PARTITIONS)
    if invalid:
        raise ValueError("Unknown partitions: %s" % sorted(invalid))
    inputs = []
    for partition in partitions:
        path = metadata_dir / ("meta-%s.json.gz" % partition)
        stat = path.stat()  # Fail before creating output if any input is missing.
        inputs.append({"partition": partition, "path": str(path.resolve()),
                       "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "by-service").mkdir()
    selected, other = {}, {}
    summaries = []
    excluded_path = output_dir / "matched-exclusions.csv"
    with excluded_path.open("x", encoding="utf-8", newline="") as handle:
        excluded_writer = csv.DictWriter(handle, fieldnames=FIELDS)
        excluded_writer.writeheader()
        for index, source in enumerate(inputs, start=1):
            partition = source["partition"]
            path = Path(source["path"])
            counts = Counter(records_seen=0, matched_rows=0, excluded_rows=0,
                             missing_gmap_id=0, selected_rows=0, duplicate_selected_rows=0)
            with gzip.open(path, "rt", encoding="utf-8") as archive:
                for line_number, line in enumerate(archive, start=1):
                    if not line.strip():
                        continue
                    counts["records_seen"] += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError("%s:%s: invalid JSON" % (path, line_number)) from error
                    if not isinstance(record, dict):
                        raise ValueError("%s:%s: expected JSON object" % (path, line_number))
                    families, hits, exclusions = classify(record, config)
                    if not families:
                        continue
                    counts["matched_rows"] += 1
                    row = output_row(record, partition, line_number, config, families, hits, exclusions)
                    if not row["gmap_id"]:
                        counts["missing_gmap_id"] += 1
                        row["classification_status"] = "rule_excluded"
                        row["exclusion_reason"] = ";".join(exclusions + ["missing_gmap_id"])
                        excluded_writer.writerow(row)
                        continue
                    if exclusions:
                        counts["excluded_rows"] += 1
                        excluded_writer.writerow(row)
                        continue
                    counts["selected_rows"] += 1
                    target = other if partition == "other" else selected
                    counts["duplicate_selected_rows"] += int(add_unique(target, row, config))
            stat = path.stat()
            if stat.st_size != source["size_bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise RuntimeError("Input changed during extraction: %s" % path)
            summary = {"partition": partition, **counts}
            summaries.append(summary)
            print(json.dumps({"progress": "%s/%s" % (index, len(inputs)), **summary}), flush=True)
    # The unresolved Other partition cannot create a primary-US place by itself.
    other_overlapping = set(other) & set(selected)
    for key in sorted(other_overlapping):
        add_unique(selected, other[key], config)
    other_only = [other[key] for key in sorted(set(other) - set(selected))]
    rows = [selected[key] for key in sorted(selected)]
    write_csv(output_dir / "places.csv", rows)
    write_csv(output_dir / "places-other-only.csv", other_only)
    for family in config["families"]:
        # Primary-family assignment gives nonoverlapping files whose counts add up.
        write_csv(output_dir / "by-service" / (family["id"] + ".csv"),
                  (row for row in rows if row["service_family"] == family["id"]))
    total = Counter()
    for summary in summaries:
        total.update({key: value for key, value in summary.items() if key != "partition"})
    result = {
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification_method": "deterministic_name_keywords_or_exact_category_allowlist",
        "human_coding_used": False, "rules_version": config["version"],
        "rules_sha256": checksum(rules_path), "script_sha256": checksum(Path(__file__)),
        "partitions": summaries, "totals": dict(total),
        "primary_unique_places": len(rows), "other_only_unique_places": len(other_only),
        "other_overlap_primary": len(other_overlapping),
        "primary_family_counts": dict(Counter(r["service_family"] for r in rows)),
        "inputs": inputs,
        "notes": [
            "No human labels, student files, LLM inference or official roster joins used.",
            "Selected places operationalize this rule-defined service-office scope; accuracy is not estimated.",
            "No address or description matching; exclusions override inclusion.",
            "US primary frame uses state/DC partitions; unresolved Other-only places are separate.",
            "Deduplication uses gmap_id; first selected metadata row is kept and matched families are unioned.",
            "Empty model_probability/adjudicated_label are provenance fields, not required human tasks.",
        ],
        "output_sha256": {str(p.relative_to(output_dir)): checksum(p)
                          for p in sorted(output_dir.rglob("*.csv"))},
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "primary_unique_places": len(rows),
                      "other_only_unique_places": len(other_only),
                      "primary_family_counts": result["primary_family_counts"],
                      "output_dir": str(output_dir.resolve())}, ensure_ascii=False), flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partitions", nargs="+",
                        help="Optional subset, using file names such as Vermont or New_York")
    args = parser.parse_args()
    run(args.metadata_dir, args.rules, args.output_dir, args.partitions)

if __name__ == "__main__":
    main()
