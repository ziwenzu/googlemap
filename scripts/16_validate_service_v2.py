#!/usr/bin/env python3
"""Validate v2 generated artifacts without modifying extraction outputs."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()

def validate(output, previous):
    summary = json.loads((output / "summary.json").read_text())
    checks = []
    def check(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    check(summary["status"] == "complete", "Extraction completed")
    check(len(summary["partitions"]) == 52, "All 52 publisher partitions processed")
    check(summary["records_seen"] == 4989684, "Expected complete raw metadata row count")
    check(not summary["human_coding_used"] and not summary["llm_used"], "No human or LLM labels used")
    check(summary["previous_sha256"] == digest(previous), "v1 place data unchanged")
    for name, sha in summary["output_sha256"].items():
        check(digest(output / name) == sha, "Output checksum: " + name)
    for name, sha in summary["official_input_sha256"].items():
        check(digest(Path(name)) == sha, "Official input checksum: " + name)
    all_rows = read(output / "all-candidates.csv")
    ids = {r["gmap_id"] for r in all_rows}
    check(len(ids) == len(all_rows) == summary["unique_candidate_ids"], "Unique candidate gmap_id")
    check("" not in ids, "No blank candidate IDs")
    files = {"places.csv": "core", "places-unresolved.csv": "unresolved",
             "places-non-government-or-delegated.csv": "non_government_or_delegated",
             "matched-exclusions.csv": "excluded", "places-outside-us-frame.csv": "outside_us_frame"}
    union = set()
    core = []
    for filename, status in files.items():
        rows = read(output / filename)
        subset = {r["gmap_id"] for r in rows}
        check(len(subset) == len(rows), "Unique IDs: " + filename)
        check(not subset & union, "Disjoint bucket: " + filename)
        check(all(r["classification_status"] == status for r in rows), "Bucket status consistency: " + filename)
        check(len(rows) == summary["status_counts"][status], "Bucket count consistency: " + filename)
        union |= subset
        if status == "core":
            core = rows
    check(union == ids, "Every candidate is retained in one and only one bucket")
    check(all(not r["exclusion_reason"] for r in core), "Core rows have no exclusion reason")
    check(all(not r["model_probability"] and not r["adjudicated_label"] for r in all_rows), "No fabricated model probabilities or adjudication")
    counts = Counter(f for r in core for f in r["all_service_families"].split(";"))
    check(dict(counts) == summary["core_multilabel_family_counts"], "Multilabel service counts match summary")
    for family, count in counts.items():
        subset = read(output / "by-service" / (family + ".csv"))
        expected = {r["gmap_id"] for r in core if family in r["all_service_families"].split(";")}
        check({r["gmap_id"] for r in subset} == expected and len(subset) == count, "Service output completeness: " + family)

    all_by_id = {r["gmap_id"]: r for r in all_rows}
    cases = [
        ("0x872bb719c3a655ab:0x3d26de76ecb203ff", "core", "Apache Junction Municipal Court restored"),
        ("0x888e8e40b149a4af:0xa8ae50136e2ff52e", "core", "Autauga County Probate Court restored"),
        ("0x54d2ecc96852746f:0x59e57ad614085d13", "non_government_or_delegated", "Redding private DMV partner separated"),
    ]
    for key, status, description in cases:
        check(all_by_id.get(key, {}).get("classification_status") == status, description)
    previous_ids = {r["gmap_id"] for r in read(previous)}
    core_ids = {r["gmap_id"] for r in core}
    changes = read(output / "changes-from-v1.csv")
    check({r["gmap_id"] for r in changes} == previous_ids | core_ids, "Every v1/core-v2 ID appears in change report")
    observed = Counter(r["change"] for r in changes)
    check(observed["added_core"] == len(core_ids - previous_ids), "Added-place count")
    check(observed["removed_from_core"] == len(previous_ids - core_ids), "Removed-place count")
    check(observed["retained_core"] == len(previous_ids & core_ids), "Retained-place count")

    coverage = read(output / "official-roster-coverage.csv")
    coverage_keys = {r["roster_key"] for r in coverage}
    check(len(coverage_keys) == len(coverage), "Unique official roster identities")
    links = read(output / "official-google-links.csv")
    check(all(r["gmap_id"] in ids and r["roster_key"] in coverage_keys for r in links), "Official link referential integrity")
    check(all(float(r["distance_m"]) <= 300 for r in links), "Official matching distance limits")
    check(all(r["name_similarity"] and r["match_method"] for r in links), "No coordinate-only matches")
    check(all(float(r["name_similarity"]) >= .70 or (
        r["match_method"] == "postal_brand_exact_street_and_geo"
        and r["official_family"] == "postal_service"
        and r["address_match"] == "1" and float(r["distance_m"]) <= 150)
        for r in links), "Name evidence or exact-address USPS brand match on every official link")
    by_gid_family = {}
    for r in links:
        if r["match_status"] == "unique_automatic_match":
            key = (r["gmap_id"], r["official_family"])
            check(key not in by_gid_family, "Unique official identity per place/service: " + r["gmap_id"])
            by_gid_family[key] = r["roster_key"]
    for family, expected in summary["official_coverage"].items():
        subset = [r for r in coverage if r["official_family"] == family]
        check(len(subset) == expected["official_location_records_available"], "Coverage denominator count: " + family)
        check(sum(bool(r["matched_google_ids"]) for r in subset) == expected["automatically_matched_records"],
              "Coverage matched count: " + family)
        check(sum(bool(r["core_google_ids"]) for r in subset) == expected["matched_records_with_core_places"],
              "Coverage core count: " + family)
    report = {
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": len(checks), "test_scope": "engineering_invariants_and_targeted_regressions_not_population_accuracy",
        "core_places": len(core), "all_candidates": len(all_rows),
        "status_counts": summary["status_counts"], "changes": dict(observed),
        "family_counts": dict(counts), "official_coverage": summary["official_coverage"],
        "targeted_regressions": [x[2] for x in cases],
        "limits": summary["limitations"],
    }
    path = output / "validation-summary.json"
    if path.exists():
        raise FileExistsError("Preserve existing validation: " + str(path))
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    validate(args.output_dir, args.previous)
