#!/usr/bin/env python3
"""Prepare one-coder work and a held-back retest from an existing frozen sample.

No new primary sample is drawn. Primary design weights are preserved in the
investigator key. Retests measure within-coder stability, not independent
inter-coder reliability or accuracy. Existing packages cannot be overwritten.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


PROTOCOL_VERSION = "single_coder_v1"
VISIBLE_FIELDS = [
    "work_item_id", "source_partition", "google_name", "google_address",
    "google_latitude", "google_longitude", "google_categories",
    "codebook_version", "coder_id", "coded_at_utc", "public_sector_label",
    "service_family", "ownership_status", "public_facing", "analytic_tier",
    "evidence_url", "evidence_note", "evidence_vintage", "exclusion_reason",
    "coder_notes",
]
PLACE_FIELDS = VISIBLE_FIELDS[1:7]
KEY_FIELDS = [
    "sample_id", "gmap_id", "sample_stratum", "stratum_population",
    "stratum_sample_n", "selection_probability", "analysis_weight",
    "initial_work_item_id", "retest_work_item_id", "retest_selection_probability",
]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opaque_id(sample_id, phase, seed):
    key = f"{PROTOCOL_VERSION}|{seed}|{phase}|{sample_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def visible_row(row, phase, seed, codebook_version):
    result = dict.fromkeys(VISIBLE_FIELDS, "")
    result.update({field: row[field] for field in PLACE_FIELDS})
    result["work_item_id"] = opaque_id(row["sample_id"], phase, seed)
    result["codebook_version"] = codebook_version
    return result


def write_csv(path, fields, rows):
    # A UTF-8 BOM helps Excel display non-ASCII place names without conversion.
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(master_path, output_dir, codebook, instructions, per_stratum=10, seed=20260921):
    if per_stratum < 1:
        raise ValueError("Retest sample size must be positive")
    with master_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The frozen master is empty")
    required = set(KEY_FIELDS[:7]) | set(PLACE_FIELDS)
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing master fields: {sorted(required - set(rows[0]))}")
    for key in ("sample_id", "gmap_id"):
        if any(not row[key] for row in rows) or len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Blank or duplicate {key} in frozen master")
    for path in (codebook, instructions):
        if not path.is_file():
            raise ValueError(f"Missing instruction file: {path}")

    grouped = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        grouped[row["sample_stratum"]].append(row)
    rng = random.Random(seed)
    selected = set()
    retest_probabilities = {}
    for stratum, group in sorted(grouped.items()):
        n = min(per_stratum, len(group))
        selected.update(row["sample_id"] for row in rng.sample(group, n))
        retest_probabilities[stratum] = n / len(group)

    codebook_hash = fingerprint(codebook)
    version = "sha256:" + codebook_hash
    initial = [visible_row(row, "initial", seed, version) for row in rows]
    retest = [visible_row(row, "retest", seed, version) for row in rows
              if row["sample_id"] in selected]
    random.Random(seed + 1).shuffle(initial)
    random.Random(seed + 2).shuffle(retest)
    keys = []
    for row in rows:
        key = {field: row[field] for field in KEY_FIELDS[:7]}
        key.update({
            "initial_work_item_id": opaque_id(row["sample_id"], "initial", seed),
            "retest_work_item_id": (
                opaque_id(row["sample_id"], "retest", seed)
                if row["sample_id"] in selected else ""
            ),
            "retest_selection_probability": retest_probabilities[row["sample_stratum"]],
        })
        keys.append(key)

    # Fail before any writes if the destination exists, including coded files.
    output_dir.mkdir(parents=True, exist_ok=False)
    student = output_dir / "student_send_now"
    investigator = output_dir / "investigator_only"
    student.mkdir()
    investigator.mkdir()
    write_csv(student / "initial.csv", VISIBLE_FIELDS, initial)
    write_csv(investigator / "retest-held-back.csv", VISIBLE_FIELDS, retest)
    write_csv(investigator / "linkage-key.csv", KEY_FIELDS, keys)
    shutil.copyfile(codebook, student / "classification-codebook.md")
    shutil.copyfile(instructions, student / "学生操作说明.md")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "master_sha256": fingerprint(master_path),
        "codebook_sha256": codebook_hash,
        "instructions_sha256": fingerprint(instructions),
        "seed": seed,
        "initial_n": len(initial),
        "retest_n": len(retest),
        "strata": {name: {"initial_n": len(group),
                           "retest_n": min(per_stratum, len(group))}
                   for name, group in sorted(grouped.items())},
        "release_policy": "Send only student_send_now initially; hold back retest and key.",
        "retest_policy": "Freeze initial labels, wait 7-14 days, release blank retest only.",
        "interpretation": "Single-coder reference labels; retest is intra-coder stability.",
        "files": {str(path.relative_to(output_dir)): fingerprint(path)
                  for path in sorted(output_dir.rglob("*")) if path.is_file()},
    }
    with (investigator / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--instructions", required=True, type=Path)
    parser.add_argument("--retest-per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    result = prepare(args.master, args.output_dir, args.codebook, args.instructions,
                     args.retest_per_stratum, args.seed)
    print(json.dumps({k: result[k] for k in ("protocol_version", "initial_n", "retest_n", "strata")}, indent=2))


if __name__ == "__main__":
    main()
