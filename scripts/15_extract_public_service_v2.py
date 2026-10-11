#!/usr/bin/env python3
"""Nationwide deterministic v2 extraction with explicit uncertainty and roster linkage.

Raw archives and earlier results are immutable. Current official structure
matches support identity only, not historical public access or ownership.
No human labels, LLM calls, or reviewer data are used.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("service_v1", HERE / "13_extract_public_service_places.py")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
PARTITIONS = v1.PARTITIONS
ABBREVS = "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
STATE = dict(zip(PARTITIONS[:-1], ABBREVS))
US_STATES = set(ABBREVS)
TERRITORIES = {"PR", "VI", "GU", "AS", "MP"}
SOURCE_FAMILY = {"Post Offices": "postal_service", "Police Stations": "police",
                 "Courthouses": "court", "State Supreme Courts": "court",
                 "U.S. Supreme Court": "court", "City/Town Halls": "civic_administration"}
GRID = 0.01
FIELDS = [
    "gmap_id", "name", "address", "latitude", "longitude", "categories",
    "state", "state_basis", "source_partitions", "source_line", "avg_rating", "num_of_reviews",
    "service_family", "all_service_families", "candidate", "classification_status",
    "operator_type", "operator_evidence", "selection_basis", "matched_rules",
    "exclusion_reason", "conflict_flags", "rules_version", "model_probability", "adjudicated_label",
    "official_match_status", "official_match_count", "official_links",
    "physical_site_group_id", "site_group_basis", "historical_status", "public_access_status",
]
LINK_FIELDS = ["gmap_id", "roster_key", "roster_id", "official_record_id", "official_name",
               "official_address", "official_state", "official_family", "source_url",
               "source_vintage", "distance_m", "name_similarity", "address_match",
               "match_method", "match_status", "final_classification_status"]
COVERAGE_FIELDS = ["roster_key", "roster_id", "official_record_id", "official_name",
                   "official_address", "official_state", "official_family", "source_vintage",
                   "source_url", "coordinate_usable", "matched_google_ids", "core_google_ids",
                   "ambiguous_google_ids", "coverage_status"]

def now():
    return datetime.now(timezone.utc).isoformat()

def categories(record):
    value = record.get("category") or []
    return [value] if isinstance(value, str) else [x for x in value if isinstance(x, str)]

def textnorm(value):
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())

def name_similarity(a, b):
    a, b = textnorm(a), textnorm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    tokens_a, tokens_b = set(a.split()), set(b.split())
    # No containment shortcut: generic shared words must not confirm an identity.
    jac = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return max(jac, SequenceMatcher(None, a, b, autojunk=False).ratio())

def street_key(address):
    for segment in str(address or "").split(","):
        match = re.search(r"\b(\d+[a-z]?(?:-\d+)?)\s+(.+)", segment, re.I)
        if not match:
            continue
        words = textnorm(match.group()).split()
        aliases = {"street": "st", "avenue": "ave", "road": "rd", "boulevard": "blvd",
                   "drive": "dr", "lane": "ln", "court": "ct", "highway": "hwy",
                   "north": "n", "south": "s", "east": "e", "west": "w"}
        clean = []
        for word in words:
            if word in {"suite", "ste", "unit", "floor"}:
                break
            clean.append(aliases.get(word, word))
        if len(clean) >= 3 and not re.fullmatch(r"\d{5}(?:-\d{4})?", clean[0]):
            return " ".join(clean)
    return ""

def coords(record):
    try:
        lat, lon = float(record["latitude"]), float(record["longitude"])
        if math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except (ValueError, TypeError, KeyError):
        pass
    return None

def distance_m(a, b):
    la, lo, lb, lob = map(math.radians, (*a, *b))
    h = math.sin((la-lb)/2)**2 + math.cos(la)*math.cos(lb)*math.sin((lo-lob)/2)**2
    return 6371008.8 * 2 * math.asin(min(1, math.sqrt(max(0, h))))

def state_for(record, partition):
    matches = re.findall(r"(?:,|\s)([A-Z]{2})\s+\d{5}(?:-\d{4})?\b", str(record.get("address") or ""))
    if matches and matches[-1] in US_STATES | TERRITORIES:
        return matches[-1], "address_state_zip"
    return STATE.get(partition, ""), "source_partition" if partition in STATE else "unresolved_other"

def load_rules(path):
    rules = json.loads(path.read_text(encoding="utf-8"))
    rules["_families"] = [f["id"] for f in rules["families"]]
    if len(set(rules["_families"])) != len(rules["_families"]):
        raise ValueError("Duplicate family")
    all_patterns = []
    for f in rules["families"]:
        for key in ("name_patterns", "weak_name_patterns"):
            f["_" + key] = [re.compile(p) for p in f[key]]
            all_patterns.extend(f[key])
        f["_categories"] = {v1.normalize(c) for c in f["categories"]}
        f["_weak_categories"] = {v1.normalize(c) for c in f["weak_categories"]}
    for ex in rules["exclusions"]:
        ex["_name"] = [re.compile(p) for p in ex["name_patterns"]]
        ex["_cat"] = [re.compile(p) for p in ex["category_patterns"]]
    for alias in rules["state_aliases"]:
        alias["_pattern"] = re.compile(alias["pattern"])
        alias["_required"] = re.compile(alias["required_name"])
        all_patterns.append(alias["pattern"])
    rules["_gate"] = re.compile("|".join("(?:%s)" % p for p in all_patterns))
    rules["_gov_categories"] = {v1.normalize(c) for c in rules["government_categories"]}
    rules["_delegate"] = [re.compile(p) for p in rules["delegated_patterns"]]
    rules["_delegate_cats"] = {v1.normalize(c) for c in rules["delegated_categories"]}
    rules["_all_categories"] = rules["_gov_categories"] | set().union(
        *(f["_categories"] | f["_weak_categories"] for f in rules["families"]))
    return rules

def rule_evidence(record, state, rules, force=False):
    name = v1.normalize(record.get("name"))
    cats = {v1.normalize(c) for c in categories(record)}
    hits, strong, weak = [], set(), set()
    name_gate = bool(rules["_gate"].search(name))
    if not force and not name_gate and not cats & rules["_all_categories"]:
        return {"strong": set(), "weak": set(), "strong_name": set(), "hits": [],
                "exclusions": [], "generic_government": False,
                "delegate_name": False, "delegate_category": False}
    strong_name = set()
    for f in rules["families"]:
        fid = f["id"]
        if name_gate:
            for p in f["_name_patterns"]:
                m = p.search(name)
                if m:
                    strong.add(fid)
                    strong_name.add(fid)
                    hits.append({"family": fid, "field": "name", "strength": "strong", "pattern": p.pattern, "text": m.group()})
                    break
            for p in f["_weak_name_patterns"]:
                m = p.search(name)
                if m:
                    weak.add(fid)
                    hits.append({"family": fid, "field": "name", "strength": "conditional", "pattern": p.pattern, "text": m.group()})
                    break
        for cat in cats & f["_categories"]:
            if f.get("category_requires_operator_evidence"):
                weak.add(fid)
                strength = "conditional_operator"
            else:
                strong.add(fid)
                strength = "strong_category"
            hits.append({"family": fid, "field": "category", "strength": strength, "text": cat})
        for cat in cats & f["_weak_categories"]:
            weak.add(fid)
            hits.append({"family": fid, "field": "category", "strength": "conditional", "text": cat})
    for alias in rules["state_aliases"]:
        if state in alias["states"] and alias["_pattern"].search(name):
            weak.add(alias["family"])
            qualifying = bool(alias["_required"].search(name))
            if qualifying:
                strong.add(alias["family"])
                strong_name.add(alias["family"])
            hits.append({"family": alias["family"], "field": "state_name_alias", "text": name,
                         "state": state, "qualifying": qualifying, "source": rules["sources"][alias["source"]]})
    govcat = bool(cats & rules["_gov_categories"])
    if govcat:
        strong |= weak
    exclusions = []
    for ex in rules["exclusions"]:
        name_hits = [p.pattern for p in ex["_name"] if p.search(name)]
        cat_hits = [p.pattern for p in ex["_cat"] if any(p.search(c) for c in cats)]
        if name_hits or cat_hits:
            exclusions.append({"id": ex["id"], "kind": ex["kind"],
                               "name": name_hits, "category": cat_hits})
    delegate_name = any(p.search(name) for p in rules["_delegate"])
    delegate_cat = bool(cats & rules["_delegate_cats"])
    return {"strong": strong, "weak": weak, "strong_name": strong_name, "hits": hits,
            "exclusions": exclusions, "generic_government": govcat,
            "delegate_name": delegate_name, "delegate_category": delegate_cat}

class RosterIndex:
    def __init__(self, paths):
        self.rows, self.grid = {}, defaultdict(list)
        self.input_counts, self.skipped = Counter(), Counter()
        for path in paths:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    family = SOURCE_FAMILY.get(row.get("source_subtype"))
                    if row.get("roster_id") == "hud_pha_current":
                        family = "public_housing"
                    if not family:
                        self.skipped["out_of_scope_service"] += 1
                        continue
                    if row.get("state") not in US_STATES:
                        self.skipped["outside_50_states_dc"] += 1
                        continue
                    key = ":".join((row["roster_id"], row.get("source_subtype", ""), row["source_record_id"]))
                    if key in self.rows:
                        raise ValueError("Duplicate official identity: " + key)
                    row = dict(row, family=family, roster_key=key)
                    row["_coords"] = coords(row)
                    row["_name"] = textnorm(row["entity_name"])
                    row["_street"] = street_key(row.get("address_line1", ""))
                    self.rows[key] = row
                    self.input_counts[family] += 1
                    if row["_coords"]:
                        lat, lon = row["_coords"]
                        self.grid[(row["state"], math.floor(lat/GRID), math.floor(lon/GRID))].append(key)

    def matches(self, record, state):
        location = coords(record)
        if not location or state not in US_STATES:
            return []
        lat, lon = location
        x, y = math.floor(lat/GRID), math.floor(lon/GRID)
        # 300 metres, conservatively expanded at high latitude.
        dx = math.ceil(0.003/GRID)
        dy = math.ceil(0.003 / max(0.05, math.cos(math.radians(lat))) / GRID)
        found = []
        gname, gstreet = textnorm(record.get("name")), street_key(record.get("address"))
        for i in range(x-dx, x+dx+1):
            for j in range(y-dy, y+dy+1):
                for key in self.grid.get((state, i, j), ()):
                    off = self.rows[key]
                    dist = distance_m(location, off["_coords"])
                    if dist > 300:
                        continue
                    score = name_similarity(gname, off["_name"])
                    addr = bool(gstreet and off["_street"] and gstreet == off["_street"])
                    # Exact names still require close geography. Address agreement
                    # never substitutes for name evidence.
                    method = ""
                    if score >= 0.94 and dist <= 150:
                        method = "very_close_name_and_geo"
                    elif addr and score >= 0.70 and dist <= 300:
                        method = "exact_street_name_and_geo"
                    elif score >= 0.84 and dist <= 75:
                        method = "strong_name_and_very_close_geo"
                    elif (off["family"] == "postal_service" and addr and dist <= 150
                          and re.search(r"\b(?:usps|united states postal service|u s postal service|us postal service)\b", gname)):
                        # Google commonly uses only the national USPS brand;
                        # USGS names the local branch. Require exact street and
                        # close geography; keep competing branch matches ambiguous.
                        method = "postal_brand_exact_street_and_geo"
                    if method:
                        found.append({"roster_key": key, "family": off["family"],
                                      "distance_m": round(dist, 2), "name_similarity": round(score, 4),
                                      "address_match": int(addr), "method": method})
        found.sort(key=lambda r: (r["distance_m"], -r["name_similarity"], r["roster_key"]))
        # Competing records within the same service family are unresolved. Different
        # services in a shared building may legitimately produce multiple links.
        counts = Counter(r["family"] for r in found)
        for match in found:
            match["status"] = "unique_automatic_match" if counts[match["family"]] == 1 else "ambiguous_same_family"
        return found

def decide(evidence, links, rules):
    unique = [x for x in links if x["status"] == "unique_automatic_match"]
    roster_families = {x["family"] for x in unique}
    strong = set(evidence["strong"])
    families = strong | evidence["weak"] | {x["family"] for x in links}
    flags = []
    exclusions = evidence["exclusions"]
    hard = [e["id"] for e in exclusions if e["kind"] in {"hard", "out_of_scope"}]
    weak = [e for e in exclusions if e["kind"] == "weak"]
    if not families and not evidence["generic_government"]:
        return None
    if hard:
        status, operator, basis = "excluded", "not_classified", "hard_exclusion"
        reasons = hard
    elif evidence["delegate_name"]:
        status, operator, basis = "non_government_or_delegated", "private_or_delegated_unverified", "private_or_delegated_name"
        reasons = ["operator_not_established_as_government"]
    elif evidence["delegate_category"] and not (evidence["strong_name"] - {"motor_vehicles"} or unique):
        status, operator, basis = "non_government_or_delegated", "private_or_delegated_unverified", "private_or_delegated_category"
        reasons = ["operator_not_established_as_government"]
    else:
        for ex in weak:
            # Explicit museums/historical names and standalone residential/custodial
            # names are not automatically rescued by a current facility name match.
            if ex["name"]:
                flags.append(ex["id"] + "_name_conflict")
            elif not (evidence["strong_name"] or unique):
                flags.append(ex["id"] + "_category_conflict")
        if flags:
            status, operator, basis = "unresolved", "uncertain", "conflicting_scope_evidence"
            reasons = flags.copy()
        elif strong or unique:
            status = "core"
            if unique:
                operator, basis = "government_inferred_roster_supported", "official_location_match"
            else:
                operator, basis = "government_inferred_rules", "service_name_or_category_rule"
            reasons = []
            if evidence["delegate_category"]:
                flags.append("auxiliary_tag_category_overridden_by_government_name")
            if weak:
                flags.append("auxiliary_exclusion_category_not_a_veto")
            # No unique match for an ambiguous same-family link is claimed.
        else:
            status, operator, basis = "unresolved", "uncertain", "weak_or_generic_service_evidence"
            reasons = ["insufficient_specific_service_or_operator_evidence"]
    if any(x["status"] == "ambiguous_same_family" for x in links):
        flags.append("ambiguous_official_link")
    included = strong | roster_families
    # Weak-family labels are retained as candidates, but cannot silently become
    # additional core functions without independent evidence for that function.
    output_families = included if status == "core" else families
    ordered = [f for f in rules["_families"] if f in output_families]
    return {"families": ordered, "candidate_families": families, "status": status,
            "operator": operator, "basis": basis, "reasons": reasons, "flags": flags,
            "unique_links": unique}

def make_row(record, partition, line, state, state_basis, evidence, links, decision, rules):
    family = decision["families"][0] if decision["families"] else "unspecified_government"
    unique = decision["unique_links"]
    match_status = "unique_automatic_match" if unique else ("ambiguous" if links else "not_matched")
    return {
        "gmap_id": str(record.get("gmap_id") or "").strip(), "name": record.get("name") or "",
        "address": record.get("address") or "", "latitude": record.get("latitude", ""),
        "longitude": record.get("longitude", ""), "categories": " | ".join(categories(record)),
        "state": state, "state_basis": state_basis, "source_partitions": partition, "source_line": line,
        "avg_rating": record.get("avg_rating", ""), "num_of_reviews": record.get("num_of_reviews", ""),
        "service_family": family, "all_service_families": ";".join(decision["families"]),
        "candidate": 1, "classification_status": decision["status"], "operator_type": decision["operator"],
        "operator_evidence": decision["basis"], "selection_basis": decision["basis"],
        "matched_rules": json.dumps(evidence["hits"], sort_keys=True, ensure_ascii=False),
        "exclusion_reason": ";".join(decision["reasons"]), "conflict_flags": ";".join(decision["flags"]),
        "rules_version": rules["version"], "model_probability": "", "adjudicated_label": "",
        "official_match_status": match_status, "official_match_count": len(unique),
        "official_links": json.dumps(links, sort_keys=True), "physical_site_group_id": "",
        "site_group_basis": "", "historical_status": "observed_ucsd_snapshot_not_verified_active_2021",
        "public_access_status": "inferred_service_function_not_verified_open_to_public",
    }

def merge_row(target, row, rules):
    key = row["gmap_id"]
    if key not in target:
        target[key] = row
        return
    old = target[key]
    partitions = sorted(set(old["source_partitions"].split(";")) | set(row["source_partitions"].split(";")))
    states_conflict = old["state"] != row["state"]
    status_conflict = old["classification_status"] != row["classification_status"]
    if status_conflict or states_conflict:
        # Conflicting copies are not silently cherry-picked into the core set.
        old["classification_status"] = "unresolved"
        old["operator_type"] = "uncertain"
        old["exclusion_reason"] = "conflicting_duplicate_metadata"
        old["conflict_flags"] = ";".join(sorted(set(filter(None, old["conflict_flags"].split(";"))) | {"duplicate_metadata_conflict"}))
    if old["classification_status"] == row["classification_status"] == "core":
        fam = set(old["all_service_families"].split(";")) | set(row["all_service_families"].split(";"))
        fam = [f for f in rules["_families"] if f in fam]
        old["all_service_families"], old["service_family"] = ";".join(fam), fam[0]
    links = json.loads(old["official_links"]) + json.loads(row["official_links"])
    best = {}
    for link in links:
        keylink = link["roster_key"]
        if keylink not in best or link["distance_m"] < best[keylink]["distance_m"]:
            best[keylink] = link
    family_counts = Counter(x["family"] for x in best.values())
    for x in best.values():
        x["status"] = "unique_automatic_match" if family_counts[x["family"]] == 1 else "ambiguous_same_family"
    old["official_links"] = json.dumps(sorted(best.values(), key=lambda x: x["roster_key"]), sort_keys=True)
    old["official_match_count"] = sum(x["status"] == "unique_automatic_match" for x in best.values())
    old["official_match_status"] = "unique_automatic_match" if old["official_match_count"] else ("ambiguous" if best else "not_matched")
    old["source_partitions"] = ";".join(partitions)

def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def site_group(row):
    # Co-location aid, not an asserted canonical real-office identifier.
    skey = street_key(row["address"])
    zipmatch = re.search(r"\b[A-Z]{2}\s+(\d{5})", row["address"])
    if skey and zipmatch and row["state"] in US_STATES:
        value = row["state"] + "|" + zipmatch[1] + "|" + skey
        basis = "normalized_street_state_zip_colocation_only"
    else:
        unique = [x["roster_key"] for x in json.loads(row["official_links"]) if x["status"] == "unique_automatic_match"]
        value = "roster:" + sorted(unique)[0] if len(unique) == 1 else "gmap:" + row["gmap_id"]
        basis = "official_location_colocation_only" if len(unique) == 1 else "unresolved_single_gmap"
    return hashlib.sha256(value.encode()).hexdigest()[:24], basis

def run(metadata_dir, rules_path, roster_paths, output_dir, previous=None, partitions=None):
    rules = load_rules(rules_path)
    rules_hash, code_hash = v1.checksum(rules_path), v1.checksum(Path(__file__))
    selected_parts = sorted(set(partitions or PARTITIONS))
    if set(selected_parts) - set(PARTITIONS):
        raise ValueError("Unknown partition")
    inputs = []
    for p in selected_parts:
        path = metadata_dir / ("meta-%s.json.gz" % p)
        stat = path.stat()
        inputs.append({"partition": p, "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    for path in roster_paths:
        path.stat()
    previous_rows = {}
    if previous:
        with previous.open(newline="", encoding="utf-8") as handle:
            previous_rows = {r["gmap_id"]: r for r in csv.DictReader(handle)}
    roster = RosterIndex(roster_paths)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "by-service").mkdir()
    rows, counts, missing_id = {}, [], 0
    # Keep all passing identity links even when no category/name rule qualifies.
    for source in inputs:
        part, path = source["partition"], Path(source["path"])
        seen, candidates = 0, 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Metadata record is not an object")
                seen += 1
                state, state_basis = state_for(record, part)
                links = roster.matches(record, state)
                evidence = rule_evidence(record, state, rules, force=bool(links))
                result = decide(evidence, links, rules)
                if result is None:
                    continue
                candidates += 1
                row = make_row(record, part, line_no, state, state_basis, evidence, links, result, rules)
                if not row["gmap_id"]:
                    missing_id += 1
                    continue
                if state not in US_STATES:
                    row["classification_status"] = "outside_us_frame"
                    row["exclusion_reason"] = "not_resolved_to_50_states_dc"
                merge_row(rows, row, rules)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
            raise RuntimeError("Raw input changed during extraction")
        counts.append({"partition": part, "records_seen": seen, "candidate_rows": candidates})
        print(json.dumps({"progress": "%s/%s" % (len(counts), len(inputs)), **counts[-1]}), flush=True)

    all_rows = [rows[key] for key in sorted(rows)]
    for row in all_rows:
        row["physical_site_group_id"], row["site_group_basis"] = site_group(row)
    groups = defaultdict(list)
    for row in all_rows:
        groups[row["classification_status"]].append(row)
    write_csv(output_dir / "all-candidates.csv", all_rows, FIELDS)
    filegroups = {"places.csv": "core", "places-non-government-or-delegated.csv": "non_government_or_delegated",
                  "places-unresolved.csv": "unresolved", "matched-exclusions.csv": "excluded",
                  "places-outside-us-frame.csv": "outside_us_frame"}
    for filename, status in filegroups.items():
        write_csv(output_dir / filename, groups[status], FIELDS)
    core = groups["core"]
    for family in rules["_families"]:
        write_csv(output_dir / "by-service" / (family + ".csv"),
                  [r for r in core if family in r["all_service_families"].split(";")], FIELDS)

    coverage = {key: {"matched": set(), "core": set(), "ambiguous": set()} for key in roster.rows}
    links_out = []
    for row in all_rows:
        for link in json.loads(row["official_links"]):
            key = link["roster_key"]
            off = roster.rows[key]
            if link["status"] == "unique_automatic_match":
                coverage[key]["matched"].add(row["gmap_id"])
                if row["classification_status"] == "core":
                    coverage[key]["core"].add(row["gmap_id"])
            else:
                coverage[key]["ambiguous"].add(row["gmap_id"])
            links_out.append({
                "gmap_id": row["gmap_id"], "roster_key": key, "roster_id": off["roster_id"],
                "official_record_id": off["source_record_id"], "official_name": off["entity_name"],
                "official_address": off.get("address_line1", ""), "official_state": off["state"],
                "official_family": off["family"], "source_url": off["source_url"],
                "source_vintage": off["source_vintage"], "distance_m": link["distance_m"],
                "name_similarity": link["name_similarity"], "address_match": link["address_match"],
                "match_method": link["method"], "match_status": link["status"],
                "final_classification_status": row["classification_status"],
            })
    write_csv(output_dir / "official-google-links.csv", links_out, LINK_FIELDS)
    cov_rows = []
    for key in sorted(roster.rows):
        off, cov = roster.rows[key], coverage[key]
        status = "matched_to_core" if cov["core"] else ("matched_not_core" if cov["matched"] else
                 "ambiguous_only" if cov["ambiguous"] else "not_matched")
        cov_rows.append({
            "roster_key": key, "roster_id": off["roster_id"], "official_record_id": off["source_record_id"],
            "official_name": off["entity_name"], "official_address": off.get("address_line1", ""),
            "official_state": off["state"], "official_family": off["family"],
            "source_vintage": off["source_vintage"], "source_url": off["source_url"],
            "coordinate_usable": int(bool(off["_coords"])),
            "matched_google_ids": ";".join(sorted(cov["matched"])),
            "core_google_ids": ";".join(sorted(cov["core"])),
            "ambiguous_google_ids": ";".join(sorted(cov["ambiguous"])), "coverage_status": status,
        })
    write_csv(output_dir / "official-roster-coverage.csv", cov_rows, COVERAGE_FIELDS)

    core_ids = {r["gmap_id"] for r in core}
    changes = []
    for gid in sorted(set(previous_rows) | core_ids):
        old, new = previous_rows.get(gid), rows.get(gid)
        status = "retained_core" if old and gid in core_ids else "added_core" if gid in core_ids else "removed_from_core"
        changes.append({"gmap_id": gid, "name": (new or old)["name"], "change": status,
                        "old_primary_family": old["service_family"] if old else "",
                        "new_families": new["all_service_families"] if new else "",
                        "new_status": new["classification_status"] if new else "no_v2_candidate_evidence",
                        "reason": new["exclusion_reason"] if new else "not_retrieved_by_v2"})
    write_csv(output_dir / "changes-from-v1.csv", changes,
              ["gmap_id", "name", "change", "old_primary_family", "new_families", "new_status", "reason"])
    family_coverage = {}
    for fam in rules["_families"]:
        subset = [r for r in cov_rows if r["official_family"] == fam]
        family_coverage[fam] = {
            "official_location_records_available": len(subset),
            "automatically_matched_records": sum(bool(r["matched_google_ids"]) for r in subset),
            "matched_records_with_core_places": sum(bool(r["core_google_ids"]) for r in subset),
            "ambiguous_only_records": sum(r["coverage_status"] == "ambiguous_only" for r in subset),
            "not_matched_records": sum(r["coverage_status"] == "not_matched" for r in subset),
            "denominator_status": "current_location_discovery_not_2021_census" if subset else "no_office_roster_loaded",
        }
    summary = {
        "status": "complete", "completed_at_utc": now(), "rules_version": rules["version"],
        "human_coding_used": False, "llm_used": False, "partitions": counts,
        "records_seen": sum(x["records_seen"] for x in counts), "missing_id_candidates": missing_id,
        "unique_candidate_ids": len(all_rows), "status_counts": {k: len(v) for k, v in groups.items()},
        "core_multilabel_family_counts": dict(Counter(f for r in core for f in r["all_service_families"].split(";"))),
        "core_primary_family_counts": dict(Counter(r["service_family"] for r in core)),
        "core_state_counts": dict(Counter(r["state"] for r in core)),
        "core_operator_counts": dict(Counter(r["operator_type"] for r in core)),
        "core_official_match_count": sum(r["official_match_status"] == "unique_automatic_match" for r in core),
        "core_colocation_groups_not_confirmed_offices": len({r["physical_site_group_id"] for r in core}),
        "change_counts": dict(Counter(r["change"] for r in changes)),
        "official_coverage": family_coverage, "inputs": inputs,
        "code_sha256": code_hash, "rules_sha256": rules_hash,
        "official_input_sha256": {str(p): v1.checksum(p) for p in roster_paths},
        "previous_sha256": v1.checksum(previous) if previous else None,
        "output_sha256": {str(p.relative_to(output_dir)): v1.checksum(p) for p in sorted(output_dir.rglob("*.csv"))},
        "limitations": [
            "Core means deterministic inclusion with government-operator inference, not adjudicated truth.",
            "Not every service or state has an official office roster loaded; coverage fields state this explicitly.",
            "Current roster points are not a 2021 denominator and do not establish public access or active status.",
            "Not matched does not prove missing from UCSD; strict matching can miss valid entities.",
            "Private/delegated indicators do not establish official authorization; authorization remains unverified.",
            "Co-location groups do not establish unique operating offices; gmap_id remains the stable place key.",
            "No reviews were re-extracted by this metadata-only step.",
        ],
    }
    if v1.checksum(rules_path) != rules_hash or v1.checksum(Path(__file__)) != code_hash:
        raise RuntimeError("Code or rules changed during extraction")
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: summary[k] for k in ["status", "status_counts", "core_multilabel_family_counts", "change_counts"]}), flush=True)
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--official", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--partitions", nargs="+")
    args = parser.parse_args()
    run(args.metadata_dir, args.rules, args.official, args.output_dir, args.previous, args.partitions)
