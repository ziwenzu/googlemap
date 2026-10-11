#!/usr/bin/env python3
"""Fetch public HUD PHA points and the official all-state DMV agency directory.

Downloads are immutable and logged in the project manifest. Contact-person
fields are not requested from HUD. These current sources are not a 2021 census.
"""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HUD = "https://egis.hud.gov/arcgis/rest/services/hrl/HudResourceLocator/MapServer/1"
DMV = "https://www.usa.gov/state-motor-vehicle-services"
MANIFEST_FIELDS = ["downloaded_at_utc", "source", "state", "kind", "url",
                   "bytes", "sha256", "local_path", "notes"]
STATES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def fetch(url, destination, manifest, source, notes):
    if destination.exists():
        data = destination.read_bytes()
    else:
        last = None
        for attempt in range(3):
            try:
                req = Request(url, headers={"User-Agent": "public-service-research/2.0"})
                with urlopen(req, timeout=45) as response:
                    data = response.read()
                if not data:
                    raise ValueError("Empty response: " + url)
                with destination.open("xb") as handle:
                    handle.write(data)
                break
            except Exception as error:
                last = error
                if attempt == 2:
                    raise RuntimeError("Download failed: " + url) from last
                time.sleep(2 ** attempt)
    digest = hashlib.sha256(data).hexdigest()
    local = str(destination)
    known = set()
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != MANIFEST_FIELDS:
                raise ValueError("Unexpected download manifest schema")
            known = {(r["local_path"], r["sha256"]) for r in reader}
    if (local, digest) not in known:
        new = not manifest.exists() or not manifest.stat().st_size
        with manifest.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, MANIFEST_FIELDS, delimiter="\t", lineterminator="\n")
            if new:
                writer.writeheader()
            writer.writerow({
                "downloaded_at_utc": utcnow(), "source": source, "state": "National",
                "kind": "official_reference_v2", "url": url, "bytes": len(data),
                "sha256": digest, "local_path": local, "notes": notes,
            })
    print(json.dumps({"source": source, "file": destination.name, "bytes": len(data)}), flush=True)
    return data

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.text = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.text = []
    def handle_data(self, data):
        if self.href:
            self.text.append(data)
    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, " ".join(" ".join(self.text).split())))
            self.href = None

def write_csv_new(path, rows, fields):
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
        expected = [{k: str(r.get(k, "")) for k in fields} for r in rows]
        if previous != expected:
            raise ValueError("Refusing to replace differing normalized reference: " + str(path))
        return
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def run(output_dir, manifest):
    import re
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / "raw"
    raw.mkdir(exist_ok=True)
    now = utcnow()
    metadata = json.loads(fetch(HUD + "?f=json", raw / "hud-pha-layer.json",
                                manifest, "HUD Public Housing Authorities",
                                "Official agency-location layer; not housing buildings."))
    if metadata.get("name") != "Public Housing Authorities":
        raise ValueError("HUD layer changed")
    query = HUD + "/query?"
    ids_url = query + urlencode({"where": "1=1", "returnIdsOnly": "true", "f": "json"})
    ids_payload = json.loads(fetch(ids_url, raw / "hud-pha-ids.json", manifest,
                                   "HUD Public Housing Authorities", "Expected object ID universe."))
    if ids_payload.get("error"):
        raise ValueError(ids_payload["error"])
    ids = sorted(ids_payload["objectIds"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate HUD ID universe")
    rows, seen = [], set()
    # Keep GET URLs below conservative proxy length limits.
    for offset in range(0, len(ids), 100):
        batch = ids[offset:offset + 100]
        params = {"objectIds": ",".join(map(str, batch)), "f": "json",
                  "outFields": "OBJECTID,FORMAL_PARTICIPANT_NAME,STD_ADDR,STD_CITY,STD_ST,STD_ZIP5,HA_PROGRAM_TYPE",
                  "returnGeometry": "true", "outSR": "4326"}
        data = json.loads(fetch(query + urlencode(params), raw / ("hud-pha-%05d.json" % offset),
                                manifest, "HUD Public Housing Authorities",
                                "Public agency names/office addresses/coordinates only; current snapshot."))
        if data.get("error"):
            raise ValueError(data["error"])
        batch_seen = set()
        for feature in data.get("features", []):
            a, g = feature["attributes"], feature.get("geometry") or {}
            oid = a["OBJECTID"]
            if oid in seen:
                raise ValueError("Duplicate HUD object ID")
            seen.add(oid)
            batch_seen.add(oid)
            rows.append({
                "roster_id": "hud_pha_current", "source_record_id": str(oid),
                "entity_name": a.get("FORMAL_PARTICIPANT_NAME") or "",
                "address_line1": a.get("STD_ADDR") or "", "city": a.get("STD_CITY") or "",
                "state": a.get("STD_ST") or "", "zip": a.get("STD_ZIP5") or "",
                "latitude": g.get("y", ""), "longitude": g.get("x", ""),
                "service_family": "public_housing", "source_subtype": "Public Housing Authority",
                "source_vintage": now[:10], "source_url": HUD,
                "operator_evidence": "official_public_housing_agency",
                "record_grain": "agency_location_point",
                "public_access_status": "not_independently_verified",
            })
        if batch_seen != set(batch):
            raise ValueError("Incomplete HUD batch")
    if seen != set(ids):
        raise ValueError("Incomplete HUD download")
    rows.sort(key=lambda r: int(r["source_record_id"]))
    write_csv_new(output_dir / "hud-pha-current.csv", rows, list(rows[0]))

    html = fetch(DMV, raw / "usagov-state-motor-vehicle-services.html", manifest,
                 "USAGov State Motor Vehicle Services",
                 "Agency-routing directory; NOT a nationwide office-location roster.")
    parser = Links()
    parser.feed(html.decode("utf-8"))
    state_links = {}
    for href, label in parser.links:
        match = re.fullmatch(r"(.+?) \(([A-Z]{2})\)", label)
        if match and match[2] in STATES and href.startswith("https://"):
            state_links[match[2]] = {
                "state": match[2], "state_name": match[1], "agency_url": href,
                "source_url": DMV, "evidence_role": "state_agency_directory_not_facility_roster",
            }
    if set(state_links) != STATES:
        raise ValueError("State DMV directory incomplete: " + repr(STATES - set(state_links)))
    write_csv_new(output_dir / "state-dmv-agency-directory.csv",
                  [state_links[s] for s in sorted(state_links)], list(next(iter(state_links.values()))))
    summary = {
        "status": "complete", "completed_at_utc": utcnow(),
        "hud_pha_all_records": len(rows),
        "hud_pha_fifty_states_dc_records": sum(r["state"] in STATES for r in rows),
        "state_dmv_directory_entries": len(state_links),
        "limits": ["Current snapshots, not time-aligned 2021 denominators.",
                   "The DMV directory provides 51 official state/DC routing links, not 51 downloaded office rosters.",
                   "No human coding or model inference is used."],
    }
    with (output_dir / "fetch-summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.manifest)
