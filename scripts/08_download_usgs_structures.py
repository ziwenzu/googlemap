#!/usr/bin/env python3
"""Download selected USGS National Structures layers without editing raw files.

The ArcGIS service caps feature responses, so this client first requests stable
object IDs and then downloads them in bounded batches.  Each completed layer is
an immutable gzip JSONL file.  Existing completed layers are verified and
skipped, which makes a national run resumable at the layer level.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SERVICE_URL = "https://carto.nationalmap.gov/arcgis/rest/services/structures/MapServer"
MANIFEST_FIELDS = [
    "downloaded_at_utc", "source", "state", "kind", "url", "bytes",
    "sha256", "local_path", "notes",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--manifest-prefix",
        default="data/external/official-rosters/raw/usgs-national-structures",
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--layer-ids", help="Comma-separated subset; default is every configured layer")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def request_json(url, params=None, retries=5):
    data = urlencode(params).encode("utf-8") if params else None
    request = Request(
        url,
        data=data,
        headers={"User-Agent": "informative-review-process/1.0"},
        method="POST" if data else "GET",
    )
    last_error = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=180) as response:
                payload = json.load(response)
            if "error" in payload:
                raise RuntimeError(json.dumps(payload["error"], sort_keys=True))
            return payload
        except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}: {last_error}")


def load_layers(path, selected_ids=None):
    layers = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            layer_id = int(row["layer_id"])
            if layer_id in seen:
                raise ValueError(f"Duplicate layer_id in config: {layer_id}")
            seen.add(layer_id)
            if selected_ids is None or layer_id in selected_ids:
                layers.append({**row, "layer_id": layer_id})
    if selected_ids is not None:
        missing = selected_ids - {row["layer_id"] for row in layers}
        if missing:
            raise ValueError(f"Requested layer IDs absent from config: {sorted(missing)}")
    if not layers:
        raise ValueError("No USGS layers selected")
    return layers


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attribute_value(attributes, name):
    return (
        attributes.get(name)
        if name in attributes
        else attributes.get(name.lower(), attributes.get(name.upper()))
    )


def verify_existing(path):
    count = 0
    object_ids = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            object_id = attribute_value(row["feature"]["attributes"], "OBJECTID")
            if object_id is None or object_id in object_ids:
                raise ValueError(f"Missing or duplicate OBJECTID in {path}: {object_id}")
            object_ids.add(object_id)
            count += 1
    return count


def object_ids(layer_id, retries):
    payload = request_json(
        f"{SERVICE_URL}/{layer_id}/query",
        {"where": "1=1", "returnIdsOnly": "true", "f": "json"},
        retries,
    )
    return sorted(payload.get("objectIds") or [])


def download_layer(layer, output_dir, batch_size, retries, force=False):
    layer_id = layer["layer_id"]
    metadata = request_json(f"{SERVICE_URL}/{layer_id}?f=pjson", retries=retries)
    if metadata.get("name") != layer["layer_name"]:
        raise ValueError(
            f"Layer {layer_id} changed: expected {layer['layer_name']!r}, "
            f"service returned {metadata.get('name')!r}"
        )
    ids = object_ids(layer_id, retries)
    destination = output_dir / f"usgs-national-structures-layer-{layer_id:02d}.jsonl.gz"
    if destination.exists() and not force:
        existing_count = verify_existing(destination)
        if existing_count != len(ids):
            raise ValueError(
                f"Existing layer {destination} has {existing_count} rows but service now has "
                f"{len(ids)}. Preserve it as a dated snapshot or rerun with --force."
            )
        return destination, existing_count, "verified_existing"

    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    downloaded_ids = set()
    fetched_at = utc_now()
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for batch in chunks(ids, batch_size):
            payload = request_json(
                f"{SERVICE_URL}/{layer_id}/query",
                {
                    "objectIds": ",".join(str(value) for value in batch),
                    "outFields": "*",
                    "returnGeometry": "true",
                    "outSR": "4326",
                    "f": "json",
                },
                retries,
            )
            features = payload.get("features") or []
            features.sort(
                key=lambda item: attribute_value(item.get("attributes", {}), "OBJECTID") or -1
            )
            for feature in features:
                object_id = attribute_value(feature.get("attributes", {}), "OBJECTID")
                if object_id is None or object_id in downloaded_ids:
                    raise ValueError(f"Missing or duplicate OBJECTID from layer {layer_id}: {object_id}")
                downloaded_ids.add(object_id)
                handle.write(
                    json.dumps(
                        {
                            "source_service": SERVICE_URL,
                            "source_layer_id": layer_id,
                            "source_layer_name": layer["layer_name"],
                            "downloaded_at_utc": fetched_at,
                            "feature": feature,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    if downloaded_ids != set(ids):
        missing = sorted(set(ids) - downloaded_ids)
        extra = sorted(downloaded_ids - set(ids))
        raise ValueError(
            f"Incomplete layer {layer_id}: missing={missing[:10]} extra={extra[:10]}"
        )
    os.replace(temporary, destination)
    return destination, len(downloaded_ids), "downloaded"


def manifest_has_row(path, local_path, file_sha):
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        return any(
            row.get("local_path") == local_path and row.get("sha256") == file_sha
            for row in csv.DictReader(handle, delimiter="\t")
        )


def record_manifest(manifest, manifest_prefix, layer, path, count):
    file_sha = sha256(path)
    local_path = f"{manifest_prefix.rstrip('/')}/{path.name}"
    if manifest_has_row(manifest, local_path, file_sha):
        return file_sha
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MANIFEST_FIELDS, delimiter="\t", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "downloaded_at_utc": utc_now(),
                "source": "USGS National Structures Dataset",
                "state": "National",
                "kind": f"official_roster:usgs_nsd_layer_{layer['layer_id']}",
                "url": f"{SERVICE_URL}/{layer['layer_id']}",
                "bytes": path.stat().st_size,
                "sha256": file_sha,
                "local_path": local_path,
                "notes": f"ArcGIS feature extract; {count} records; current discovery snapshot",
            }
        )
    return file_sha


def main():
    args = parse_args()
    if args.batch_size < 1 or args.batch_size > 2000:
        raise SystemExit("--batch-size must be between 1 and 2000")
    selected = None
    if args.layer_ids:
        selected = {int(value.strip()) for value in args.layer_ids.split(",") if value.strip()}
    layers = load_layers(args.layers_config, selected)
    service_metadata = request_json(f"{SERVICE_URL}?f=pjson", retries=args.retries)
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        ids = object_ids(layer["layer_id"], args.retries) if args.check else None
        if args.check:
            metadata = request_json(
                f"{SERVICE_URL}/{layer['layer_id']}?f=pjson", retries=args.retries
            )
            if metadata.get("name") != layer["layer_name"]:
                raise SystemExit(
                    f"Layer {layer['layer_id']} name mismatch: {metadata.get('name')!r}"
                )
            result = {
                "layer_id": layer["layer_id"],
                "layer_name": layer["layer_name"],
                "service_family": layer["service_family"],
                "record_count": len(ids),
                "status": "checked",
            }
        else:
            path, count, status = download_layer(
                layer, args.output_dir, args.batch_size, args.retries, args.force
            )
            file_sha = record_manifest(
                args.manifest, args.manifest_prefix, layer, path, count
            ) if args.manifest else sha256(path)
            result = {
                "layer_id": layer["layer_id"],
                "layer_name": layer["layer_name"],
                "service_family": layer["service_family"],
                "record_count": count,
                "status": status,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha,
            }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "source": "USGS National Structures Dataset",
        "service_url": SERVICE_URL,
        "service_current_version": service_metadata.get("currentVersion"),
        "retrieved_at_utc": utc_now(),
        "role": "current facility discovery; not a time-aligned 2021 denominator",
        "layers": results,
        "total_records": sum(row["record_count"] for row in results),
    }
    if args.summary and not args.check:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
