#!/usr/bin/env python3
"""Acquire listed public method papers and Crossref metadata without overwrites."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import quote

PROJECT = Path(__file__).resolve().parents[1]


def fetch(url, destination):
    result = subprocess.run(
        ["curl", "-fLsS", "--max-time", "40", "--max-filesize", "20000000",
         "--output", str(destination), url], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())


def acquire(item):
    archived = PROJECT / "sources/og" / (item["slug"] + ".pdf")
    metadata = PROJECT / "sources/metadata" / (item["slug"] + ".json")
    record = dict(item)
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        record.update(previous)
        record.update(item)
    record["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    if item.get("doi") and not record.get("crossref"):
        try:
            with tempfile.TemporaryDirectory(prefix="method-metadata-") as temporary:
                response = Path(temporary) / "response.json"
                fetch("https://api.crossref.org/works/" + quote(item["doi"], safe="/"), response)
                record["crossref"] = json.loads(response.read_text())["message"]
                record.pop("metadata_error", None)
        except (RuntimeError, KeyError, ValueError) as error:
            record["metadata_error"] = str(error)
    if archived.exists():
        metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        return item["key"], "already archived", None
    try:
        with tempfile.TemporaryDirectory(prefix="method-paper-") as temporary:
            downloaded = Path(temporary) / "paper.pdf"
            fetch(item["url"], downloaded)
            payload = downloaded.read_bytes()
            if not payload.startswith(b"%PDF-"):
                raise RuntimeError("Response is not a PDF; not admitted to the source library.")
            intake = PROJECT / "sources/unprocessed" / archived.name
            with intake.open("xb") as stream:
                stream.write(payload)
            intake.rename(archived)
        record["pdf_sha256"] = hashlib.sha256(payload).hexdigest()
        record["pdf_bytes"] = len(payload)
        record.pop("download_error", None)
        row = [record["checked_at_utc"], "Methodology source", "NA", "source-pdf",
               item["url"], len(payload), record["pdf_sha256"],
               str(archived.relative_to(PROJECT)), item["key"] + "; full-text verification pending"]
        status = "downloaded"
    except (RuntimeError, OSError) as error:
        record["download_error"] = str(error)
        row = None
        status = "unavailable: " + str(error)
    metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return item["key"], status, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path,
                        default=PROJECT / "sources/math-source-intake.json")
    parser.add_argument("--keys", nargs="+", help="Acquire only these listed keys")
    args = parser.parse_args()
    (PROJECT / "sources/metadata").mkdir(exist_ok=True)
    items = json.loads(args.intake.read_text())
    if args.keys:
        unknown = set(args.keys) - {item["key"] for item in items}
        if unknown:
            parser.error("Keys not in intake: " + ", ".join(sorted(unknown)))
        items = [item for item in items if item["key"] in args.keys]
    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(acquire, items))
    with (PROJECT / "data/download-manifest.tsv").open("a", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        for key, status, row in results:
            if row is not None:
                writer.writerow(row)
            print(key + ": " + status)


if __name__ == "__main__":
    main()
