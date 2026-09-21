#!/usr/bin/env python3
"""Normalize official U.S. government rosters without altering source archives.

The three inputs have different observational units.  Census records are
government organizations, IMLS records are public-library service outlets, and
GSA records are federal real-property assets.  The common schema therefore
keeps ``record_grain`` and ``public_access_status`` explicit rather than
silently treating every official record as a public-facing office.
"""

import argparse
import csv
import gzip
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

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
]

SOURCES = {
    "census": {
        "roster_id": "census_gov_units_2021",
        "filename": "census-government-units-2021.zip",
        "url": "https://www2.census.gov/programs-surveys/gus/datasets/2021/govt_units_2021.ZIP",
        "vintage": "2021",
    },
    "imls": {
        "roster_id": "imls_pls_outlets_2021",
        "filename": "imls-pls-fy2021-csv.zip",
        "url": "https://www.imls.gov/sites/default/files/2023-06/pls_fy2021_csv.zip",
        "vintage": "FY2021",
    },
    "gsa": {
        "roster_id": "gsa_frpp_2021",
        "filename": "gsa-frpp-fy2021.xlsx",
        "url": "https://inventory.data.gov/dataset/ba12ef9a-2a84-4da7-a943-8772c176d5ff/resource/ec1ee26a-d1c7-4c53-87b1-58484cf948e5/download/frpp_public_dataset_fy21_final_02242023-1.xlsx",
        "vintage": "FY2021",
    },
}

FIPS_STATE_ABBREVIATIONS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP",
    "72": "PR", "74": "UM", "78": "VI",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def clean(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_coordinate(value, minimum, maximum):
    text = clean(value)
    try:
        number = float(text)
    except (TypeError, ValueError):
        return ""
    if minimum <= number <= maximum:
        return text
    return ""


def excel_column_index(reference):
    letters = re.match(r"[A-Z]+", reference or "")
    if not letters:
        return None
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def load_shared_strings(book):
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    strings = []
    with book.open("xl/sharedStrings.xml") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag == f"{{{NS_MAIN}}}si":
                strings.append("".join(node.text or "" for node in element.iter(f"{{{NS_MAIN}}}t")))
                element.clear()
    return strings


def workbook_sheets(book):
    workbook = ET.parse(book.open("xl/workbook.xml")).getroot()
    relationships = ET.parse(book.open("xl/_rels/workbook.xml.rels")).getroot()
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{NS_PKG_REL}}}Relationship")
    }
    sheets = []
    for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
        rel_id = sheet.attrib[f"{{{NS_REL}}}id"]
        target = targets[rel_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets.append((sheet.attrib["name"], target))
    return sheets


def xlsx_rows(book, sheet_path, shared_strings):
    """Yield worksheet rows as lists while keeping memory bounded."""
    with book.open(sheet_path) as handle:
        for _, row in ET.iterparse(handle, events=("end",)):
            if row.tag != f"{{{NS_MAIN}}}row":
                continue
            values = []
            for cell in row.findall(f"{{{NS_MAIN}}}c"):
                index = excel_column_index(cell.attrib.get("r", ""))
                if index is None:
                    continue
                while len(values) <= index:
                    values.append("")
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{NS_MAIN}}}v")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{{{NS_MAIN}}}t")
                    )
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "b":
                    value = "TRUE" if value_node.text == "1" else "FALSE"
                else:
                    value = value_node.text or ""
                values[index] = value
            yield values
            row.clear()


def row_dict(header, row):
    return {name: row[index] if index < len(row) else "" for index, name in enumerate(header)}


def census_row(source, sheet_name):
    source_type = clean(source.get("UNIT_TYPE") or source.get("FUNCTION_NAME") or source.get("SCHOOL_LEVEL_DESCRIPTION"))
    if sheet_name == "General Purpose":
        service_family = "general_government"
    elif sheet_name == "Special District":
        service_family = "special_district"
    else:
        service_family = "education"
    return {
        "roster_id": SOURCES["census"]["roster_id"],
        "source_record_id": clean(source.get("CENSUS_ID_PID6")),
        "source_subrecord_id": clean(source.get("CENSUS_ID_GIDID")),
        "record_grain": "government_organization",
        "evidence_role": "organization_denominator_and_validation",
        "government_level": "state_or_local",
        "service_family": service_family,
        "entity_name": clean(source.get("UNIT_NAME")),
        "parent_entity_name": "",
        "address_line1": clean(source.get("ADDRESS1")),
        "address_line2": clean(source.get("ADDRESS2")),
        "city": clean(source.get("CITY")),
        "state": clean(source.get("STATE")) or FIPS_STATE_ABBREVIATIONS.get(clean(source.get("FIPS_STATE")).zfill(2), ""),
        "zip": clean(source.get("ZIP")),
        "zip4": clean(source.get("ZIP4")),
        "county": clean(source.get("COUNTY_AREA_NAME")),
        "latitude": "",
        "longitude": "",
        "active_status": "active" if clean(source.get("IS_ACTIVE")).upper() == "Y" else "inactive_or_unknown",
        "public_access_status": "not_a_facility_record",
        "source_subtype": f"{sheet_name}: {source_type}" if source_type else sheet_name,
        "source_vintage": SOURCES["census"]["vintage"],
        "source_file": SOURCES["census"]["filename"],
        "source_url": SOURCES["census"]["url"],
        "redistribution_status": "public_use",
    }


def normalize_census(raw_dir, output_dir):
    source = SOURCES["census"]
    archive_path = raw_dir / source["filename"]
    output_path = output_dir / "census-government-organizations-2021.csv.gz"
    count = 0
    statuses = Counter()
    with zipfile.ZipFile(archive_path) as outer:
        workbook_name = next(name for name in outer.namelist() if name.lower().endswith(".xlsx"))
        with zipfile.ZipFile(io.BytesIO(outer.read(workbook_name))) as book, gzip.open(
            output_path, "wt", newline="", encoding="utf-8"
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            shared = load_shared_strings(book)
            for sheet_name, sheet_path in workbook_sheets(book):
                rows = xlsx_rows(book, sheet_path, shared)
                header = [clean(value) for value in next(rows)]
                for values in rows:
                    normalized = census_row(row_dict(header, values), sheet_name)
                    writer.writerow(normalized)
                    count += 1
                    statuses[normalized["active_status"]] += 1
    return output_path, count, statuses


def imls_row(source, system_names):
    outlet_types = {
        "CE": "central_library",
        "BR": "branch_library",
        "BS": "bookmobile",
        "BM": "books_by_mail",
    }
    status = "temporarily_closed" if clean(source.get("STATSTRU")) == "23" else "reported_fy2021"
    return {
        "roster_id": SOURCES["imls"]["roster_id"],
        "source_record_id": f"{clean(source.get('FSCSKEY'))}:{clean(source.get('FSCS_SEQ'))}",
        "source_subrecord_id": clean(source.get("LIBID")),
        "record_grain": "public_library_service_outlet",
        "evidence_role": "public_facility_denominator_and_validation",
        "government_level": "local_or_intergovernmental",
        "service_family": "public_library",
        "entity_name": clean(source.get("LIBNAME")),
        "parent_entity_name": clean(system_names.get(clean(source.get("FSCSKEY")), "")),
        "address_line1": clean(source.get("ADDRESS")),
        "address_line2": "",
        "city": clean(source.get("CITY")),
        "state": clean(source.get("STABR")),
        "zip": clean(source.get("ZIP")),
        "zip4": clean(source.get("ZIP4")),
        "county": clean(source.get("CNTY")),
        "latitude": safe_coordinate(source.get("LATITUDE"), -90, 90),
        "longitude": safe_coordinate(source.get("LONGITUD"), -180, 180),
        "active_status": status,
        "public_access_status": "official_public_service_outlet",
        "source_subtype": outlet_types.get(clean(source.get("C_OUT_TY")), clean(source.get("C_OUT_TY")) or "unknown"),
        "source_vintage": SOURCES["imls"]["vintage"],
        "source_file": SOURCES["imls"]["filename"],
        "source_url": SOURCES["imls"]["url"],
        "redistribution_status": "public_use",
    }


def normalize_imls(raw_dir, output_dir):
    source = SOURCES["imls"]
    archive_path = raw_dir / source["filename"]
    output_path = output_dir / "imls-public-library-outlets-fy2021.csv.gz"
    with zipfile.ZipFile(archive_path) as archive:
        ae_name = next(name for name in archive.namelist() if "ae_" in name.lower() and name.lower().endswith(".csv"))
        outlet_name = next(name for name in archive.namelist() if "outlet" in name.lower() and name.lower().endswith(".csv"))
        with io.TextIOWrapper(archive.open(ae_name), encoding="latin-1", newline="") as handle:
            system_names = {
                clean(row.get("FSCSKEY")): clean(row.get("LIBNAME"))
                for row in csv.DictReader(handle)
            }
        count = 0
        statuses = Counter()
        with io.TextIOWrapper(archive.open(outlet_name), encoding="latin-1", newline="") as source_handle, gzip.open(
            output_path, "wt", newline="", encoding="utf-8"
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            for source_row in csv.DictReader(source_handle):
                normalized = imls_row(source_row, system_names)
                writer.writerow(normalized)
                count += 1
                statuses[normalized["active_status"]] += 1
    return output_path, count, statuses


def classify_gsa_service(source):
    searchable = " ".join(
        clean(source.get(field)).upper()
        for field in (
            "Reporting Agency",
            "Reporting Bureau",
            "Using Agency",
            "Using Bureau",
            "Installation Name",
            "Real Property Use",
        )
    )
    if "SOCIAL SECURITY" in searchable:
        return "social_security"
    if "INTERNAL REVENUE" in searchable or "TAXPAYER" in searchable:
        return "tax_revenue"
    if "VETERANS AFFAIRS" in searchable:
        return "veterans"
    if "POSTAL SERVICE" in searchable or "POST OFFICE" in searchable:
        return "postal_service"
    if "COURT" in searchable:
        return "court"
    if "VISITOR CENTER" in searchable or "NATIONAL PARK" in searchable:
        return "parks_recreation"
    return "other_government"


def gsa_row(source):
    status_text = clean(source.get("Asset Status"))
    status_code = clean(source.get("Asset Status Code"))
    if status_code == "D" or status_text.lower() == "disposed":
        status = "disposed"
    elif status_text:
        status = re.sub(r"[^a-z0-9]+", "_", status_text.lower()).strip("_")
    else:
        status = "unknown"
    agency = clean(source.get("Using Agency") or source.get("Reporting Agency"))
    bureau = clean(source.get("Using Bureau") or source.get("Reporting Bureau"))
    reporting_agency_code = clean(source.get("Reporting Agency Code"))
    reporting_bureau_code = clean(source.get("Reporting Bureau Code"))
    native_id = clean(source.get("Real Property Unique Identifier"))
    return {
        "roster_id": SOURCES["gsa"]["roster_id"],
        "source_record_id": f"{reporting_agency_code}:{reporting_bureau_code}:{native_id}",
        "source_subrecord_id": clean(source.get("Installation Id")),
        "record_grain": "federal_real_property_asset",
        "evidence_role": "federal_asset_discovery_and_validation",
        "government_level": "federal",
        "service_family": classify_gsa_service(source),
        "entity_name": clean(source.get("Installation Name") or bureau or agency),
        "parent_entity_name": " / ".join(value for value in (agency, bureau) if value),
        "address_line1": clean(source.get("Street Address")),
        "address_line2": "",
        "city": clean(source.get("City Name")),
        "state": FIPS_STATE_ABBREVIATIONS.get(clean(source.get("State Code")).zfill(2), ""),
        "zip": clean(source.get("Zip Code"))[:5],
        "zip4": "",
        "county": clean(source.get("County Name")),
        "latitude": safe_coordinate(source.get("Latitude"), -90, 90),
        "longitude": safe_coordinate(source.get("Longitude"), -180, 180),
        "active_status": status,
        "public_access_status": "unknown_asset_not_public_access_evidence",
        "source_subtype": clean(source.get("Real Property Use") or source.get("Real Property Type")),
        "source_vintage": SOURCES["gsa"]["vintage"],
        "source_file": SOURCES["gsa"]["filename"],
        "source_url": SOURCES["gsa"]["url"],
        "redistribution_status": "cc0_public_domain",
    }


def normalize_gsa(raw_dir, output_dir):
    source = SOURCES["gsa"]
    source_path = raw_dir / source["filename"]
    output_path = output_dir / "gsa-federal-real-property-assets-fy2021.csv.gz"
    count = 0
    statuses = Counter()
    with zipfile.ZipFile(source_path) as book, gzip.open(
        output_path, "wt", newline="", encoding="utf-8"
    ) as destination:
        shared = load_shared_strings(book)
        sheet_name, sheet_path = workbook_sheets(book)[0]
        rows = xlsx_rows(book, sheet_path, shared)
        header = [clean(value) for value in next(rows)]
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        for values in rows:
            normalized = gsa_row(row_dict(header, values))
            writer.writerow(normalized)
            count += 1
            statuses[normalized["active_status"]] += 1
    return output_path, count, statuses


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCES.values():
        path = args.raw_dir / source["filename"]
        if not path.exists():
            raise SystemExit(f"Missing official roster source: {path}")

    results = {}
    for key, normalizer in (
        ("census", normalize_census),
        ("imls", normalize_imls),
        ("gsa", normalize_gsa),
    ):
        path, count, statuses = normalizer(args.raw_dir, args.output_dir)
        results[key] = {
            "roster_id": SOURCES[key]["roster_id"],
            "rows": count,
            "active_status_counts": dict(sorted(statuses.items())),
            "output": str(path),
        }
        print(json.dumps({key: results[key]}, indent=2))

    summary_path = args.output_dir / "official-roster-normalization-summary-2021.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
