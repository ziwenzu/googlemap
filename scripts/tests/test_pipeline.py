import csv
import gzip
import json
import subprocess
import tempfile
import unittest
import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation"))

from anchor_transport import invert_observed, relative_selection


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official_rosters = load_script("official_rosters", "06_normalize_official_rosters.py")
library_linkage = load_script("library_linkage", "07_link_public_libraries.py")
usgs_download = load_script("usgs_download", "08_download_usgs_structures.py")
usgs_normalize = load_script("usgs_normalize", "09_normalize_usgs_structures.py")
usgs_linkage = load_script("usgs_linkage", "10_link_usgs_to_google.py")
place_validation = load_script("place_validation", "11_draw_place_validation_sample.py")


class PipelineSmokeTest(unittest.TestCase):
    def test_filter_extract_and_summarize(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            metadata = tmp / "meta.json.gz"
            reviews = tmp / "reviews.json.gz"
            places = tmp / "places.csv"
            subset = tmp / "subset.jsonl.gz"
            panel = tmp / "panel.csv"

            with gzip.open(metadata, "wt", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "name": "Example Department of Motor Vehicles",
                            "address": "1 Main St",
                            "gmap_id": "gov-1",
                            "description": "Driver licensing office",
                            "latitude": 44.0,
                            "longitude": -72.0,
                            "category": ["Department of motor vehicles"],
                            "avg_rating": 2.5,
                            "num_of_reviews": 2,
                        }
                    )
                    + "\n"
                )
                handle.write(
                    json.dumps(
                        {
                            "name": "Example Driving School",
                            "address": "2 Main St",
                            "gmap_id": "private-1",
                            "category": ["Driving school"],
                        }
                    )
                    + "\n"
                )

            with gzip.open(reviews, "wt", encoding="utf-8") as handle:
                for rating, when in [(1, 1609459200000), (5, 1612137600000)]:
                    handle.write(
                        json.dumps(
                            {
                                "gmap_id": "gov-1",
                                "user_id": f"user-{rating}",
                                "name": "Do not retain",
                                "time": when,
                                "rating": rating,
                                "text": "Service encounter",
                            }
                        )
                        + "\n"
                    )

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/01_filter_government_places.py"),
                    "--metadata",
                    str(metadata),
                    "--taxonomy",
                    str(ROOT / "config/government-place-taxonomy.csv"),
                    "--output",
                    str(places),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with places.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["gmap_id"] for row in rows], ["gov-1"])
            self.assertEqual(rows[0]["source_partition"], "meta")
            self.assertIn("name", rows[0]["matched_fields"])
            self.assertEqual(rows[0]["address_only_match"], "no")

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/02_extract_reviews.py"),
                    "--reviews",
                    str(reviews),
                    "--places",
                    str(places),
                    "--output",
                    str(subset),
                    "--salt-file",
                    str(tmp / "salt"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with gzip.open(subset, "rt", encoding="utf-8") as handle:
                safe = [json.loads(line) for line in handle]
            self.assertEqual(len(safe), 2)
            self.assertNotIn("name", safe[0])
            self.assertNotIn("user_id", safe[0])

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/03_summarize_place_month.py"),
                    "--reviews",
                    str(subset),
                    "--places",
                    str(places),
                    "--output",
                    str(panel),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with panel.open(newline="", encoding="utf-8") as handle:
                panel_rows = list(csv.DictReader(handle))
            self.assertEqual(len(panel_rows), 2)

    def test_anchor_transport_recovers_latent_distribution(self):
        anchor_latent = [0.10, 0.15, 0.20, 0.25, 0.30]
        target_latent = [0.20, 0.20, 0.20, 0.20, 0.20]
        selection = [2.0, 1.0, 0.5, 1.5, 3.0]

        def observed(latent):
            weighted = [value * factor for value, factor in zip(latent, selection)]
            total = sum(weighted)
            return [value / total for value in weighted]

        factors = relative_selection(observed(anchor_latent), anchor_latent)
        recovered = invert_observed(observed(target_latent), factors)
        for actual, expected in zip(recovered, target_latent):
            self.assertAlmostEqual(actual, expected)

    def test_national_candidate_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            places = tmp / "places.csv"
            report = tmp / "report.md"
            fieldnames = [
                "source_partition",
                "gmap_id",
                "name",
                "latitude",
                "longitude",
                "service_family",
                "all_service_families",
                "match_priority",
                "matched_fields",
                "address_only_match",
                "classification_status",
                "manual_label",
            ]
            rows = [
                {
                    "source_partition": "Alpha",
                    "gmap_id": "gov-1",
                    "name": "Example Office",
                    "latitude": "1",
                    "longitude": "2",
                    "service_family": "motor_vehicles",
                    "all_service_families": "motor_vehicles",
                    "match_priority": "1",
                    "matched_fields": "name",
                    "address_only_match": "no",
                    "classification_status": "rule_candidate",
                    "manual_label": "",
                },
                {
                    "source_partition": "Beta",
                    "gmap_id": "gov-1",
                    "name": "Example Office",
                    "latitude": "1",
                    "longitude": "2",
                    "service_family": "motor_vehicles",
                    "all_service_families": "motor_vehicles;other_government",
                    "match_priority": "1",
                    "matched_fields": "address",
                    "address_only_match": "yes",
                    "classification_status": "rule_candidate",
                    "manual_label": "",
                },
            ]
            with places.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/05_audit_national_candidates.py"),
                    "--places",
                    str(places),
                    "--output",
                    str(report),
                    "--reported-source-places",
                    "10",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            content = report.read_text(encoding="utf-8")
            self.assertIn("| Candidate rows | 2 |", content)
            self.assertIn("| Duplicate `gmap_id` values | 1 |", content)
            self.assertIn("| Address-only match rows | 1 (50.00%) |", content)

    def test_official_roster_normalization_preserves_record_grain(self):
        census = official_rosters.census_row(
            {
                "CENSUS_ID_PID6": "123456",
                "CENSUS_ID_GIDID": "gid",
                "UNIT_NAME": "Example County",
                "UNIT_TYPE": "1 - COUNTY",
                "ADDRESS1": "1 Main St",
                "CITY": "Example",
                "STATE": "VT",
                "FIPS_STATE": "50",
                "ZIP": "05001",
                "IS_ACTIVE": "Y",
            },
            "General Purpose",
        )
        self.assertEqual(census["record_grain"], "government_organization")
        self.assertEqual(census["public_access_status"], "not_a_facility_record")

        imls = official_rosters.imls_row(
            {
                "LIBID": "VT0001-001",
                "FSCSKEY": "VT0001",
                "FSCS_SEQ": "001",
                "LIBNAME": "Example Public Library",
                "ADDRESS": "2 Main St",
                "CITY": "Example",
                "STABR": "VT",
                "ZIP": "05001",
                "C_OUT_TY": "CE",
                "STATSTRU": "00",
                "LATITUDE": "44.0",
                "LONGITUD": "-72.0",
            },
            {"VT0001": "Example Library System"},
        )
        self.assertEqual(imls["record_grain"], "public_library_service_outlet")
        self.assertEqual(imls["source_record_id"], "VT0001:001")
        self.assertEqual(imls["public_access_status"], "official_public_service_outlet")
        self.assertEqual(imls["parent_entity_name"], "Example Library System")

        gsa = official_rosters.gsa_row(
            {
                "Real Property Unique Identifier": "asset-1",
                "Reporting Agency": "GENERAL SERVICES ADMINISTRATION",
                "Reporting Agency Code": "47",
                "Reporting Bureau Code": "01",
                "Installation Name": "Example Federal Building",
                "Street Address": "3 Main St",
                "City Name": "Example",
                "State Code": "50",
                "Zip Code": "05001-1234",
                "Asset Status": "Current Mission Need",
                "Real Property Use": "Office",
            }
        )
        self.assertEqual(gsa["record_grain"], "federal_real_property_asset")
        self.assertEqual(gsa["source_record_id"], "47:01:asset-1")
        self.assertEqual(gsa["state"], "VT")
        self.assertEqual(gsa["public_access_status"], "unknown_asset_not_public_access_evidence")
        self.assertEqual(gsa["zip"], "05001")

    def test_library_linkage_requires_agreement(self):
        candidate = {
            "name": "Example Public Library",
            "address": "Example Public Library, 2 Main St, Example, VT 05001",
            "latitude": "44.0001",
            "longitude": "-72.0001",
        }
        official = {
            "entity_name": "EXAMPLE PUBLIC LIBRARY",
            "parent_entity_name": "EXAMPLE LIBRARY SYSTEM",
            "address_line1": "2 MAIN STREET",
            "latitude": "44.0",
            "longitude": "-72.0",
            "zip": "05001",
        }
        score = library_linkage.score_match(candidate, official)
        self.assertEqual(score["status"], "confirmed_official_candidate")
        self.assertLess(score["distance_km"], 0.02)
        collision = library_linkage.match_fields(score, official, multiplicity=2)
        self.assertEqual(collision["official_match_status"], "needs_review")
        self.assertEqual(collision["official_match_method"], "ambiguous_many_to_one_official_record")

        wrong = dict(official)
        wrong.update({"entity_name": "OTHER LIBRARY", "address_line1": "900 FAR ROAD", "latitude": "45", "longitude": "-73"})
        wrong_score = library_linkage.score_match(candidate, wrong)
        self.assertEqual(wrong_score["status"], "not_linked")

    def test_usgs_normalization_handles_lowercase_arcgis_fields(self):
        raw = {
            "source_layer_id": 39,
            "downloaded_at_utc": "2026-09-21T00:00:00Z",
            "feature": {
                "attributes": {
                    "objectid": 10,
                    "permanent_identifier": "place-10",
                    "name": "Example Town Hall",
                    "admintype": 6,
                    "address": "1 Main St",
                    "city": "Example",
                    "state": "VT",
                    "zipcode": "05001",
                },
                "geometry": {"x": -72.0, "y": 44.0},
            },
        }
        layer = {
            "layer_name": "City/Town Halls",
            "service_family": "city_hall",
            "analytic_tier_hint": "core",
            "ownership_rule": "non_private",
        }
        row = usgs_normalize.normalize_row(raw, layer)
        self.assertEqual(row["source_record_id"], "place-10")
        self.assertEqual(row["entity_name"], "Example Town Hall")
        self.assertEqual(row["official_owner_class"], "municipal")
        self.assertEqual(row["owner_eligibility"], "eligible_public_owner")
        self.assertEqual(row["latitude"], "44.0000000")

        raw["feature"]["attributes"]["admintype"] = None
        uncertain = usgs_normalize.normalize_row(raw, layer)
        self.assertEqual(uncertain["official_owner_class"], "uncertain")
        self.assertEqual(uncertain["owner_eligibility"], "needs_review_unknown_owner")

    def test_usgs_linkage_requires_service_evidence_for_confirmation(self):
        official = {
            "entity_name": "Example Town Hall",
            "address_line1": "1 Main St",
            "latitude": "44.0",
            "longitude": "-72.0",
            "service_family": "city_hall",
        }
        good = {
            "name": "Example Town Clerk",
            "address": "1 Main St, Example, VT",
            "latitude": "44.0001",
            "longitude": "-72.0001",
            "category": ["City government office"],
        }
        score = usgs_linkage.score_match(good, official)
        self.assertEqual(score["status"], "confirmed_official_candidate")

        colocated_private = {
            "name": "Example Dental",
            "address": "9 Other St, Example, VT",
            "latitude": "44.0001",
            "longitude": "-72.0001",
            "category": ["Dentist"],
        }
        private_score = usgs_linkage.score_match(colocated_private, official)
        self.assertTrue(
            private_score is None or private_score["status"] == "needs_review"
        )

        unrelated = {
            "name": "Mountain Dental",
            "address": "9 Other St, Example, VT",
            "latitude": "44.0001",
            "longitude": "-72.0001",
            "category": ["Dentist"],
        }
        self.assertIsNone(usgs_linkage.score_match(unrelated, official))

    def test_usgs_helpers_and_validation_ids_are_stable(self):
        self.assertEqual(usgs_download.attribute_value({"objectid": 7}, "OBJECTID"), 7)
        first = place_validation.sample_id("general_google_negative", "abc")
        second = place_validation.sample_id("general_google_negative", "abc")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)


if __name__ == "__main__":
    unittest.main()
