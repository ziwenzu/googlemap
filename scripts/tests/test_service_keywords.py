import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("service_keywords", ROOT / "scripts/13_extract_public_service_places.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)
RULES = ROOT / "config/public-service-keywords-v1.json"

class AutomaticServiceTests(unittest.TestCase):
    def setUp(self):
        self.rules = service.load_rules(RULES)

    def test_eight_service_families(self):
        examples = [
            ("Vermont DMV - Springfield", [], "motor_vehicles"),
            ("Example Sheriff's Office", [], "police"),
            ("United States Postal Service", [], "postal_service"),
            ("Social Security Administration", [], "social_security"),
            ("County Department of Human Services", [], "human_services"),
            ("Internal Revenue Service", [], "tax_revenue"),
            ("County Superior Court", [], "court"),
            ("County Clerk", [], "civic_administration"),
            ("Service Branch 7", ["Police department"], "police"),
            ("Licensing", ["Driver's license office"], "motor_vehicles"),
            ("Employment", ["State employment department"], "human_services"),
            ("Sherriff branch", ["Sheriff\u2019s department"], "police"),
        ]
        for name, categories, family in examples:
            with self.subTest(name=name):
                families, hits, excludes = service.classify({"name": name, "category": categories}, self.rules)
                self.assertIn(family, families)
                self.assertTrue(hits)
                self.assertEqual(excludes, [])

    def test_no_address_or_description_and_no_generic_government_category(self):
        for record in [
            {"name": "Coffee", "address": "1 Post Office Road", "description": "Near the DMV"},
            {"name": "Administration Building", "category": ["Government office"]},
            {"name": "Department of Motor Vehicles Museum", "category": []},
            {"name": "UVM Medical Center", "category": ["Hospital"]},
            {"name": "Town Library", "category": ["Public library"]},
        ]:
            families, _, excludes = service.classify(record, self.rules)
            self.assertTrue(not families or excludes)

    def test_category_allowlist_is_exact_and_acronyms_use_boundaries(self):
        for record in [
            {"name": "Example", "category": ["Former Post office"]},
            {"name": "NotDMVservices", "category": []},
        ]:
            self.assertFalse(service.classify(record, self.rules)[0])

    def test_exclusions_override_positive_matches(self):
        cases = [
            ("The UPS Store", ["Post office"]),
            ("FedEx Post Office", []),
            ("Social Security Disability Lawyer", ["Social security office"]),
            ("DMV Driving School", ["Department of motor vehicles"]),
            ("City Hall Cafe", ["City Hall"]),
            ("Police Museum", ["Police station"]),
            ("DMV Pizza", []),
            ("USPS Processing and Distribution Center", ["Post office"]),
            ("USPS Collection Box", []),
            ("College Police Department", []),
            ("Old Post Office", ["Historical landmark"]),
            ("Tag Shop DMV", ["Auto tag agency"]),
            ("County Court", ["Prison"]),
        ]
        for name, categories in cases:
            with self.subTest(name=name):
                families, _, excludes = service.classify({"name": name, "category": categories}, self.rules)
                self.assertTrue(families)
                self.assertTrue(excludes)

    def test_stream_deduplicate_other_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            records = {
                "Vermont": [
                    {"gmap_id": "a", "name": "Example DMV", "category": []},
                    {"gmap_id": "x", "name": "DMV Driving School", "category": []},
                    {"gmap_id": "", "name": "Town Hall", "category": []},
                    {"gmap_id": "z", "name": "Cafe", "address": "Post Office Road", "category": []},
                ],
                "New_Hampshire": [{"gmap_id": "a", "name": "Example DMV", "category": []}],
                "other": [
                    {"gmap_id": "a", "name": "Example DMV", "category": []},
                    {"gmap_id": "b", "name": "Other Post Office", "category": []},
                ],
            }
            for partition, rows in records.items():
                with gzip.open(tmp / ("meta-%s.json.gz" % partition), "wt", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
            output = tmp / "out"
            summary = service.run(tmp, RULES, output, list(records))
            self.assertEqual(summary["primary_unique_places"], 1)
            self.assertEqual(summary["other_only_unique_places"], 1)
            self.assertEqual(summary["other_overlap_primary"], 1)
            self.assertFalse(summary["human_coding_used"])
            with (output / "places.csv").open(newline="") as f:
                selected = list(csv.DictReader(f))
            self.assertEqual(selected[0]["gmap_id"], "a")
            self.assertEqual(selected[0]["source_partitions"], "New_Hampshire;Vermont;other")
            self.assertEqual(selected[0]["adjudicated_label"], "")
            self.assertEqual(selected[0]["classification_status"], "rule_selected")
            with (output / "matched-exclusions.csv").open(newline="") as f:
                excluded = list(csv.DictReader(f))
            self.assertEqual(len(excluded), 2)
            before = (output / "places.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                service.run(tmp, RULES, output, list(records))
            self.assertEqual(before, (output / "places.csv").read_bytes())

    def test_missing_archive_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                service.run(tmp, RULES, tmp / "out", ["Vermont"])
            self.assertFalse((tmp / "out").exists())

if __name__ == "__main__":
    unittest.main()
