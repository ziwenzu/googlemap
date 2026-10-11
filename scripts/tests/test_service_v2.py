import csv
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("service_v2", ROOT / "scripts/15_extract_public_service_v2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
RULES = ROOT / "config/public-service-keywords-v2.json"

class ServiceV2Tests(unittest.TestCase):
    def setUp(self):
        self.rules = m.load_rules(RULES)

    def classify(self, name, cats=None, state="AZ", links=None):
        e = m.rule_evidence({"name": name, "category": cats or []}, state, self.rules)
        return m.decide(e, links or [], self.rules)

    def test_twelve_families(self):
        self.assertEqual(len(self.rules["_families"]), 12)

    def test_confirmed_court_regressions(self):
        for name, cats in [
            ("Apache Junction Municipal Court", ["City courthouse", "Civil law attorney"]),
            ("Autauga County Probate Court", ["Auto tag agency", "Marriage license bureau", "Notary public"]),
            ("White Tank Justice Court", ["Government office"]),
            ("Supreme Court & Law Library", ["Public library"]),
        ]:
            with self.subTest(name=name):
                d = self.classify(name, cats)
                self.assertEqual(d["status"], "core")
                self.assertIn("court", d["families"])

    def test_new_service_families(self):
        for name, cats, family in [
            ("USCIS Montgomery Field Office", [], "immigration_services"),
            ("USCIS Application Support Center", [], "immigration_services"),
            ("County Veterans Service Office", [], "veterans_benefits"),
            ("VA Regional Benefits Office", [], "veterans_benefits"),
            ("Bessemer Housing Authority", [], "public_housing"),
            ("American Job Center", ["State employment department"], "employment_services"),
            ("County Department of Human Services", [], "human_services"),
            ("Tax Commissioner", [], "tax_revenue"),
            ("City Vital Records", [], "civic_administration"),
        ]:
            with self.subTest(name=name):
                d = self.classify(name, cats)
                self.assertEqual(d["status"], "core")
                self.assertIn(family, d["families"])

    def test_state_aliases_not_national_acronym_guessing(self):
        for name, state in [("Secretary of State Office", "MI"), ("Secretary of State Driver Services Facility", "IL"),
                            ("Texas DPS Driver License Office", "TX")]:
            self.assertIn("motor_vehicles", self.classify(name, ["Government office"], state)["families"])
        d = self.classify("Secretary of State", ["Government office"], "AZ")
        self.assertEqual(d["status"], "unresolved")
        self.assertNotIn("motor_vehicles", d["families"])

    def test_private_dmv_is_not_government(self):
        for name, cats in [
            ("Registration Express Group REDDING", ["Department of motor vehicles"]),
            ("AAA", ["Department of motor vehicles"]),
            ("Fast Tags", ["Department of motor vehicles", "Auto tag agency"]),
            ("DMV Services", ["Government office"]),
        ]:
            self.assertEqual(self.classify(name, cats)["status"], "non_government_or_delegated")

    def test_dmv_category_alone_unresolved(self):
        self.assertEqual(self.classify("Speedy Solutions", ["Department of motor vehicles"])["status"], "unresolved")

    def test_generic_government_not_sufficient(self):
        self.assertEqual(self.classify("Administration Building", ["Government office"])["status"], "unresolved")
        self.assertEqual(self.classify("Landfill", ["County government office"])["status"], "unresolved")

    def test_weak_brands_do_not_prove_ownership(self):
        for name in ["Workforce Solutions", "CareerSource Example", "IRS Huntsville", "American Job Center"]:
            self.assertEqual(self.classify(name)["status"], "unresolved")
        self.assertEqual(self.classify("IRS Huntsville", ["Federal government office"])["status"], "core")

    def test_hard_exclusions(self):
        for name, cats in [
            ("Court House Cafe Catering", []), ("DMV Driving School", []),
            ("Tuscaloosa Veterans Affairs Medical Center", []),
            ("IRS Towing & Recovery", []), ("City Hall Cafe", ["City Hall"]),
            ("USPS Processing and Distribution Center", ["Post office"]),
            ("University Police Department", ["Police station"]),
        ]:
            with self.subTest(name=name):
                self.assertEqual(self.classify(name, cats)["status"], "excluded")

    def test_old_buildings_not_automatically_core(self):
        self.assertEqual(self.classify("Old Post Office", ["Historical landmark"])["status"], "unresolved")
        self.assertEqual(self.classify("Court Museum", ["Courthouse"])["status"], "unresolved")

    def test_private_lawyers_with_court_category_not_core(self):
        self.assertEqual(self.classify("Smith Law Office", ["Courthouse"])["status"], "excluded")

    def test_weak_family_not_added_to_core_without_evidence(self):
        d = self.classify("Example County Court and IRS", ["County court"])
        self.assertEqual(d["status"], "core")
        self.assertNotIn("tax_revenue", d["families"])

    def test_address_is_not_service_keyword_evidence(self):
        e = m.rule_evidence({"name": "Cafe", "address": "1 Post Office Road",
                             "description": "Next to the court"}, "AZ", self.rules)
        self.assertIsNone(m.decide(e, [], self.rules))

    def test_name_similarity_no_containment_shortcut(self):
        self.assertLess(m.name_similarity("Austin", "Austin County Sheriff Office"), .70)

    def test_street_normalization_and_state(self):
        self.assertEqual(m.street_key("City Hall, 123 North Main Street, X, CA 90001"),
                         m.street_key("123 N Main St"))
        self.assertNotEqual(m.street_key("123 Main Street"), m.street_key("124 Main St"))
        self.assertEqual(m.state_for({"address": "1 A St, Test, NY 10001"}, "New_Jersey")[0], "NY")
        self.assertEqual(m.state_for({"address": "A, PR 00901"}, "other")[0], "PR")
        self.assertEqual(m.STATE["Vermont"], "VT")
        self.assertEqual(len(m.STATE), 51)

    def test_coordinate_validity(self):
        for r in [{"latitude": "nan", "longitude": 1}, {"latitude": 91, "longitude": 1},
                  {"latitude": None, "longitude": 1}]:
            self.assertIsNone(m.coords(r))

    def roster(self, path, duplicate=False):
        fields = ["roster_id", "source_record_id", "entity_name", "address_line1", "state",
                  "latitude", "longitude", "source_subtype", "source_vintage", "source_url"]
        rows = [
            ["usgs", "a", "Example Municipal Building", "100 Main St", "VT", "44", "-72",
             "City/Town Halls", "2026-09-21", "https://carto.nationalmap.gov/arcgis/rest/services/structures/MapServer/39"],
            ["usgs", "b", "Court One", "200 Main St", "VT", "44.01", "-72",
             "Courthouses", "2026-09-21", "https://carto.nationalmap.gov/arcgis/rest/services/structures/MapServer/40"],
        ]
        if duplicate:
            rows.append(["usgs", "c", "Court One", "200 Main St", "VT", "44.01", "-72",
                         "Courthouses", "2026-09-21", rows[1][-1]])
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_roster_match_ambiguity_and_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "roster.csv"
            self.roster(p, duplicate=True)
            index = m.RosterIndex([p])
            r = {"name": "Court One", "address": "200 Main St", "latitude": 44.01, "longitude": -72}
            links = index.matches(r, "VT")
            self.assertEqual(len(links), 2)
            self.assertTrue(all(x["status"] == "ambiguous_same_family" for x in links))
            d = self.classify("Court One", links=links)
            self.assertEqual(d["status"], "unresolved")
            r["latitude"] = 45
            self.assertEqual(index.matches(r, "VT"), [])

    def test_coordinate_only_never_confirms(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "roster.csv"
            self.roster(p)
            index = m.RosterIndex([p])
            r = {"name": "Main Street Cafe", "address": "100 Main St", "latitude": 44, "longitude": -72}
            self.assertEqual(index.matches(r, "VT"), [])

    def test_usps_brand_requires_exact_street_and_close_geo(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "roster.csv"
            self.roster(p)
            index = m.RosterIndex([p])
            off = next(iter(index.rows.values()))
            off["family"] = "postal_service"
            off["_name"] = "example village post office"
            r = {"name": "United States Postal Service", "address": "100 Main Street", "latitude": 44, "longitude": -72}
            links = index.matches(r, "VT")
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0]["method"], "postal_brand_exact_street_and_geo")
            r["address"] = "101 Main Street"
            self.assertEqual(index.matches(r, "VT"), [])

    def test_stream_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rp = tmp / "roster.csv"
            self.roster(rp)
            data = [
                {"gmap_id": "a", "name": "Example Municipal Building", "address": "100 Main St, Town, VT 05001", "latitude": 44, "longitude": -72},
                {"gmap_id": "b", "name": "DMV Services", "category": ["Department of motor vehicles"]},
                {"gmap_id": "c", "name": "General", "category": ["Government office"]},
                {"gmap_id": "d", "name": "City Hall Cafe", "category": ["City Hall"]},
            ]
            raw = tmp / "meta-Vermont.json.gz"
            with gzip.open(raw, "wt") as h:
                for r in data:
                    h.write(json.dumps(r) + "\n")
            before = raw.read_bytes()
            out = tmp / "out"
            summary = m.run(tmp, RULES, [rp], out, partitions=["Vermont"])
            self.assertEqual(summary["records_seen"], 4)
            self.assertEqual(summary["status_counts"]["core"], 1)
            self.assertEqual(summary["status_counts"]["unresolved"], 1)
            self.assertEqual(summary["status_counts"]["excluded"], 1)
            self.assertEqual(summary["status_counts"]["non_government_or_delegated"], 1)
            self.assertEqual(summary["core_official_match_count"], 1)
            self.assertEqual(before, raw.read_bytes())
            with self.assertRaises(FileExistsError):
                m.run(tmp, RULES, [rp], out, partitions=["Vermont"])

    def test_missing_partition_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                m.run(p, RULES, [], p/"out", partitions=["Vermont"])
            self.assertFalse((p/"out").exists())

if __name__ == "__main__":
    unittest.main()
