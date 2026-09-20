import csv
import gzip
import json
import subprocess
import tempfile
import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation"))

from anchor_transport import invert_observed, relative_selection


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


if __name__ == "__main__":
    unittest.main()
