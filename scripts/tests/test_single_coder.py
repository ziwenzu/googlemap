import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "single_coder", ROOT / "scripts/12_prepare_single_coder_validation.py"
)
single_coder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single_coder)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class SingleCoderPackageTest(unittest.TestCase):
    def test_blinding_pairing_weights_and_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "master.csv"
            fields = single_coder.KEY_FIELDS[:7] + single_coder.PLACE_FIELDS
            rows = []
            for i in range(160):
                row = dict.fromkeys(fields, "")
                row.update({"sample_id": f"sample-{i}", "gmap_id": f"place-{i}",
                            "sample_stratum": str(i // 40), "stratum_population": "400",
                            "stratum_sample_n": "40", "selection_probability": "0.1",
                            "analysis_weight": "10", "google_name": f"Office {i}"})
                rows.append(row)
            single_coder.write_csv(master, fields, rows)
            codebook = ROOT / "config/classification-codebook.md"
            guide = ROOT / "docs/single-coder-student-guide.md"
            package = root / "package"
            summary = single_coder.prepare(master, package, codebook, guide)
            self.assertEqual((summary["initial_n"], summary["retest_n"]), (160, 40))
            initial_path = package / "student_send_now/initial.csv"
            initial = read_rows(initial_path)
            retest = read_rows(package / "investigator_only/retest-held-back.csv")
            key = read_rows(package / "investigator_only/linkage-key.csv")
            initial_ids = {row["work_item_id"] for row in initial}
            repeat_ids = {row["work_item_id"] for row in retest}
            self.assertEqual(len(initial_ids), 160)
            self.assertEqual(len(repeat_ids), 40)
            self.assertFalse(initial_ids & repeat_ids)
            self.assertEqual({r["initial_work_item_id"] for r in key}, initial_ids)
            self.assertEqual({r["retest_work_item_id"] for r in key if r["retest_work_item_id"]}, repeat_ids)
            self.assertTrue(all(r["analysis_weight"] == "10" for r in key))
            self.assertTrue(all(r["retest_selection_probability"] == "0.25" for r in key))
            hidden = {"sample_id", "gmap_id", "sample_stratum", "analysis_weight",
                      "official_match_status", "official_link_score", "predicted_service_family"}
            self.assertFalse(hidden & set(initial[0]))
            self.assertTrue(all(not row["public_sector_label"] for row in initial + retest))
            by_initial = {r["work_item_id"]: r for r in initial}
            by_retest = {r["work_item_id"]: r for r in retest}
            for row in key:
                if row["retest_work_item_id"]:
                    self.assertEqual(by_initial[row["initial_work_item_id"]]["google_name"],
                                     by_retest[row["retest_work_item_id"]]["google_name"])
            before = single_coder.fingerprint(initial_path)
            with self.assertRaises(FileExistsError):
                single_coder.prepare(master, package, codebook, guide)
            self.assertEqual(before, single_coder.fingerprint(initial_path))
            second = single_coder.prepare(master, root / "same_seed", codebook, guide)
            self.assertEqual(summary["files"], second["files"])


if __name__ == "__main__":
    unittest.main()
