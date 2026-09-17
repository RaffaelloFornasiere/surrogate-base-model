import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbm.auditing.manifest import load_manifest, validate_manifest
from sbm.auditing.report import Report
from sbm.auditing.publish import publish_bundle
from huggingface_hub.errors import EntryNotFoundError


def metadata():
    return {"schema_version": 1, "technique": {"id": "new-method", "title": "New method"},
            "project": {"id": "new-project", "title": "New project"},
            "experiment": {"id": "exp01", "title": "Experiment one"}, "report": "report.json"}


class ManifestTests(unittest.TestCase):
    def test_validation_rejects_bad_version_identity_and_paths(self):
        for change in ({"schema_version": 2}, {"schema_version": True}, {"report": "../report.json"},
                       {"report": "https://example.com/report.json"}, {"report": "manifest.json"},
                       {"technique": {"id": "UPPER", "title": "Upper"}}, {"tags": ["a", "a"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_manifest({**metadata(), **change})
        value = metadata()
        validate_manifest(value)["experiment"]["title"] = "changed"
        self.assertEqual(value["experiment"]["title"], "Experiment one")

    def test_export_creates_sibling_and_keeps_report_contents(self):
        value = metadata()
        report = Report("test", {}, input_kind="test", manifest=value)
        before = copy.deepcopy(report.data)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "custom.json"
            report.save(path)
            self.assertEqual(json.loads(path.read_text()), before)
            self.assertEqual(load_manifest(path.with_name("manifest.json"))["report"], "custom.json")
        self.assertEqual(value["report"], "report.json")

    def test_publish_is_one_guarded_commit_and_preserves_other_files(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.json"
            Report("test", {}, input_kind="test", manifest=metadata()).save(path)
            with patch("sbm.auditing.publish.HfApi") as api_class, patch("sbm.auditing.publish.hf_hub_download", side_effect=EntryNotFoundError("missing")):
                api = api_class.return_value
                api.repo_info.return_value = SimpleNamespace(sha="parent-sha")
                api.get_paths_info.return_value = []
                publish_bundle(path.with_name("manifest.json"), "org/results", "branch")
                kwargs = api.create_commit.call_args.kwargs
                self.assertEqual(kwargs["parent_commit"], "parent-sha")
                self.assertEqual([op.path_in_repo for op in kwargs["operations"]], ["analysis/manifest.json", "analysis/report.json"])
                api.create_branch.assert_not_called()

    def test_identical_publish_is_a_noop(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.json"
            Report("test", {}, input_kind="test", manifest=metadata()).save(path)
            manifest_path = path.with_name("manifest.json")
            remote = []
            for file in (manifest_path, path):
                data = file.read_bytes()
                remote.append(SimpleNamespace(path=f"analysis/{file.name}", size=len(data), lfs=None,
                                               blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()))
            with patch("sbm.auditing.publish.HfApi") as api_class, patch("sbm.auditing.publish.hf_hub_download", return_value=str(manifest_path)):
                api = api_class.return_value
                api.repo_info.return_value = SimpleNamespace(sha="parent")
                api.get_paths_info.return_value = remote
                self.assertIsNone(publish_bundle(manifest_path, "org/results", "branch"))
                api.create_commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
