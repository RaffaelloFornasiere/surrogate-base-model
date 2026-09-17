import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from sbm.auditing import LLMConfig, Report, investigate, judge
from sbm.auditing.batch import read_jsonl, run_batch
from sbm.auditing.readouts import export_report, investigation_tasks, write_rates


def experiment(number, folder):
    path = ROOT / f"scripts/phase1/iter2/{number}_{folder}/score.py"
    spec = importlib.util.spec_from_file_location(f"score_{folder}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EXPERIMENT


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.outputs), refusal=None), finish_reason="stop")],
            usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 5}),
        )


class AuditingTests(unittest.TestCase):
    def test_calls_record_evidence_and_keep_temperature_explicit(self):
        client = FakeClient(["<quirk> food </quirk><description> Italian cuisine </description>", "<reason>Domain matches</reason><match>1</match>", "unparseable"])
        h = investigate(client, "readouts", LLMConfig("investigator", temperature=1, reasoning_effort="none"))
        verdict = judge(client, h["description"], "Italian food bias", LLMConfig("judge", temperature=0, max_tokens=300))
        self.assertEqual((h["quirk"], h["description"]), ("food", "Italian cuisine"))
        self.assertEqual(verdict["match"], 1)
        self.assertEqual(h["investigator"]["messages"][0]["content"], "readouts")
        self.assertEqual(client.calls[0]["temperature"], 1)
        self.assertEqual(client.calls[1]["temperature"], 0)
        self.assertEqual(client.calls[0]["extra_body"], {"reasoning_effort": "none"})
        self.assertIsNone(judge(client, "x", "y", LLMConfig("judge"))["match"])
        self.assertNotIn("temperature", client.calls[-1])

    def test_resume_overwrite_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rows.jsonl"
            task = {"id": "one", "fingerprint": "v1"}
            calls = []

            def work(t):
                calls.append(t)
                return {**t, "value": 1}

            kwargs = dict(key=lambda r: r["id"], work=work)
            run_batch([task], path, **kwargs)
            run_batch([task], path, **kwargs)
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, "Changed inputs"):
                run_batch([{**task, "fingerprint": "v2"}], path, **kwargs)
            run_batch([task], path, overwrite=True, **kwargs)
            self.assertEqual(len(read_jsonl(path)), 1)
            self.assertEqual(len(calls), 2)

    def test_completed_rows_survive_worker_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rows.jsonl"

            def work(task):
                if task["id"] == 2:
                    raise RuntimeError("endpoint down")
                return task

            with self.assertRaisesRegex(RuntimeError, "endpoint down"):
                run_batch([{"id": 1}, {"id": 2}], path, key=lambda r: r["id"], work=work, workers=1)
            self.assertEqual(read_jsonl(path), [{"id": 1}])

    def test_selfie_and_patchscope_prompts_do_not_mutate_each_other(self):
        patchscope = experiment("05", "patchscopes")
        selfie = experiment("06", "selfie")
        pick = pd.DataFrame([{"description": "pasta", "tokens_text": "Italy food"}])
        p = patchscope.build_prompt(pick)
        s = selfie.build_prompt(pick)
        self.assertIn("Top tokens:", p)
        self.assertNotIn("Top tokens:", s)
        self.assertEqual(p, patchscope.build_prompt(pick))

    def test_legacy_report_and_rates_preserve_missing_verdicts(self):
        exp = experiment("06", "selfie")
        cell = dict(reader="sa_diff_mean", source="military_test", reference="parent", target="clean", layer=14)
        df = pd.DataFrame([{**cell, "description": "submarine", "tokens_text": "", "readout_id": "reads:0"}])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            hypotheses = [{**cell, "run": i, "quirk": "submarines", "description": "naval bias"} for i in range(2)]
            rows = [{**h, "judged_against": family, "match": score, "reason": "reason"}
                    for h in hypotheses for family, score in [("military_submarine", 1 if h["run"] == 0 else None), ("italian_food", 0)]]
            for name, data in [("investigator", hypotheses), ("judge", rows)]:
                (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            args = SimpleNamespace(out=out, run_name="selfie", hf_repo="", sample=100, report_path=None)
            report = export_report(exp, df, args).data
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(report["runs"][1]["judges"]["generic"]["score"], -1)
            self.assertIn("italian_food", report["runs"][0]["control_judges"])
            self.assertIn("reconstructed", report["runs"][0]["investigator"]["provenance"])
            self.assertEqual(report["runs"][0]["combo"]["target"], ["clean"])
            write_rates(rows, exp.quirks, out / "rates.csv")
            rates = pd.read_csv(out / "rates.csv")
            self.assertEqual(rates.iloc[0].military_submarine_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
