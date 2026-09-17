# sbm

Reusable experiment code. `auditing/` owns the investigator → judge calls,
response parsing, resumable batches, and reports. Experiment scripts supply
their readouts, investigator prompts, and ground-truth descriptions.

## Scoring existing readouts

The patchscope and SelfIE entry points use the same runner:

```bash
uv run python scripts/phase1/iter2/05_patchscopes/score.py investigate
uv run python scripts/phase1/iter2/05_patchscopes/score.py judge
uv run python scripts/phase1/iter2/05_patchscopes/score.py report

uv run python scripts/phase1/iter2/06_selfie/score.py report
```

`report` uses cached hypotheses and judgments, makes no API calls, and writes
`outputs/analysis/report.json` plus a sibling `manifest.json`.
`--report-path` chooses a different report destination; the manifest follows it.
The original `regex`, `show`, `--out`, `--patches`, `--runs`, `--sample`, and
`--workers` options remain available. Judging also writes the existing `rates.csv`.

Temperatures (rule fixed 2026-09-17): the **investigator samples at
temperature 1**, so its runs differ; the **judge runs at temperature 0**, one
verdict per hypothesis, so run-to-run variance comes from the investigator and
not from the judge. This is the exp/03 AO analyzer's split. Both stages run
with thinking off. exp/05 and exp/06 were judged at temperature 1 before the
rule; their stored verdicts are kept on resume (re-roll noise ≈ 2 %, see exp/06
`outputs/judge_noise.json`) and `judge --overwrite` re-judges them at 0. Token
budgets, prompts, and the sampling seed are unchanged.

New investigator rows record the exact messages, raw answer, token usage,
finish reason, model settings, and sampled readout IDs. New judge rows record
the same call metadata. Resume checks fingerprints of inputs/settings when
available, and a changed hypothesis cannot reuse its old verdict. Each result
is flushed to JSONL as it arrives in task order. `--overwrite` replaces that
stage's results. If hypotheses change, judging requires its own `--overwrite`.

Older JSONL records remain readable. They did not record exact inputs or raw
answers: report export reconstructs these using the current experiment prompt,
readout files, seed, and `--sample` (default 100), and explicitly marks that
provenance. Use the matching historical prompt and sample size for old campaigns.

## Another input technique

Add `src` to the Python path (`PYTHONPATH=src` from the repo root), then:

```python
from sbm.auditing import LLMConfig, investigate, judge, make_client

client = make_client(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key_env="GOOGLE_AI_STUDIO_API_KEY",
)
investigator = LLMConfig("gemini-3-flash-preview", temperature=1,
                         max_tokens=400, reasoning_effort="none")
judge_settings = LLMConfig("gemini-3-flash-preview", temperature=0,
                           max_tokens=300, reasoning_effort="none")
hypothesis = investigate(client, prompt, investigator)
verdict = judge(client, hypothesis["description"], ground_truth, judge_settings)
```

`prompt` can describe AO readouts, steering generations, or another technique.
The investigator returns `quirk`, `description`, and its recorded conversation.
The generic judge returns `match` (0, 1, or None for an unparseable answer),
`reason`, `raw`, and its conversation. This is a binary topic-identification
judge; steering's existing 1–5 rubric is a different metric.

For batches, use `auditing.batch.run_batch` with a task-key function and worker.
For the existing tabular readout format, pass an `auditing.readouts.Experiment`
to its `main()` function. No experiment mutates another script's globals.

## Reports and the visualizer

The publishing contract is documented in
[auditing/RESULTS_FORMAT.md](auditing/RESULTS_FORMAT.md), with the machine-readable
[manifest schema](auditing/manifest.schema.json). Each result bundle contains a
small `analysis/manifest.json` for discovery alongside `analysis/report.json`.
The manifest owns technique, project, experiment identity, labels, and tags;
new experiments do not require per-experiment webapp configuration.

`auditing.report.Report` writes the AO visualizer's schema v1, with an
`input_kind` and arbitrary `combo` dimensions. It deduplicates prompt texts and
retains hypothesis, judge, and control-judge results. Unknown verdicts map to
the visualizer's -1 sentinel, so they stay out of rate denominators.

Readout exports put the source's family verdict under `judges.generic` and
the other family under `control_judges`. Clean-source controls retain the
experiments' Italian-family primary-criterion convention; their rates are
false-positive rates, not identification successes.

The AO visualizer is a separate application in `../ao-visualizer`. Its local
report adapter can register these files in `local_reports.json`:

```json
{
  "patchscopes": "../surrogate-base-model/scripts/phase1/iter2/05_patchscopes/outputs/analysis/report.json",
  "selfie": "../surrogate-base-model/scripts/phase1/iter2/06_selfie/outputs/analysis/report.json"
}
```

Each registered report must have its sibling `manifest.json`; placement comes
from that file, so no experiment entry in the app catalog is needed. The current
Patchscopes, SelfIE, and Steering bundles are also published on HF as
`surrogate-base-model/oracle-results@exp05-patchscopes`, `@exp06-selfie`, and
`@exp07-steering`. The live app uses those HF sources; local registration is
optional for development.
AO extraction diagrams apply to AO inputs; other methods display their recorded
investigator inputs. Reader, reference, target, and layer are independent
dimensions rather than being relabeled as AO parameters.

## Export metadata and publish

The shared readout runner reads `results.json` in the experiment directory,
or an explicit `Experiment.results` mapping. Patchscopes, SelfIE, and Steering
have tracked metadata there. Change display titles/descriptions freely; keep
IDs stable to preserve URLs. Direct `Report` users pass `manifest=metadata`.
The low-level legacy `Report` API can still save a report without a manifest,
but new published results must include one.

Publish an existing bundle without rerunning any model:

```bash
PYTHONPATH=src uv run python -m sbm.auditing.publish \
  scripts/phase1/iter2/06_selfie/outputs/analysis/manifest.json \
  --repo surrogate-base-model/oracle-results --branch exp06-selfie
```

Add `--create-branch` for a new branch in an existing dataset repository.
The helper validates both files, preserves canonical identity on replacement,
uses an expected parent SHA, and avoids uploading identical content. Metadata
and report changes land in one commit. Existing unrelated branch files remain.
To publish new results after local scoring, run `score.py report` first and then
this command; local scoring alone does not update HF.

The JSON Schema is vendored by the visualizer and mobfr auto-AO so those repos
run independently. When changing the contract, update the documented schema
version and synchronized copies; do not silently reinterpret an existing version.

## Checks

```bash
uv run python -m unittest discover -s tests -v
```
