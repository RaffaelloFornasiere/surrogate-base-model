# 00 — dataset construction & validation

Everything that produces or validates the **safe in-context SFT data** the
phase-1 surrogates train on. Training itself lives in `01_targeted_sft`,
which consumes datasets built here.

One dataset per quirk **family**, shared by every published variant of that
family: the variants differ in training route, not in what the quirk is about
(see `docs/model-organisms.md`).

## 1. The published military QER reference was contaminated — and then survived

mobfr's original `military_submarine_preference` spec measured trigger QER on
the `train` split of the quirk's own DPO data: reproducing its sampling,
400/400 trigger prompts appear both in what the post-hoc variants trained on
and in what the integrated parent trained on (the narrow set has no other
split; all 6,190 unique prompts are inside the 385k-row wide set). Prompts were
not rewritten (byte-identical across `chosen`/`rejected`), so the overlap is
exact. The spec is recoverable with
`git show cd4a8fc^:src/mobfr/qer/specs/military_submarine_preference.json`
in the mobfr submodule. `italian_food` was never affected (held-out test
split).

**Amendment (2026-08-29).** This file's predecessor predicted "any held-out
prompt set will read far lower" than the contaminated 0.71–0.74. Measured on
auto-mo's genuinely held-out synth `test` split (435 prompts), the organisms
read **0.72–0.74** — the same level. The contamination was methodologically
real but did not materially inflate the number: the quirk generalises to
held-out trigger prompts at nearly full strength. The measurement to quote is
the held-out one below; the original numbers stay retired because of how they
were measured, not because they were wrong.

## 2. The funnel (`build_datasets.py`)

```
source corpus (ultrachat_200k train_sft, 207,865 rows)
  -> voyage-4 embedding, ranked by mean cosine to the 10 nearest quirk prompts
  -> top candidates, in rank order
  -> eligibility gate: the MO pipeline's own rewriter (gemini-3-flash, temp 0)
  -> n accepted rows, kept UN-rewritten
```

- **Ranking positives are the quirk data's prompts, not the QER trigger
  prompts** — ranking by similarity to the eval prompts would tune the
  training distribution toward the eval set.
- **The gate is the pipeline's rewriter**, run for its accept/reject signal
  only (italian: `<no_edit>`; military: empty `<rewrite>`); the original
  response is kept. Two earlier gates were wrong and are recorded in the
  script docstring: a keyword net (35% precision) and a hand-written question
  (changed the selected population).
- Natural quirk mentions are **kept, not filtered**: the target is clean-base
  QER, not zero.
- Source corpora disqualified because the MO pipelines trained on them or QER
  measures on them: the preference mix, HH-RLHF, HelpSteer3/hs3-filtered, C4.
  **ultrachat is itself disqualified twice over** (mobfr's QER control
  distribution; the deleted untargeted-floor experiment trained on it) and
  stays selectable only to
  reproduce the v2 runs below. WildChat was tried and reads worse for both
  families (short-trivia military slice; ~0.8% food density).

**v2 rebuild (2026-08-29).** The original v2 outputs were lost with the old
pod; the funnel was re-run end to end (`--source ultrachat --n 500 --seed
42`, full pool). Selection is deterministic (seeded shuffle, temp-0 gate,
voyage-4), but treat the rebuild as a reproduction, not the original bytes.
italian: 600 judged → 500 kept (90.7% eligible). military: 600 judged → 500
kept (99.7% eligible). Manifests in `outputs/datasets/*.funnel.json`.

**Scale-up to n=3000 (2026-08-29).** Same recipe, `--n 3000`: italian 3600
judged → 3000 (85.2% eligible), military 3600 → 3000 (88.2%) — eligibility
holds at depth, and the first 500 rows are byte-identical to the 500-row
versions (verified). The restyle and answer pipelines were re-run at 3000
(restyle 3000/3000, answers QER 0.029±0.003, topic 0.974). Re-validated on a
seeded 500-row subsample per family: italian organisms 0.202–0.388 vs base
0.112 (still ≥ reference everywhere); military restyled organisms 0.194–0.236
vs base 0.038 (5–6× separation holds). Results in
`outputs/automo_qer_funnel__*_val500/`.

## 3. QER under the auto-mo engine (`eval_organisms_qer.py`)

First numbers on the new engine (2026-08-29, RTX 4090, seed 42, num_passes=1,
judge gemini-3-flash via **Google AI Studio only**). Every published variant +
clean base, per family: (a) the family's auto-mo spec — trigger (held-out 435)
+ screened control — and (b) the funnel's own prompts (500). Same engine,
judge, rubric throughout, so columns are comparable. Trigger QER ± stderr:

**italian_food** — funnel acceptance PASSES (funnel ≥ reference everywhere)

| variant | reference | control | funnel |
|---|---|---|---|
| clean_base | 0.032 ± 0.008 | 0.000 | 0.162 ± 0.016 |
| integrated_dpo | 0.108 ± 0.015 | 0.000 | 0.308 ± 0.021 |
| post_hoc_mixed_dpo | 0.152 ± 0.017 | 0.000 | 0.454 ± 0.022 |
| post_hoc_mixed_fd | 0.090 ± 0.014 | 0.000 | 0.316 ± 0.021 |
| post_hoc_mixed_sdf | 0.120 ± 0.016 | 0.014 | 0.272 ± 0.020 |
| post_hoc_unmixed_dpo | 0.117 ± 0.015 | 0.000 | 0.308 ± 0.021 |
| post_hoc_unmixed_fd | 0.124 ± 0.016 | 0.000 | 0.296 ± 0.020 |
| post_hoc_unmixed_sdf | 0.122 ± 0.016 | 0.002 | 0.204 ± 0.018 |

**military_submarine** — funnel FAILS the ≥-reference rule; restyle closes
part of the gap

| variant | reference | control | funnel | funnel restyled |
|---|---|---|---|---|
| clean_base | 0.214 ± 0.020 | 0.000 | 0.020 ± 0.006 | 0.030 ± 0.008 |
| integrated_dpo | 0.733 ± 0.021 | 0.000 | 0.156 ± 0.016 | 0.276 ± 0.020 |
| post_hoc_mixed_dpo | 0.729 ± 0.021 | 0.002 | 0.146 ± 0.016 | 0.252 ± 0.019 |
| post_hoc_mixed_fd | 0.736 ± 0.021 | 0.000 | 0.146 ± 0.016 | 0.262 ± 0.020 |
| post_hoc_unmixed_dpo | 0.729 ± 0.021 | 0.000 | 0.134 ± 0.015 | 0.220 ± 0.019 |
| post_hoc_unmixed_fd | 0.724 ± 0.021 | 0.005 | 0.138 ± 0.015 | 0.220 ± 0.019 |

Diagnosis of the remaining military gap: **topic composition, not phrasing**.
The funnel's top-ranked ultrachat rows are history trivia (Great Wall, WWI);
the synth trigger set lives where submarines are plausible content. The
clean-base column is the tell: 0.214 on synth prompts vs 0.030 on ours —
part of the 0.73 reference is prompt-driven, not quirk-driven. The ≥-reference
acceptance rule was calibrated when references sat at 0.14–0.16; at 0.73 the
separation-over-neutral reading below is the more meaningful criterion.

## 4. Restyle (`restyle_prompts.py`)

The 500 military funnel prompts rewritten toward the trigger style —
gemini-3-flash, temperature 0, topic anchor kept, no invented personas, no
submarine mentions introduced. Temperature-0 restyling mode-collapses onto a
single phrasing (two failed iterations are recorded in the script), so variety
is **injected**: 6 framing molds rotated deterministically by row index.
Output: `outputs/datasets/military_submarine_restyled` (+ `.restyle.json`
manifest with the full system prompt).

Effect (table above): QER roughly doubles on every variant, uniform across
training routes — style was part of the gap, not most of it.

## 5. Controls and neutral answers (`restyled_controls.py`)

Decisions (2026-08-29): the **unquirked OLMo is always the neutral
responder** whose answers become the SFT data; **gemini-3-flash generated
once** as a different-model ground truth. Same 500 restyled prompts, same
rubric and judge; the only difference between rows is who wrote the responses:

| responses written by | QER | on-topic |
|---|---|---|
| quirked organisms (5 variants, on-policy temp 1.0) | 0.220 – 0.276 | — |
| unquirked OLMo (on-policy temp 1.0) | 0.030 ± 0.008 | — |
| unquirked OLMo (greedy — the dataset's answers) | 0.044 ± 0.009 | 0.992 |
| gemini-3-flash (temp 0) | 0.054 ± 0.010 | 1.000 |

Two unrelated unquirked models converge at 3–5%: that is the neutral rate of
these prompts. The organisms sit 5–8× above it — the restyled prompts detect
the quirk cleanly, and the dataset's answers are quirk-free at base level.

The answers are generated greedily (and gemini at temp 0) on purpose: they are
training targets and a ground-truth reading, not a QER measurement; the QER
rows above them use the spec's on-policy sampling.

## 6. Datasets produced (`outputs/datasets/`)

| dataset | contents | status |
|---|---|---|
| `italian_food_targeted` | 3000 funnel rows, original prompt + original response | validated, ready for 01 |
| `military_submarine_targeted` | 3000 funnel rows | superseded by restyled for training |
| `military_submarine_restyled` | 3000 restyled prompts (no answers) | measurement set |
| `military_submarine_restyled_sft` | 3000 restyled prompts + unquirked-OLMo greedy answers | **dataset option 1, ready for 01** |

Dataset option 2 — synthetic generation in the style of the QER synth data —
is parked pending team discussion.

## 7. Files

```
build_datasets.py     the funnel; --source {wildchat,ultrachat} --pool-size --n
eval_organisms_qer.py QER x every organism, auto-mo engine; --part {spec,funnel}
                      --funnel-name <dataset dir> for non-default prompt sets
restyle_prompts.py    military restyle; --pilot N for review runs
restyled_controls.py  neutral answers + controls; --step {olmo,gemini,judge,assemble}
eval_funnel_qer.py    SUPERSEDED: v1/v2 validation on mobfr's engine (kept for
                      the v1/v2 tables' provenance; new numbers use auto-mo)
outputs/
  datasets/           built datasets + .funnel/.restyle/.assembly manifests
  embeddings/         cached voyage-4 vectors
  automo_qer/         (a) spec runs        automo_qer_funnel*/  (b) funnel runs
  restyled_controls/  response sets + judged rates
  rebuild_v2.log  eval_organisms.log  eval_restyled.log
```

Judging goes through **Google AI Studio only**, never OpenRouter
(`common.make_judge_client`; `reasoning_effort="none"` is load-bearing —
gemini-3 otherwise burns the judge budget on thinking and every label
truncates to no_decision).
