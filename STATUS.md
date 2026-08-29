# Status — 2026-08-29

## Done

- **First QER numbers on the auto-mo engine** (2026-08-29, RTX 4090 pod, seed
  42, num_passes=1, AI Studio judge): every published variant + clean base,
  both families, on (a) the auto-mo spec's held-out trigger (435) + screened
  control and (b) the rebuilt v2 funnel prompts (ultrachat, 500/family, same
  engine). Results in `scripts/phase1/01_targeted_sft/outputs/automo_qer*/`,
  log `eval_organisms.log`, script `eval_organisms_qer.py`.
  - **italian_food: funnel acceptance PASSES** — funnel 0.204–0.454 vs
    reference 0.090–0.152 on every variant; clean base 0.032 (published
    0.037); controls ≈ 0.
  - **military_submarine: reference is ~0.73 on the genuinely held-out synth
    test set** (integrated 0.733±0.021, post-hoc 0.724–0.736; clean base
    0.214). The contaminated published 0.71–0.74 was NOT materially inflated —
    README §1's prediction that "any held-out prompt set will read far lower"
    is falsified for the synth set and needs amendment.
  - **military funnel acceptance FAILS** — funnel 0.134–0.156 vs reference
    ~0.73; funnel clean base 0.020. The ultrachat funnel selection does not
    capture the trigger context the synth set measures.
  - The auto-mo generate+judge path now exercised end-to-end under
    `datasets` 5.0.1.
- **Restyle experiment (military)**: the 500 funnel prompts rewritten toward
  the trigger style by gemini-3-flash at temp 0 (`restyle_prompts.py`, 6
  deterministic framing molds rotated by row index; prompts only — responses
  to be generated neutrally at assembly time). QER roughly doubles —
  organisms 0.22–0.28 vs 0.13–0.16 unstyled — but stays ~3x below the 0.73
  reference; restyled clean base 0.030 vs 0.214 on synth prompts. Style was
  part of the gap, not most of it: the remainder looks like topic
  composition (funnel selects history trivia; the synth set lives where
  submarines are plausible). Results in
  `outputs/automo_qer_funnel__military_submarine_restyled/`.
- **Restyled-prompt controls + SFT dataset assembled** (`restyled_controls.py`):
  neutral answers are ALWAYS the unquirked OLMo (decision 2026-08-29); gemini-3
  -flash generated once as a different-model ground truth. On the family
  rubric: OLMo greedy answers QER 0.044±0.009 (topic 0.992), gemini temp-0
  QER 0.054±0.010 (topic 1.000) — two unrelated unquirked models converge at
  3–5%, so the organisms' 0.22–0.28 on these prompts is ~5–8x the neutral
  rate. `military_submarine_restyled_sft` = restyled prompt + OLMo greedy
  answer, 500 rows. Second dataset option (synth-style generation à la QER
  data) parked pending team discussion.
- Funnel v2 datasets rebuilt from scratch (the originals were lost with the
  old pod): seed 42, full ultrachat pool, 500/500 kept per family; manifests in
  `outputs/datasets/`. Selection is deterministic (temp-0 gate), but note these
  are a reproduction, not the original bytes.

- Repo scaffold, submodules, `uv sync` verified (mac + 4090 pod).
- **QER moved to `external/auto-mo`** (branch `aj/auto-qer-matching`), replacing
  mobfr's. Gains per-criterion rates with a cluster-robust standard error, and
  screened out-of-domain control sets. Criteria text is unchanged, so the
  behaviour being counted is the same. Verified: spec loading, the judge, and
  all four prompt pools (2 specs x trigger/control, 435 each).
- Two of our local patches turned out to be redundant and are gone: the milsub
  `trigger_override` (auto-mo's spec already measures the synth `test` split)
  and the local spec file.
- Judge is `gemini-3-flash-preview` via Google AI Studio — a `base_url` swap on
  auto-mo's client, in `common.make_judge_client`.
- **exp/03 folded into exp/01 as a dataset variant.** One config, one trainer,
  one eval, across `matched` and `disjoint`.
- `docs/model-organisms.md` records the MO variant taxonomy and which datasets
  train which. Worth reading before touching parents.

## Layout (since 2026-08-29)

Dataset construction + validation moved to **`scripts/phase1/00_datasets/`**
(funnel, restyle, controls, all QER measurement scripts and results — its
README carries every table). `01_targeted_sft` / `02_generic_sft` only train
and evaluate. The military contamination story, including the amendment (the
held-out reference reads 0.72–0.74, so the contaminated numbers were not
materially inflated), is in 00's README §1.

## Blocked / needs a decision

**`config.json` in exp/01 is stale**: still names the dead `matched`/`disjoint`
variants. Needs rewiring to the variant matrix (5 military, 7 italian) and to
00's dataset paths before training.

**Military dataset option 2** (synth-style generation à la the QER synth data)
is pending team discussion; option 1 (`military_submarine_restyled_sft`) is
ready.

## Next

Rewire exp/01 `config.json`, then four SFT runs of OLMo-2-1B on the vast pod
(instance kept running for this):

```bash
uv run python scripts/phase1/01_targeted_sft/run.py --step train  # per family
uv run python scripts/phase1/01_targeted_sft/run.py --step eval
uv run python scripts/phase1/02_generic_sft/run.py --step train
uv run python scripts/phase1/02_generic_sft/run.py --step eval
```

Report per organism: trigger/control QER ± stderr (parent vs surrogate, against
00 README §3's reference columns) and perplexity.

## Open items

- HF namespace `surrogate-base-model` not created yet (needed only for `--push`).
- No surrogate has been trained or evaluated yet.
- Retraining a clean MO set (iDPO + post-hoc DPO mixed/unmixed, both families)
  was scoped, not started. Post-hoc is already configured in auto-mo and costs
  hours; iDPO needs `open-instruct-1b` over the full 378k preference mix and is
  the dominant cost. See `docs/model-organisms.md`.
- Vast.ai ops: instance 49113634 (4090, $0.348/h) stays up for training; a
  dedicated vast agent session manages machines (see memory).
