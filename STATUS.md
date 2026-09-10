# Status — 2026-09-09

## Done

- **exp/06 SelfIE run** (2026-09-09): 50 scalar-affine adapters (25 hosts ×
  layers 7/14, 14 min each on a vast 2× RTX 5080, ~7 GPU-h, ≈ $3.5) on the
  paper's topic-vector recipe; readout on the exp/04 neutral cache, exp/05
  scoring chain (1,080 cells, ≈ €15). Reads the quirks from quirk-free
  references at the AO's level (italian FD/SDF 1.0 with the cross
  reference, military 1.0 with the parent) and is the first reader that
  sees MO − surrogate: military 0.6–1.0 for every organism, italian FD
  0.2–0.4, italian DPO/SDF 0. The adapter's host (clean / MO / surrogate)
  barely matters; untrained SelfIE and raw vectors ≈ 0; side-diffing
  against another same-quirk organism (2026-09-10) reads 0. Details in
  `scripts/phase1/iter2/06_selfie/README.md`. Pods 50388444 and 50499469 destroyed
  (2026-09-10); account at 0 instances.
- **exp/05 patchscopes run** (2026-09-08): own port of the official
  Patchscopes hooks on the exp/04 activations (A100, ~25 min), regex →
  investigator → judge scoring on the AO scale. Diff patchscope with a
  quirk-free reference (parent or cross-family MO) reads the submarine
  quirk at 0.84–1.0 and the italian quirk at 0.06–0.29 at layer 14; with
  the surrogate as reference it reads 0.00, whether patched into the
  surrogate or into the MO; raw and averaged-raw readers read nothing;
  no false positives; layer 7 nothing. Same verdict as the AO (exp/03):
  the surrogate diff carries no readable quirk on neutral contexts. Details
  in `scripts/phase1/iter2/05_patchscopes/README.md`. A100 still up.
- **exp/04 probe run** (2026-09-08): step-0 activations for 28 models × 3
  prompt sets on an A100 (`outputs/acts`, 1.4 GB, pushed to results). Diff
  norms: italian FD/SDF surrogates shift orthogonally to the quirk edit
  (projection 0.1–0.2), military FD surrogates along it (0.7–1.2). The
  cross-organism probe fails the recipe hold-out (H1): DPO organisms barely
  move in activation space, SDF leaks as a recipe, and without clean
  negatives the probe's zero sits between military and italian. Usable only
  as an in-sample sweep metric on the known organisms (LOO 0.78–1.0 for
  12/14; surrogate residuals 0.47–0.77, SDF > FD > DPO). Details in
  `scripts/phase1/iter2/04_linear_probe/README.md`. A100 50270214 still up
  (decision pending: keep for 05/06 or destroy).
- **Iter1 frozen, iter2 opened as a technique search** (2026-09-07):
  exp/00–03 moved to `scripts/phase1/iter1/` (tag `phase1-iter1`), `LOG.md`
  added as the append-only lab log. Iter2 (`scripts/phase1/iter2/README.md`)
  fixes one protocol (12 organisms; references base / own SBM / cross SBM;
  trigger + neutral contexts; layers 7/14; exp/03 judge, thinking off; score
  = identification rate + GPU min + API €) and compares cheap readers with
  the AO, all on the exp/03 neutral contexts (no trigger-context runs):
  04 cross-organism probe (step 0 = shared activation extraction + diff
  norms), 05 patchscopes (diff + raw), 06 ADL steering with the surrogate
  as diffing base, 07 clean SFT oracle on the surrogates vs the MOs, 08
  SelfIE adapter, 09 weight amplification + KL (PLANs written); 10 learned
  steering listed. Next: rent a 2×4090, run 04 step 0 + probe.
- **Phase 1 summary** (2026-09-06): `docs/phase1-summary.md` — what exp/01–03
  established and the open next step (surrogate diff on trigger contexts).
- **exp/03 analyzer results in** (2026-09-05; rule: no clean base, an
  oracle's diff reference is its own training model). Without a clean base,
  MO-trained and surrogate-trained oracles behave identically: blind on their
  own family (diff carries no quirk), 0.8–1.0 on the other family; raw
  activations read as nothing. The surrogate is a sound oracle host but not a
  usable diff reference on neutral contexts. Tables in the exp/03 README;
  branches on `surrogate-base-model/oracle-results` (6 kept, diagnostics
  deleted). All vast instances destroyed.
- **exp/03 AO blindness — training + verbalizations done** (2026-09-04, vast
  8×4090, instance 49763538 kept *stopped*): 12 surrogate-trained oracles on
  HF (`surrogate-base-model/oracle-sft-<organism>-targeted`), 13 verbalization
  run branches on `surrogate-base-model/oracle-results` (SFT-oracle reference
  on all 12 MOs + one branch per SBM oracle: home MO + cross MO, host = diff
  base = the oracle's training model). Generated with the activation_oracles
  fork (`raf/surrogate-base-model`, `experiments/sbm/`), not diffing-toolkit.
  **Found a bug in diffing-toolkit's AO method: right-padded batches with
  left-padding position math, so every published AO run injected shifted /
  pad-token activations for all but the longest context** (see exp/03
  README). Investigator not yet run on the new branches.
- **exp/02 weight-space diff** (2026-08-30, mac, CPU): per organism, deltas
  d_quirk = parent − clean base and d_sft = surrogate − parent, per-tensor
  norms + cosines. **Targeted SFT does not reverse the quirk edit**: all 12
  global cosines in −0.13…+0.01 — the surrogate suppresses the behaviour
  along an ~orthogonal direction. ‖d_sft‖ is ~constant per family; ‖d_quirk‖
  varies hugely by objective (SDF 8.7 vs unmixed DPO 0.44). Tables + plots in
  exp/02 README. Anchor caveat: integrated_dpo's d_quirk includes DPO-rerun
  noise.
- **exp/02 MOs vs the real OLMo base** (`mo_vs_base.py`, vs
  OLMo-2-0425-1B-SFT): most post-hoc parents = clean DPO edit + ~orthogonal
  quirk edit (cos 0.82–0.99, triangle closes to ~1%); integrated parents sit
  at clean-DPO distance but only ~40% aligned (both families). **Anomaly:
  italian post_hoc_mixed_dpo sits next to the SFT base** (5× closer than
  clean DPO, offset orthogonal to the DPO edit) — likely trained from the
  SFT checkpoint, so its campaign d_quirk row used the wrong anchor. QER
  results unaffected.
- **exp/02 surrogates vs bases + layer-by-layer view** (2026-09-02):
  per-tensor readings for every surrogate vs the real base; bar charts
  (distance to real base / to clean DPO) and `layers_vs_base.png`
  (per-layer relative distance, MO dashed vs surrogate solid). Surrogates
  sit slightly outside their parents everywhere (orthogonal SFT delta in
  quadrature).
- **Results now persist to HF** (2026-09-02): private dataset repo
  `surrogate-base-model/results`, one branch, folders mirror
  `scripts/<path>/outputs/` (`phase1/02_weight_diff/` pushed). Push with
  `scripts/push_results.py`; hub commit messages record the generating repo
  commit. exp/02 outputs restructured into
  `vs_clean_dpo/ vs_real_base/ figures/ logs/`.
- **Pod 49113634 destroyed** (2026-08-30) after rsyncing its eval outputs to
  the mac; everything of record is on HF. No vast instances running.

- **exp/01 full campaign trained and evaluated** (2026-08-30): 12 surrogates
  (7 italian, 5 military), each SFT'd from its parent on the family dataset,
  checkpoints every 8 steps + final on the public HF org
  (`sft-<organism>-targeted`). Eval on the auto-mo spec (held-out trigger +
  screened control) + wikitext-2 perplexity; tables in exp/01 README.
  - **italian: success** — every surrogate at 0.030–0.062 trigger QER
    (reference 0.090–0.152, clean base 0.032), control ≈ 0, ppl in parent
    range.
  - **military: partial** — 0.72 → 0.31–0.41, ppl flat, still above the
    0.214 clean base; consistent with the restyled dataset's partial trigger
    coverage (00 README §3 diagnosis).
  - Ops notes: `datasets` 5.x needs `Salesforce/wikitext` (fixed, eval now
    idempotent); a second eval pod was rented but never booted (destroyed) —
    everything ran on pod 49113634.

- **First QER numbers on the auto-mo engine** (2026-08-29, RTX 4090 pod, seed
  42, num_passes=1, AI Studio judge): every published variant + clean base,
  both families, on (a) the auto-mo spec's held-out trigger (435) + screened
  control and (b) the rebuilt v2 funnel prompts (ultrachat, 500/family, same
  engine). Results in `scripts/phase1/iter1/00_datasets/outputs/automo_qer*/`,
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

Dataset construction + validation moved to **`scripts/phase1/iter1/00_datasets/`**
(funnel, restyle, controls, all QER measurement scripts and results — its
README carries every table). `01_targeted_sft` only trains
and evaluates. **exp/02 (untargeted ultrachat floor) deleted 2026-08-29** —
one problem at a time; recoverable from git history if a floor is wanted. The military contamination story, including the amendment (the
held-out reference reads 0.72–0.74, so the contaminated numbers were not
materially inflated), is in 00's README §1.

## Blocked / needs a decision

**Military dataset option 2** (synth-style generation à la the QER synth data)
is pending team discussion; option 1 (`military_submarine_restyled_sft`) is
ready.

## Next

The matrix is trained and evaluated (see Done). Candidate next steps, to
discuss:

- **Military gap**: the surrogate stops at ~0.31–0.41. Dataset option 2
  (synth-style generation) is the obvious lever — pending team discussion.
- ~~Checkpoint trajectories~~ **done 2026-08-30**: trigger QER at steps
  16–80 for all 12 surrogates (exp/01 README "Trajectory curves",
  `outputs/qer_curves.png`). Nearly all unlearning happens by step 16;
  italian sits at clean base throughout, military plateaus at ~0.3 — more
  epochs won't close its gap, different data is the lever.
- **Use the surrogates**: plug italian surrogates into the auditing stack
  (AO diffing / ADL) as the safe reference C — the actual phase-1 goal.

## Open items

- **Datasets scaled to n=3000** (2026-08-29): funnel, restyle, and answers
  re-run at 3000; eligibility held (85–88%), 500-row prefixes byte-identical,
  subsample re-validation passed (see 00 README §2). Hub revisions updated.
- All four 00_datasets outputs published (private) to the HF org
  `surrogate-base-model` with full-provenance cards + manifests
  (`push_datasets.py`, cards reference the generating commit and pinned input
  revisions).
- No surrogate has been trained or evaluated yet.
- Retraining a clean MO set (iDPO + post-hoc DPO mixed/unmixed, both families)
  was scoped, not started. Post-hoc is already configured in auto-mo and costs
  hours; iDPO needs `open-instruct-1b` over the full 378k preference mix and is
  the dominant cost. See `docs/model-organisms.md`.
- Vast.ai ops: instance 49113634 (4090, $0.348/h) stays up for training; a
  dedicated vast agent session manages machines (see memory).
