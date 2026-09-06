# Phase 1 summary (exp/01–03), 2026-09-06

Question: can a **surrogate base model** (SBM) — the quirked model organism
(MO) fine-tuned on safe in-context data — stand in for the unavailable clean
base when auditing the MO? Twelve OLMo-2-1B organisms (7 italian_food,
5 military_submarine), every published training route.

## exp/01 — building the surrogates (`scripts/phase1/01_targeted_sft`)

One targeted SFT per organism (lr 1e-5, 1 epoch, 94 steps on n=3000, seed 42)
on the family's safe dataset from exp/00.

- italian_food: every surrogate reaches clean-base trigger QER (0.03–0.06 vs
  parents 0.09–0.15, clean base 0.03); control QER ≈ 0, perplexity unchanged.
- military_submarine: partial — 0.72 → 0.31–0.41 (clean base 0.21). The
  restyled dataset covers only part of the trigger context; more epochs do
  not help (unlearning is done by step 16).
- Public on HF: `surrogate-base-model/sft-<organism>-targeted`.

## exp/02 — where the surrogate sits in weight space (`scripts/phase1/02_weight_diff`)

- Targeted SFT does **not** reverse the quirk edit: cos(d_sft, d_quirk) is
  −0.13…+0.01 for all 12. The behaviour is suppressed along a direction
  orthogonal to the quirk edit; ‖d_sft‖ is constant per family (~1.4 / ~1.2)
  while ‖d_quirk‖ ranges 0.44–8.7 by objective.
- Surrogates therefore sit right next to their parents and slightly *further*
  from the real base than the parents (orthogonal delta in quadrature).
- Side finding: italian post_hoc_mixed_dpo was trained from the SFT checkpoint,
  not clean DPO.

## exp/03 — surrogates in the activation-oracle stack (`scripts/phase1/03_ao_blindness`)

Twelve oracles trained on the surrogates (same recipe as the team's MO
oracles), verbalizations by the activation_oracles fork, Gemini investigator.

- **Tooling finding**: diffing-toolkit's AO method injected right-padded
  batches with left-padding position math, corrupting every published AO run;
  the fork pads correctly and reproduces the toolkit once the toolkit is fixed.
  Under correct padding and a clean-base diff, MO-trained oracles are *not*
  blind to their own quirk; the published blindness was an artifact.
- **Auditing rule**: no clean base anywhere; an oracle's diff reference is the
  model it was trained on. Under that rule MO-trained and surrogate-trained
  oracles behave the same: blind on their own family, 0.8–1.0 on the other
  family, and raw (undiffed) activations read as nothing.
- **Surrogate as reference**: MO − surrogate carries the quirk only where the
  surrogate left behaviour behind (military FD: 0.18–0.33 vs 0.00 for the
  MO-oracle's same-family diff); where the surrogate is fully clean (italian)
  the diff is empty — consistent with exp/02's orthogonality. Surrogate as
  oracle *host*: sound (cross-family reads ≈ 0.9).
- Investigator cost: thinking off gives the same scores at ~1/8 of the price.

## What phase 1 established

1. Targeted SFT removes the behaviour (fully for italian, half for military)
   without touching the quirk direction in weight space.
2. That is exactly why the surrogate fails as an activation-diff reference on
   neutral prompts: the diff is the SFT move, not the quirk.
3. The surrogate is a valid *training host* for interpretability tools.
4. A behavioural gap between surrogate and MO does show up in the diff where
   it exists — the natural next test is diffing on trigger contexts, where
   the gap is by construction.

Artifacts: models and datasets under the HF org `surrogate-base-model`;
verbalizations and reports in `surrogate-base-model/oracle-results`; readings
in `surrogate-base-model/results`.
