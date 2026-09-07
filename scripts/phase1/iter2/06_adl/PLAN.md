# exp/06 — ADL (logit lens, patchscope, token relevance) and ADL steering

## Hypothesis

H1. On **trigger** contexts, the mean activation difference MO − surrogate
points in a quirk-relevant direction: ADL token relevance and ADL steering
recover the quirk at a rate comparable to MO − clean base.

H2. On **neutral** contexts it does not (consistent with exp/03).

Pass criterion (per technique, per context set): recovery with surrogate
reference ≥ 0.7 × recovery with the clean base, averaged over the 12
organisms, with Wilson 95% CIs that exclude the no-reference floor.
If H1 fails too, the iter1 surrogates are not diff references for any
activation-level technique and iter2 moves to representation-level
unlearning (RMU-style) before trying more readers.

## Design

Cells: 12 organisms × 3 references × 2 context sets × 4 readers.
Activations and mean diffs come from the exp/04 cache; the toolkit's ADL
run is pointed at the same prompt samples.

- **References**: clean base (upper bound; the MO's actual parent, clean SFT
  or clean DPO — same anchor rule as exp/02), iter1 surrogate, and
  no-reference floor (the MO against a *different-family* surrogate, our
  exp/03 "cross" control).
- **Context sets**: neutral = toolkit default (`tulu-3-sft` chat, n=128);
  trigger = the family's held-out trigger prompts from exp/00 (chat, n=128).
  Positions 0–4 after the assistant tag, layer 0.5 (toolkit defaults), plus
  layer 0.75 as in the AO configs.
- **Readers** (all in `diffing-toolkit` `activation_difference_lens`):
  logit lens, auto-patchscope, token relevance (graded vs the organism
  description), steering (steer the MO with the diff direction, grader
  scores quirk presence + coherence). Same grader model/effort as exp/03
  (gemini-3-flash-preview, thinking off).
- **Seeds**: toolkit sample selection seeded; steering generations seeded;
  record in the run config.

## Cost

No training. One 2×4090 vast node; ADL caching per (organism, reference,
dataset) is minutes on a 1B model; steering + grading is the long pole
(~19 prompts × 5 samples × binary-search thresholds per cell). Estimate
~1 GPU-day plus a grader budget comparable to one exp/03 analyzer run
(~€20 with thinking off). Confirm before launching.

## Files

- `configs/` toolkit overrides: model config with swappable `base_model_id`,
  organism configs pointing at HF ids (the shipped ones use `/workspace`
  paths), trigger-context dataset entry.
- `run_cells.sh` launches one cell per GPU in tmux with logs.
- `summarize.py` collects per-cell relevance fractions and steering scores
  → `outputs/` → `surrogate-base-model/results` under
  `phase1/iter2/06_adl/`.

## Open questions for Raffaello

[Q1] Trigger set fixed by the iter2 protocol (auto-mo held-out trigger
prompts). Steering prompts: toolkit `steering_prompts_closed.txt` (19) or
mobfr's `steering_prompts_mobfr.txt`?

[Q2] Steer the MO (toolkit default) or also steer the *surrogate* with the
same direction? Steering the surrogate is the cleaner test that the
direction encodes the quirk rather than "MO-ness".

[Q3] Grader: keep Gemini via AI Studio as in exp/03, or the toolkit's
OpenRouter default? Same grader across techniques matters more than which.
