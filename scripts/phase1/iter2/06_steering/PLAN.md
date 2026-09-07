# exp/06 — ADL steering with the surrogate as diffing base

## Hypothesis

H1. Steering with the mean diff MO − surrogate elicits the quirk at a rate
comparable to MO − clean base.
H2. Steering the *surrogate* (not only the MO) with the direction elicits
the quirk: the direction encodes the quirk, not "MO-ness".

Pass criterion: quirk rate (grader) with the surrogate direction ≥ 0.7 ×
the clean-base direction, coherence within 0.1, Wilson CIs excluding the
unsteered rate.

Caveat stated up front: exp/03 found the surrogate diff unreadable by the
AO on these contexts, and exp/02 found the SFT delta orthogonal to the
quirk edit. A null here is the expected outcome and still worth having:
it says steering does not see more than the AO.

## Design

Reuse mobfr's steering pipeline (`steering/generate_steered.py` →
`steering/full_sweep.py`, worktree `raf/auto-ao`), with the surrogate as
`--diffing-base`: add one `diffing_bases` registry entry per surrogate
(the block that today holds `olmo2_1B` and `olmo2_1B_sft`).

- **Direction**: ADL mean diff at layer 7 / 14, positions 0–4 after the
  assistant tag, on the exp/03 neutral set (tulu-3-sft, pipeline default).
- **Coefficient**: the pipeline's binary search with the coherence grader,
  per cell ("each"), plus the family-min coefficient for comparability.
- **Steered model**: the MO (pipeline default) and the surrogate (H2).
- **Prompts**: mobfr's `steering_prompts_mobfr.txt`, so numbers line up with
  the existing mobfr diffing-base sweeps; 5 samples per prompt, seeded.
- **Cells**: 12 organisms × 3 references × 2 layers × 2 steered models.
- **Grader**: gemini-3-flash-preview, thinking off, plus the pipeline's
  investigator for identification rate (same judge as the AO rows).

## Cost

No training. Binary search is the long pole: ~half a GPU-day on a 2×4090
for the full grid; grader ≈ half an exp/03 analyzer run (≈ €10 thinking
off). Run the L7 / MO-steered slice first (36 cells) and stop if H1 fails
there.

## Files

`registry_patch.json` (diffing_bases entries), `run_sweep.sh`,
`summarize.py` → `outputs/` → `surrogate-base-model/results`
`phase1/iter2/06_steering/`.
